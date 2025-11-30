#!/usr/bin/env python3
# multimodal_train.py — simple multimodal trainer: ViT (images) + RoBERTa (text) -> small fusion head
# Requirements: transformers, datasets, accelerate, torch, torchvision, peft, timm, pillow

import os
import math
import random
import argparse
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    ViTImageProcessor, ViTModel,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# labels
ISSUE_LABELS = ["water_stress","nutrient_def","pest_risk","disease_risk","heat_stress"]
NUM_LABELS = len(ISSUE_LABELS)

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="CSV with image_path,text,labels (comma ints)")
    p.add_argument("--image_root", type=str, default=".", help="prefix for image_path")
    p.add_argument("--vit_name", type=str, default="google/vit-base-patch16-224-in21k")
    p.add_argument("--text_model", type=str, default="roberta-base")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save_dir", type=str, default="checkpoints_paper")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--freeze_text_base", action="store_true")
    p.add_argument("--freeze_vit", action="store_false", dest="freeze_vit")  # default False -> do not freeze
    p.set_defaults(freeze_vit=False)
    return p.parse_args()

class MultimodalDataset(Dataset):
    def __init__(self, df, image_root, img_processor, tokenizer, max_len=128, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.img_processor = img_processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transforms = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_path"]) if not os.path.isabs(row["image_path"]) else row["image_path"]
        # load image
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        else:
            # fallback: resize+center crop to processor size
            img = img.resize((self.img_processor.size["shortest_edge"],)*2)
        # process text
        text = str(row["text"])
        tok = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        # labels -> multi-hot vector
        labs = row.get("labels", "")
        if isinstance(labs, str):
            labs = [int(x) for x in labs.split(",") if x.strip() != ""]
        y = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in labs:
            if 0 <= int(k) < NUM_LABELS:
                y[int(k)] = 1.0
        return {
            "pixel_values": img,  # if transforms returns tensor, else handled in collate
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "labels": y
        }

def collate_fn(batch):
    # pixel_values likely already tensors (from transforms.ToTensor)
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class FusionModel(nn.Module):
    def __init__(self, vit_model, text_model, hidden_dim=512, freeze_vit=False, freeze_text=False, lora_cfg=None):
        super().__init__()
        self.vit = vit_model
        self.text = text_model

        # get dims
        # viT outputs last_hidden_state and pooler / cls token
        vit_dim = self.vit.config.hidden_size
        text_dim = self.text.config.hidden_size

        # optional LoRA already applied to text_model externally if using peft
        self.fusion = nn.Sequential(
            nn.Linear(vit_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_LABELS)
        )

        if freeze_vit:
            for p in self.vit.parameters(): p.requires_grad = False
        if freeze_text:
            for p in self.text.parameters(): p.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        # ViT forward: get pooled representation
        vit_out = self.vit(pixel_values=pixel_values)
        # Get CLS token representation if available or pooler
        if hasattr(vit_out, "pooler_output") and vit_out.pooler_output is not None:
            v = vit_out.pooler_output
        else:
            # use last_hidden_state[:,0,:]
            v = vit_out.last_hidden_state[:, 0, :]

        # Text forward (take CLS / pooled output)
        txt_out = self.text(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(txt_out, "pooler_output") and txt_out.pooler_output is not None:
            t = txt_out.pooler_output
        else:
            # some transformers (like roberta) may not have pooler — use first token
            t = txt_out.last_hidden_state[:, 0, :]

        # concat and predict
        cat = torch.cat([v, t], dim=1)
        logits = self.fusion(cat)
        return logits

def train_epoch(model, loader, opt, device, scheduler=None):
    model.train()
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    for b in tqdm(loader, desc="train"):
        pixel = b["pixel_values"].to(device)
        input_ids = b["input_ids"].to(device)
        att = b["attention_mask"].to(device)
        labels = b["labels"].to(device)
        logits = model(pixel, input_ids, att)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler: scheduler.step()
        total_loss += float(loss.item()) * labels.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    tot_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for b in loader:
            pixel = b["pixel_values"].to(device)
            input_ids = b["input_ids"].to(device)
            att = b["attention_mask"].to(device)
            labels = b["labels"].to(device)
            logits = model(pixel, input_ids, att)
            loss = loss_fn(logits, labels)
            tot_loss += float(loss.item()) * labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    # compute simple micro-F1 via threshold 0.5
    preds = (torch.sigmoid(logits) >= 0.5).int().numpy()
    labs = labels.int().numpy()
    from sklearn.metrics import f1_score
    micro = f1_score(labs, preds, average="micro", zero_division=0)
    return tot_loss / len(loader.dataset), micro

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    # require columns image_path,text,labels
    assert {"image_path","text","labels"}.issubset(set(df.columns)), "CSV must have image_path,text,labels"

    # split
    train_df = df.sample(frac=0.9, random_state=SEED)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    device = torch.device(args.device)

    # processors / tokenizers
    print("[INFO] Loading image processor and tokenizer...")
    img_processor = ViTImageProcessor.from_pretrained(args.vit_name)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    # models (base)
    print("[INFO] Loading ViT and text models (feature extractors)...")
    vit = ViTModel.from_pretrained(args.vit_name)
    text_config = AutoConfig.from_pretrained(args.text_model)
    text = AutoModel.from_config(text_config)
    # load pretrained text base weights into AutoModel for representations
    # we prefer pre-trained weights:
    try:
        from transformers import AutoModelForMaskedLM
        tmp = AutoModel.from_pretrained(args.text_model)
        text = tmp
    except Exception as e:
        print("[WARN] couldn't load pretrained text weights directly:", e)

    # Apply LoRA to text model (PEFT) — only if not freezing text base
    if args.lora_r > 0:
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, task_type="SEQ_CLS", target_modules=["query","key","value","dense"])
        print("[INFO] Applying LoRA to text model...")
        text = get_peft_model(text, lcfg)

    model = FusionModel(vit, text, freeze_vit=args.freeze_vit, freeze_text=args.freeze_text_base)
    model.to(device)

    # transforms: use torchvision transforms -> tensor & normalize using ViT default values
    mean = img_processor.image_mean if hasattr(img_processor, "image_mean") else [0.5,0.5,0.5]
    std = img_processor.image_std if hasattr(img_processor, "image_std") else [0.5,0.5,0.5]
    tfrm = transforms.Compose([
        transforms.Resize((img_processor.size["height"], img_processor.size["width"]) if isinstance(img_processor.size, dict) else (224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_ds = MultimodalDataset(train_df, args.image_root, img_processor, tokenizer, max_len=args.max_len, transforms=tfrm)
    val_ds = MultimodalDataset(val_df, args.image_root, img_processor, tokenizer, max_len=args.max_len, transforms=tfrm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # optimizer includes only trainable params (LoRA + fusion head + optionally unfrozen vit/text params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, int(0.05*total_steps)), num_training_steps=total_steps)

    print(f"[TRAIN] epochs={args.epochs} steps={total_steps} bs={args.batch_size}")

    best_micro = -1.0
    for epoch in range(1, args.epochs+1):
        print(f"== Epoch {epoch}/{args.epochs} ==")
        tr_loss = train_epoch(model, train_loader, opt, device, scheduler=scheduler)
        val_loss, micro = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} micro_f1={micro:.4f}")
        # save best
        if micro > best_micro:
            best_micro = micro
            outp = os.path.join(args.save_dir, "multimodal_best.pt")
            # save model state dict (cpu)
            torch.save(model.state_dict(), outp)
            print("[SAVED]", outp)

    # also attempt to save peft adapters if present
    try:
        from peft import get_peft_model_state_dict
        sd = get_peft_model_state_dict(model.text)
        torch.save(sd, os.path.join(args.save_dir, "text_lora_adapters.pt"))
        print("[SAVED] text_lora_adapters.pt")
    except Exception as e:
        print("[INFO] no peft adapters saved:", e)

if __name__ == "__main__":
    main()
