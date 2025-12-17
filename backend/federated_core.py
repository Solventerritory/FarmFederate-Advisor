#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
federated_core.py — core training utilities for multimodal FarmFederate.

Contains:
    - MultiModalDataset   (text + optional HF image dataset)
    - FocalLoss           (for multi-label)
    - helper functions: make_weights_for_balanced_classes, split_clients_dirichlet,
      train_one_client (used by train_fed_multimodal.py)
"""

import math
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import get_linear_schedule_with_warmup

from datasets_loader import ISSUE_LABELS, NUM_LABELS

SEED = 123
random.seed(SEED)
np.random.seed(SEED)


# ------------- Dataset -------------
class MultiModalDataset(Dataset):
    """
    Each item: {input_ids, attention_mask, pixel_values, labels, raw_text}
    - `df_text`: DataFrame with columns ["text", "labels"]
    - `tokenizer`: HF tokenizer for text encoder
    - `image_processor`: HF image processor for ViT
    - `image_ds`: optional HF dataset with column "image" (PIL-like)
    - If `image_ds` is None, uses a gray dummy image.
    """
    def __init__(self, df_text: pd.DataFrame, tokenizer, image_processor,
                 image_ds=None, max_len: int = 160):
        self.df = df_text.reset_index(drop=True)
        self.tok = tokenizer
        self.im_proc = image_processor
        self.image_ds = image_ds
        self.max_len = max_len

        # build a dummy image tensor once (3x224x224)
        from PIL import Image
        dummy = Image.new("RGB", (224, 224), (128, 128, 128))
        self._dummy_pixels = self.im_proc(dummy, return_tensors="pt")["pixel_values"][0]

    def __len__(self):
        return len(self.df)

    def _get_pixels(self, idx: int):
        if self.image_ds is None or len(self.image_ds) == 0:
            return self._dummy_pixels.clone()
        j = idx % len(self.image_ds)
        sample = self.image_ds[j]
        img = sample.get("image", None)
        if img is None:
            return self._dummy_pixels.clone()
        pixels = self.im_proc(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return pixels

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        text = row["text"]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in row["labels"]:
            if 0 <= k < NUM_LABELS:
                labels[k] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": self._get_pixels(i),
            "labels": labels,
            "raw_text": text,
        }


# ------------- Class balancing & loss -------------
def make_weights_for_balanced_classes(df: pd.DataFrame):
    counts = np.zeros(NUM_LABELS)
    for labs in df["labels"]:
        for k in labs:
            counts[k] += 1
    inv = 1.0 / np.maximum(counts, 1)
    inst_w = []
    for labs in df["labels"]:
        if labs:
            w = np.mean([inv[k] for k in labs])
        else:
            w = inv.mean()
        inst_w.append(w)
    inst_w = np.array(inst_w, dtype=np.float32)
    inst_w = inst_w / (inst_w.mean() + 1e-12)
    return inst_w, counts


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, label_smoothing: float = 0.02):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: [B, C]
        targets: [B, C] float in {0,1}
        """
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha.view(1, -1)
        return loss.mean()

# ------------- Dirichlet split + train_one_client -------------
def split_clients_dirichlet(df: pd.DataFrame, n_clients: int, alpha: float) -> List[pd.DataFrame]:
    """
    Non-IID client split: each label class has its own Dirichlet distribution over clients.
    A row's "primary" label selects which distribution to sample client from.
    """
    num_classes = NUM_LABELS
    rng = np.random.default_rng(SEED)

    prim = []
    for labs in df["labels"]:
        if labs:
            prim.append(int(rng.choice(labs)))
        else:
            prim.append(int(rng.integers(0, num_classes)))
    df2 = df.copy()
    df2["_y"] = prim

    class_client_probs = rng.dirichlet([alpha] * n_clients, size=num_classes)  # [C, n]

    client_bins = [[] for _ in range(n_clients)]
    for idx, y in enumerate(df2["_y"].tolist()):
        cli = int(rng.choice(n_clients, p=class_client_probs[y]))
        client_bins[cli].append(idx)

    out = []
    for k in range(n_clients):
        part = df2.iloc[client_bins[k]].drop(columns=["_y"]).reset_index(drop=True)
        out.append(part)
    return out


def train_one_client(
    model,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    tokenizer,
    image_processor,
    image_ds,
    max_len: int,
    batch_size: int,
    local_epochs: int,
    lr: float,
    device: str,
    grad_accum: int = 1,
    weight_decay: float = 0.01,
    alpha_per_class: Optional[torch.Tensor] = None,
):
    """
    Returns: (state_dict_cpu, train_loss, val_loss)
    (only the *trainable* parameters of the model are included in state_dict_cpu)
    """
    train_ds = MultiModalDataset(df_train, tokenizer, image_processor, image_ds, max_len)
    val_ds = MultiModalDataset(df_val, tokenizer, image_processor, image_ds, max_len)

    weights, counts = make_weights_for_balanced_classes(df_train)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=max(len(train_ds), batch_size),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(batch_size, 16), shuffle=False, num_workers=0
    )

    loss_fn = FocalLoss(alpha=alpha_per_class.to(device) if alpha_per_class is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum)))
    total_steps = local_epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps
    )

    model.to(device)
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad(set_to_none=True)

    total_loss = 0.0
    step_count = 0

    for epoch in range(local_epochs):
        for it, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = loss_fn(logits, labels) / max(1, grad_accum)

            scaler.scale(loss).backward()
            total_loss += loss.item()
            step_count += 1

            if it % max(1, grad_accum) == 0:
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

    # simple val loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = loss_fn(logits, labels)
            val_loss += loss.item() * labels.size(0)
    val_loss /= max(1, len(val_ds))

    # only return CPU state_dict to reduce memory
    state_dict_cpu = {k: v.detach().cpu()
                      for k, v in model.state_dict().items()
                      if v.requires_grad}

    return state_dict_cpu, total_loss / max(1, step_count), val_loss
