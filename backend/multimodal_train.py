#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multimodal_train.py — centralized training of the multimodal model.

- Builds a mixed text corpus from HF datasets + synthetic logs.
- Uses multiple plant-stress image datasets via Hugging Face
  (merged by load_stress_image_datasets_hf).
- Trains MultimodalClassifier end-to-end (no federated splitting).
- Saves model weights to checkpoints/global_central.pt
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from multimodal_model import MultimodalClassifier, build_image_processor
from datasets_loader import (
    build_text_corpus_mix,
    load_stress_image_datasets_hf,
    summarize_labels,
    ISSUE_LABELS,
    NUM_LABELS,
)
from federated_core import MultiModalDataset, make_weights_for_balanced_classes, FocalLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 1) text corpus
    df = build_text_corpus_mix(
        mix_sources="gardian,argilla,agnews,localmini",
        max_per_source=2000,
        max_samples=6000,
    )
    summarize_labels(df, "train")

    # 2) images: merged HF plant stress datasets (auto-downloadable)
    image_ds = load_stress_image_datasets_hf(
        max_total_images=20000,
        max_per_dataset=6000,
    )

    # 3) model + tokenizers
    text_model_name = "roberta-base"
    image_model_name = "google/vit-base-patch16-224-in21k"
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    image_processor = build_image_processor(image_model_name)

    model = MultimodalClassifier(
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        num_labels=NUM_LABELS,
        freeze_backbones=False,
    ).to(DEVICE)

    # 4) dataset / dataloader
    ds = MultiModalDataset(df, tokenizer, image_processor, image_ds, max_len=160)
    weights, counts = make_weights_for_balanced_classes(df)

    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    # 5) loss + optim
    inv = 1.0 / np.maximum(counts, 1)
    alpha = (inv / inv.mean()).astype(np.float32)
    alpha = torch.tensor(alpha)

    loss_fn = FocalLoss(alpha=alpha.to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    print(f"Training on {len(ds)} samples, device={DEVICE}")

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"[epoch {epoch+1}] loss={avg_loss:.4f}")

    ckpt_path = os.path.join(save_dir, "global_central.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[save] model → {ckpt_path}")


if __name__ == "__main__":
    main()
