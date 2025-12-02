#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fed_multimodal.py — federated training for the multimodal FarmFederate model.

- Text data: mix of HF agri corpora + synthetic LocalMini
- Images: plantvillage (HF) if available
- FedAvg over non-IID clients (Dirichlet)
- Saves checkpoints/global_round{r}.pt (and final .pt)

This is a lighter, multimodal counterpart to farm_advisor.py adapted for
your backend + Flutter demo.
"""

import os
import math
from typing import List

import numpy as np
import pandas as pd
import torch

from multimodal_model import MultimodalClassifier, build_tokenizer, build_image_processor
from datasets_loader import (
    build_text_corpus_mix,
    load_plant_images_hf,
    summarize_labels,
    ISSUE_LABELS,
    NUM_LABELS,
)
from federated_core import split_clients_dirichlet, train_one_client, make_weights_for_balanced_classes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)


def fedavg_weighted(states: List[dict], sizes: List[int]):
    total = float(sum(sizes))
    weights = [s / total for s in sizes]
    out = {}
    keys = states[0].keys()
    for k in keys:
        accum = 0.0
        for st, w in zip(states, weights):
            accum = accum + st[k].float() * w
        out[k] = accum
    return out


def main():
    rounds = 2
    clients = 4
    local_epochs = 1
    batch_size = 16
    lr = 3e-5
    dirichlet_alpha = 0.25

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print("Building text corpus (mix of HF + LocalMini)...")
    df = build_text_corpus_mix(
        mix_sources="gardian,argilla,agnews,localmini",
        max_per_source=2000,
        max_samples=6000,
    )
    summarize_labels(df, "global")

    # train/val split
    df_train = df.sample(frac=0.85, random_state=SEED)
    df_val = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    # class weights for FocalLoss alpha
    _, counts = make_weights_for_balanced_classes(df_train)
    inv = 1.0 / np.maximum(counts, 1)
    alpha = (inv / inv.mean()).astype(np.float32)
    alpha = torch.tensor(alpha)

    # images
    image_ds = load_plant_images_hf(max_images=4000)

    # tokenizer / image processor
    text_model_name = "roberta-base"
    image_model_name = "google/vit-base-patch16-224-in21k"
    tokenizer = build_tokenizer(text_model_name)
    image_processor = build_image_processor(image_model_name)

    # global model
    global_model = MultimodalClassifier(
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        num_labels=NUM_LABELS,
        freeze_backbones=False,
    ).to(DEVICE)

    # federated client splits on train set
    client_dfs = split_clients_dirichlet(df_train, n_clients=clients, alpha=dirichlet_alpha)

    print(f"Non-IID clients: {[len(cdf) for cdf in client_dfs]}")

    for r in range(1, rounds + 1):
        print(f"\n=== Federated round {r}/{rounds} ===")
        states = []
        sizes = []

        for cid, cdf in enumerate(client_dfs):
            if len(cdf) < 100:
                print(f"[client {cid}] skipped (too small: n={len(cdf)})")
                continue

            # local train/val split for this client
            n = len(cdf)
            val_n = max(1, int(0.15 * n))
            df_c_val = cdf.iloc[:val_n].reset_index(drop=True)
            df_c_train = cdf.iloc[val_n:].reset_index(drop=True)

            # fresh model copy
            local_model = MultimodalClassifier(
                text_model_name=text_model_name,
                image_model_name=image_model_name,
                num_labels=NUM_LABELS,
                freeze_backbones=False,
            ).to(DEVICE)

            local_model.load_state_dict(global_model.state_dict(), strict=True)

            state_cpu, train_loss, val_loss = train_one_client(
                model=local_model,
                df_train=df_c_train,
                df_val=df_c_val,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_ds=image_ds,
                max_len=160,
                batch_size=batch_size,
                local_epochs=local_epochs,
                lr=lr,
                device=DEVICE,
                grad_accum=1,
                weight_decay=0.01,
                alpha_per_class=alpha,
            )

            print(
                f"[client {cid}] n={len(cdf)} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

            states.append(state_cpu)
            sizes.append(len(df_c_train))

            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not states:
            print("No client updates this round, skipping FedAvg.")
            continue

        # FedAvg update
        avg_state = fedavg_weighted(states, sizes)
        global_state = global_model.state_dict()
        for k, v in avg_state.items():
            if k in global_state:
                global_state[k] = v
        global_model.load_state_dict(global_state)

        ckpt_path = os.path.join(save_dir, f"global_round{r}.pt")
        torch.save(global_model.state_dict(), ckpt_path)
        print(f"[save] round {r} → {ckpt_path}")

    final_path = os.path.join(save_dir, "global_round_final.pt")
    torch.save(global_model.state_dict(), final_path)
    print(f"[save] final model → {final_path}")


if __name__ == "__main__":
    main()
