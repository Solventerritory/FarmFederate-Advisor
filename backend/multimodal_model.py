#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multimodal_model.py — Roberta + ViT multimodal classifier for crop stress.

- Text encoder: roberta-base
- Image encoder: google/vit-base-patch16-224-in21k
- Projection of both to 256-d, concat → 512-d → MLP → 5 labels

This matches the architecture used in train_fed_multimodal.py (text+image
dataset). Checkpoints from that script can be loaded into this class.
"""

from typing import Optional

import torch
from torch import nn

from transformers import AutoModel, AutoTokenizer
from transformers import ViTModel, AutoImageProcessor


ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
NUM_LABELS = len(ISSUE_LABELS)


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        text_model_name: str = "roberta-base",
        image_model_name: str = "google/vit-base-patch16-224-in21k",
        num_labels: int = NUM_LABELS,
        freeze_backbones: bool = False,
    ):
        super().__init__()

        # ----- backbones -----
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        self.image_backbone = ViTModel.from_pretrained(image_model_name)

        if freeze_backbones:
            for p in self.text_backbone.parameters():
                p.requires_grad = False
            for p in self.image_backbone.parameters():
                p.requires_grad = False

        hidden_t = self.text_backbone.config.hidden_size
        hidden_i = self.image_backbone.config.hidden_size

        # ----- projections -----
        self.text_proj = nn.Linear(hidden_t, 256)
        self.image_proj = nn.Linear(hidden_i, 256)

        # ----- fusion head -----
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        input_ids:     [B, L]
        attention_mask:[B, L]
        pixel_values:  [B, 3, H, W]

        Returns
        -------
        logits: [B, num_labels]
        """
        # Roberta CLS
        t_out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        t_cls = t_out.last_hidden_state[:, 0]  # [B, Ht]

        # ViT CLS
        i_out = self.image_backbone(pixel_values=pixel_values)
        i_cls = i_out.last_hidden_state[:, 0]  # [B, Hi]

        t_feat = self.text_proj(t_cls)
        i_feat = self.image_proj(i_cls)

        fused = torch.cat([t_feat, i_feat], dim=-1)
        logits = self.fusion(fused)
        return logits


def build_tokenizer(model_name: str = "roberta-base"):
    return AutoTokenizer.from_pretrained(model_name)


def build_image_processor(model_name: str = "google/vit-base-patch16-224-in21k"):
    return AutoImageProcessor.from_pretrained(model_name)
