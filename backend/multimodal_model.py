#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multimodal_model.py — Roberta + ViT multimodal classifier for crop stress.

- Text encoder: roberta-base
- Image encoder: google/vit-base-patch16-224-in21k
- Projection of both to 256-d, concat → 512-d → MLP → 5 labels

This file provides:
 - class MultiModalModel : the server expects this name.
 - alias MultimodalClassifier -> MultiModalModel for compatibility.
 - .text_encoder, .vision_encoder attributes (used by PEFT / adapter loading).
 - .classifier attribute (used by server fallback).
 - forward(...) accepts image=None (text-only), returns object with .logits
 - helper methods: forward_text_image(...) and predict(...)
"""

from typing import Optional, Tuple

import torch
from torch import nn
from types import SimpleNamespace

from transformers import AutoModel, AutoTokenizer
from transformers import ViTModel, AutoImageProcessor

ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
NUM_LABELS = len(ISSUE_LABELS)


class MultiModalModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = "roberta-base",
        image_model_name: str = "google/vit-base-patch16-224-in21k",
        num_labels: int = NUM_LABELS,
        freeze_backbones: bool = False,
    ):
        """
        Args
        - text_model_name: HuggingFace name for text encoder (AutoModel)
        - image_model_name: HuggingFace name for ViT encoder
        - freeze_backbones: if True, freeze encoder weights (useful for LoRA)
        """
        super().__init__()

        # ----- backbones -----
        # Keep attribute names expected by external code / peft utilities:
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.vision_encoder = ViTModel.from_pretrained(image_model_name)

        if freeze_backbones:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        hidden_t = self.text_encoder.config.hidden_size
        hidden_i = self.vision_encoder.config.hidden_size

        # ----- projections -----
        self.text_proj = nn.Linear(hidden_t, 256)
        self.image_proj = nn.Linear(hidden_i, 256)

        # ----- fusion head -----
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # final classifier head (exposed as `.classifier` to help server fallback)
        self.classifier = nn.Linear(512, num_labels)

        # convenience alias for backwards compatibility
        # some code expects .text_backbone or .image_backbone names
        self.text_backbone = self.text_encoder
        self.image_backbone = self.vision_encoder

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Return pooled text features [B, hidden_t] (CLS pooling).
        """
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # use pooler_output if present (some models), else CLS token
        if hasattr(t_out, "pooler_output") and t_out.pooler_output is not None:
            t_cls = t_out.pooler_output
        else:
            # last_hidden_state [B, L, H]
            t_cls = t_out.last_hidden_state[:, 0, :]
        return t_cls

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Return pooled image features [B, hidden_i] (CLS pooling).
        """
        i_out = self.vision_encoder(pixel_values=pixel_values)
        if hasattr(i_out, "pooler_output") and i_out.pooler_output is not None:
            i_cls = i_out.pooler_output
        else:
            i_cls = i_out.last_hidden_state[:, 0, :]
        return i_cls

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> SimpleNamespace:
        """
        Standard forward used by server code.

        Args:
          input_ids: [B, L]
          attention_mask: [B, L]
          pixel_values: [B, 3, H, W] or None (text-only)

        Returns:
          SimpleNamespace with attribute `logits` (Tensor [B, num_labels])
        """
        # Text features
        t_cls = self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)  # [B, Ht]
        t_feat = self.text_proj(t_cls)  # [B, 256]

        # Image features (if present)
        if pixel_values is not None:
            i_cls = self.get_image_features(pixel_values=pixel_values)  # [B, Hi]
            i_feat = self.image_proj(i_cls)  # [B, 256]
        else:
            # if no image provided, use zeros for image feature (learned neutrality)
            device = t_feat.device
            i_feat = torch.zeros_like(t_feat, device=device)  # shape [B, 256]

        # fuse
        fused_small = torch.cat([t_feat, i_feat], dim=-1)  # [B, 512]
        fused = self.fusion(fused_small)  # [B, 512]
        logits = self.classifier(fused)  # [B, num_labels]

        return SimpleNamespace(logits=logits)

    # explicit helper used by server fallback
    def forward_text_image(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: Optional[torch.Tensor] = None,
    ) -> SimpleNamespace:
        """
        Same as forward but matches server naming used in fallback attempts.
        Accepts `image` (not pixel_values) for convenience.
        """
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)

    # convenience predict wrapper (tokenize outside typically)
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return logits Tensor [B, num_labels] directly (no namespace).
        """
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)
        return out.logits


# Provide original name as alias for compatibility (if some code imports that)
MultimodalClassifier = MultiModalModel


# Tokenizer / image processor helpers (kept for convenience)
def build_tokenizer(model_name: str = "roberta-base"):
    """
    Returns AutoTokenizer (kept signature similar to earlier code).
    """
    return AutoTokenizer.from_pretrained(model_name)


def build_image_processor(model_name: str = "google/vit-base-patch16-224-in21k"):
    """
    Returns AutoImageProcessor for ViT preprocessing.
    """
    return AutoImageProcessor.from_pretrained(model_name)
