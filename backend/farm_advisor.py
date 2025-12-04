# backend/farm_advisor.py
# Light wrapper to satisfy server.py imports.
# Put this file in the same directory as server.py.

import os
import warnings
from typing import List, Optional
import numpy as np
import torch

# Try to import canonical implementations from existing modules.
try:
    # prefer the explicit multimodal training module if available
    from farm_advisor_multimodal_full import ISSUE_LABELS, MultiModalModel, build_tokenizer, apply_priors_to_logits  # type: ignore
except Exception:
    try:
        # fallback to multimodal_model (class name might differ)
        from multimodal_model import ISSUE_LABELS, MultimodalClassifier, build_tokenizer, build_image_processor  # type: ignore
        MultiModalModel = MultimodalClassifier  # alias for server.py
        # multimodal_model doesn't implement apply_priors_to_logits; use no-op below
        apply_priors_to_logits = None
    except Exception:
        # last-resort defaults
        ISSUE_LABELS = ["water_stress","nutrient_def","pest_risk","disease_risk","heat_stress"]
        MultiModalModel = None
        build_tokenizer = None
        apply_priors_to_logits = None

# If build_tokenizer was not found, provide a safe stub that raises a clear error
if build_tokenizer is None:
    def build_tokenizer(*args, **kwargs):
        raise RuntimeError("Tokenizer builder not found. Please ensure `build_tokenizer` exists in your training module.")
# If apply_priors_to_logits is missing, provide a safe no-op implementation
if apply_priors_to_logits is None:
    def apply_priors_to_logits(logits: torch.Tensor, texts: Optional[List[str]]):
        """
        No-op priors fallback. If you have a sensor_priors implementation in your training
        code, edit this function to import + use it instead (so server applies sensor priors).
        """
        # logits: Tensor [B, C] -> return unchanged
        return logits
