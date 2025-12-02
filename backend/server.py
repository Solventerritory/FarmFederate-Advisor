#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
server.py — FastAPI backend for FarmFederate multimodal advisor.

Uses:
- MultimodalClassifier (Roberta text + ViT image)
- Same issue labels as farm_advisor.py
- Sensor priors applied in logit space (scaled) at inference only
- Optional image input (file upload from Flutter)

Expected checkpoint (multimodal):
    checkpoints/global_round2.pt
produced by train_fed_multimodal.py (the script that used leaf images).

The schema matches the Flutter frontend:
    {
      "labels": [...],
      "probs": {"water_stress": 0.45, ...},
      "advice": "Recommended actions: ..."
    }
"""

import io
import os
import re
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

import torch
from torch.nn.functional import sigmoid

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from multimodal_model import (
    MultimodalClassifier,
    build_tokenizer,
    build_image_processor,
    ISSUE_LABELS,
    NUM_LABELS,
)

# ----------------- Config -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_MODEL_NAME = "roberta-base"
IMAGE_MODEL_NAME = "google/vit-base-patch16-224-in21k"
MAX_LEN = 160

# Path to the multimodal checkpoint trained with images
CHECKPOINT_PATH = os.path.join("checkpoints", "global_round2.pt")

# If you later save calibrated thresholds, put them here:
THRESHOLD_PATH = os.path.join("checkpoints", "thresholds.npy")

# Per-label default thresholds (if no thresholds.npy yet)
DEFAULT_THRESHOLDS = np.array([0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float32)

# Strength of sensor priors applied to logits
PRIOR_SCALE = 0.30

ADVICE = {
    "water_stress": (
        "Irrigate earlier in the day, add mulch to reduce evaporation, and "
        "monitor soil moisture in the morning and afternoon."
    ),
    "nutrient_def": (
        "Balance NPK (with emphasis on nitrogen if older leaves are yellow), "
        "apply appropriate fertilizer, and verify with Leaf Color Chart or SPAD if available."
    ),
    "pest_risk": (
        "Inspect the undersides of leaves, use sticky traps, remove heavily "
        "infested leaves, and consider early biocontrol or mild soap-based sprays."
    ),
    "disease_risk": (
        "Remove infected leaves, improve airflow (wider spacing / pruning), "
        "avoid late overhead irrigation, and consider preventive fungicides if disease pressure is high."
    ),
    "heat_stress": (
        "Provide shade during peak heat (shade nets or temporary cover), keep soil moisture "
        "stable with mulching, and ensure adequate potassium supply."
    ),
}

# ----------------- FastAPI setup -----------------
app = FastAPI(title="FarmFederate Multimodal Advisor Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Model + preprocessors -----------------
tokenizer = build_tokenizer(TEXT_MODEL_NAME)
image_processor = build_image_processor(IMAGE_MODEL_NAME)

model = MultimodalClassifier(
    text_model_name=TEXT_MODEL_NAME,
    image_model_name=IMAGE_MODEL_NAME,
    num_labels=NUM_LABELS,
    freeze_backbones=False,
)

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print("[server] Missing keys:", missing)
    if unexpected:
        print("[server]  Unexpected keys:", unexpected)
else:
    print(
        f"[server] WARNING: multimodal checkpoint not found at {CHECKPOINT_PATH}; "
        "using randomly initialized weights. Train with train_fed_multimodal.py."
    )

if os.path.exists(THRESHOLD_PATH):
    THRESHOLDS = np.load(THRESHOLD_PATH).astype(np.float32)
    if THRESHOLDS.shape[0] != NUM_LABELS:
        print("[server] thresholds.npy shape mismatch; falling back to defaults.")
        THRESHOLDS = DEFAULT_THRESHOLDS
else:
    THRESHOLDS = DEFAULT_THRESHOLDS

model.to(DEVICE)
model.eval()

# dummy image if user sends text-only
_dummy_image = Image.new("RGB", (224, 224), (128, 128, 128))
_dummy_pixels = image_processor(_dummy_image, return_tensors="pt")["pixel_values"].to(DEVICE)

# ----------------- Pydantic models -----------------
class PredictRequest(BaseModel):
    text: str
    sensors: Optional[str] = ""
    client_id: Optional[str] = "api_client"


class PredictResponse(BaseModel):
    labels: List[str]
    probs: Dict[str, float]
    advice: str


# ----------------- Sensor / priors utilities (ported from farm_advisor style) -----------------
# Expected sensor string inside SENSORS: ... e.g.
# "SENSORS: soil_moisture=24%, soil_pH=6.5, temp=35°C, humidity=68%, VPD=1.7 kPa, rainfall_24h=0.0mm (trend: ↑)."

_SENS_RE = re.compile(
    r"soil_moisture=(?P<sm>\d+(?:\.\d+)?)%.*?"
    r"soil_pH=(?P<ph>\d+(?:\.\d+)?).*?"
    r"temp=(?P<t>\d+(?:\.\d+)?)°C.*?"
    r"humidity=(?P<h>\d+(?:\.\d+)?)%.*?"
    r"VPD=(?P<vpd>\d+(?:\.\d+)?) kPa.*?"
    r"rainfall_24h=(?P<rf>\d+(?:\.\d+)?)mm",
    re.I | re.S,
)


def _parse_sensors(text: str) -> Optional[Dict[str, float]]:
    m = _SENS_RE.search(text)
    if not m:
        return None
    try:
        return dict(
            sm=float(m.group("sm")),
            ph=float(m.group("ph")),
            t=float(m.group("t")),
            h=float(m.group("h")),
            vpd=float(m.group("vpd")),
            rf=float(m.group("rf")),
        )
    except Exception:
        return None


def sensor_priors(text: str) -> np.ndarray:
    """
    Produce a small bias vector in R^NUM_LABELS based on parsed sensors.
    Logic is intentionally simple & noisy, matching farm_advisor behaviour.
    """
    b = np.zeros(NUM_LABELS, dtype=np.float32)
    s = _parse_sensors(text)
    if not s:
        return b

    sm, ph, t, h, vpd, rf = s["sm"], s["ph"], s["t"], s["h"], s["vpd"], s["rf"]

    # Sample heuristics (same flavour as farm_advisor)
    if sm >= 28 and vpd <= 1.2:
        b[0] -= 0.25  # less likely water stress
    if sm <= 18 or vpd >= 2.0:
        b[0] += 0.18  # more likely water stress
    if ph < 5.8 or ph > 7.4:
        b[1] += 0.12  # more nutrient def
    if 45 <= h <= 70 and rf <= 2.0:
        b[2] += 0.05  # pests
    if h >= 70 or rf >= 2.0:
        b[3] += 0.10  # diseases
    if h <= 45 and rf == 0 and vpd >= 2.0:
        b[3] -= 0.12  # less disease if very dry
    if t >= 36 or vpd >= 2.2:
        b[4] += 0.15  # heat stress
    if t <= 24:
        b[4] -= 0.15

    # small noise + 10% chance of "missing sensors"
    b = b + np.random.normal(0, 0.03, size=b.shape).astype(np.float32)
    if np.random.rand() < 0.10:
        b *= 0.0

    return b


def apply_priors_to_logits(logits: torch.Tensor, texts: Optional[List[str]]) -> torch.Tensor:
    if texts is None or PRIOR_SCALE <= 0:
        return logits
    biases = [
        torch.tensor(sensor_priors(t), dtype=logits.dtype, device=logits.device)
        for t in texts
    ]
    return logits + PRIOR_SCALE * torch.stack(biases, dim=0)


# ----------------- Text fusion & image encoding -----------------
def build_text_with_sensors(text: str, sensors: str) -> str:
    """
    Combine free-text description with simple sensors string.

    If sensors already begins with 'SENSORS:' we keep it,
    otherwise we prepend that label.
    """
    text = (text or "").strip()
    sensors = (sensors or "").strip()

    if sensors:
        if not sensors.lower().startswith("sensors"):
            sensors_block = f"SENSORS: {sensors}"
        else:
            sensors_block = sensors
        return f"{sensors_block}\nLOG: {text or '(no log text)'}"
    else:
        # if no explicit sensors passed, treat text as the log only
        return f"LOG: {text or 'plant status report'}"


def encode_text(text_full: str):
    enc = tokenizer(
        text_full,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def encode_image(upload: Optional[UploadFile]) -> torch.Tensor:
    if upload is None:
        return _dummy_pixels
    try:
        data = upload.file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        pv = image_processor(img, return_tensors="pt")["pixel_values"].to(DEVICE)
        return pv
    except Exception as e:
        print(f"[server] image decode error: {e}")
        return _dummy_pixels


def apply_thresholds(probs: np.ndarray, thr: np.ndarray) -> List[str]:
    mask = probs >= thr
    labels = [lab for lab, m in zip(ISSUE_LABELS, mask) if m]
    if not labels:
        # guarantee at least top-1
        top_idx = int(probs.argmax())
        labels = [ISSUE_LABELS[top_idx]]
    return labels


def build_advice(labels: List[str]) -> str:
    if not labels:
        return (
            "Conditions look mostly normal based on the model. "
            "Continue routine monitoring and adjust irrigation and nutrition as usual."
        )
    lines = ["Recommended actions:"]
    for lab in labels:
        tip = ADVICE.get(lab)
        if tip:
            lines.append(f"- {lab}: {tip}")
    return "\n".join(lines)


# ----------------- Routes -----------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    text: str = Form(default=""),
    sensors: str = Form(default=""),
    client_id: str = Form(default="flutter_client"),
    image: UploadFile | None = File(default=None),
):
    """
    Main endpoint used by Flutter.

    Supports multipart/form-data (with or without image).
    - text: description typed by farmer/user
    - sensors: free string or 'SENSORS: ...'
    - image: optional leaf/plant photo
    """

    # 1) fuse text + sensors similar to farm_advisor
    text_full = build_text_with_sensors(text, sensors)

    # 2) tokenize + preprocess image
    input_ids, attention_mask = encode_text(text_full)
    pixel_values = encode_image(image)

    # 3) forward pass + priors
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        logits = apply_priors_to_logits(logits, [text_full])
        probs_t = sigmoid(logits)[0].cpu().numpy()  # [C]

    # 4) thresholding + advice
    labels = apply_thresholds(probs_t, THRESHOLDS)
    advice = build_advice(labels)
    probs_dict = {lab: float(p) for lab, p in zip(ISSUE_LABELS, probs_t)}

    return PredictResponse(labels=labels, probs=probs_dict, advice=advice)


# Optional pure-JSON endpoint (no image); handy for curl testing.
@app.post("/predict_json", response_model=PredictResponse)
async def predict_json(req: PredictRequest):
    text_full = build_text_with_sensors(req.text, req.sensors or "")
    input_ids, attention_mask = encode_text(text_full)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=_dummy_pixels,
        )
        logits = apply_priors_to_logits(logits, [text_full])
        probs_t = sigmoid(logits)[0].cpu().numpy()

    labels = apply_thresholds(probs_t, THRESHOLDS)
    advice = build_advice(labels)
    probs_dict = {lab: float(p) for lab, p in zip(ISSUE_LABELS, probs_t)}

    return PredictResponse(labels=labels, probs=probs_dict, advice=advice)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
