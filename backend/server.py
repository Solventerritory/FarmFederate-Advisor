#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
server.py — FastAPI backend for FarmFederate-Advisor (multimodal RoBERTa + ViT + LoRA).

Features
--------
- Loads the same MultiModalModel architecture as in farm_advisor.py
- Loads LoRA text adapters from:   <SAVE_DIR>/global_lora_text.pt
- Loads thresholds from:           <SAVE_DIR>/thresholds.npy
- Single /predict endpoint that accepts:
    • JSON:    { "text": "...", "sensors": "...", "client_id": "..." }
    • Multipart: fields text, sensors, client_id + optional file field "image"
- Optional image: ViT-based branch (google/vit-base-patch16-224-in21k)
- Optional sensor priors: if you pass a "SENSORS: ..." line, priors are applied.
"""

import os
import io
import json
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------
# Import the model + tokenizer + priors + labels from your training file
# ---------------------------------------------------------------------
from farm_advisor import (
    ISSUE_LABELS,
    MultiModalModel,
    build_tokenizer,
    apply_priors_to_logits,
)

# ----------------- Config -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory where farm_advisor.py saved the final adapters & thresholds
SAVE_DIR = os.environ.get("SAVE_DIR", "checkpoints_paper")
ADAPTER_PATH = os.path.join(SAVE_DIR, "global_lora_text.pt")
THR_PATH = os.path.join(SAVE_DIR, "thresholds.npy")

TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL_NAME", "roberta-base")
VIT_MODEL_NAME = os.environ.get("VIT_MODEL_NAME", "google/vit-base-patch16-224-in21k")
FREEZE_TEXT = True      # matches farm_advisor default (LoRA on frozen base)
FREEZE_VISION = False   # usually fine for inference, we aren't training here
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LEN = 160
IMG_SIZE = 224

# ----------------- Globals -----------------
app = FastAPI(title="FarmFederate-Advisor Backend", version="1.0.0")

# Allow dev from anywhere; tighten in production if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TOKENIZER = None
MODEL: Optional[MultiModalModel] = None
THRESHOLDS: Optional[np.ndarray] = None

IMAGE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


# ----------------- Init / load model -----------------
def load_model() -> None:
    global TOKENIZER, MODEL, THRESHOLDS

    # 1) tokenizer
    TOKENIZER = build_tokenizer(TEXT_MODEL_NAME)

    # 2) model architecture (same as training)
    MODEL = MultiModalModel(
        text_model_name=TEXT_MODEL_NAME,
        vit_name=VIT_MODEL_NAME,
        num_labels=len(ISSUE_LABELS),
        freeze_text=FREEZE_TEXT,
        freeze_vision=FREEZE_VISION,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    ).to(DEVICE)
    MODEL.eval()

    # 3) load text LoRA adapters
    if os.path.exists(ADAPTER_PATH):
        sd = torch.load(ADAPTER_PATH, map_location="cpu")
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(MODEL.text_encoder, sd)
        print(f"[server] Loaded text adapters from {ADAPTER_PATH}")
    else:
        print(f"[server][WARN] LoRA adapter file not found: {ADAPTER_PATH}")
        print("         Using randomly-initialized adapters (for testing only).")

    # 4) thresholds
    if os.path.exists(THR_PATH):
        THRESHOLDS = np.load(THR_PATH)
        print(f"[server] Loaded thresholds from {THR_PATH}: {THRESHOLDS}")
    else:
        THRESHOLDS = np.array([0.5] * len(ISSUE_LABELS), dtype=np.float32)
        print(f"[server][WARN] Threshold file not found: {THR_PATH}")
        print("         Falling back to 0.5 for all labels.")


@app.on_event("startup")
async def _startup_event():
    load_model()


# ----------------- Helpers -----------------
def preprocess_image(upload: Any) -> Optional[torch.Tensor]:
    """
    upload: starlette.datastructures.UploadFile from request.form()["image"]

    Returns:
        torch.FloatTensor of shape [3, H, W] or None if not valid.
    """
    if upload is None:
        return None
    try:
        # read file into PIL
        content = upload.file.read()
        upload.file.seek(0)
        img = Image.open(io.BytesIO(content)).convert("RGB")
        tensor = IMAGE_TRANSFORM(img)  # [3,H,W]
        return tensor
    except Exception as e:
        print("[server][WARN] Failed to load image:", e)
        return None


def build_text_with_sensors(text: str, sensors: str) -> str:
    """
    Build the combined text passed into the model.

    We don't invent fake sensors here — if the client does not send
    sensors, we just add a neutral line so the format matches training-ish.

    For *priors* to actually kick in, sensors should be a string like:
    'SENSORS: soil_moisture=20%, soil_pH=6.0, temp=35°C, humidity=40%, VPD=2.1 kPa, rainfall_24h=0.0mm'
    (same format as in farm_advisor.py)
    """
    text = (text or "").strip()
    sensors = (sensors or "").strip()

    if sensors:
        # If user already sent "SENSORS: ..." keep it; else add prefix.
        if sensors.upper().startswith("SENSORS:"):
            sensors_line = sensors
        else:
            sensors_line = "SENSORS: " + sensors
    else:
        sensors_line = "SENSORS: (not provided)."

    if not text:
        text = "(no free-text log)."

    # The training pipeline typically used:
    #   SENSORS: ...
    #   LOG: ...
    return f"{sensors_line}\nLOG: {text}"


def logits_to_output(text: str,
                     logits: torch.Tensor,
                     thresholds: np.ndarray) -> Dict[str, Any]:
    """
    Apply priors + sigmoid + thresholds, return nice dict for JSON.
    """
    # logits: [1, num_labels]
    # Apply priors based on the combined text.
    logits = apply_priors_to_logits(logits, [text])  # [1, C]
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]  # [C]

    thr = thresholds.astype(np.float32)
    mask = (probs >= thr).astype(int)

    labels = []
    for i, v in enumerate(mask):
        if v == 1:
            labels.append({
                "label": ISSUE_LABELS[i],
                "prob": float(probs[i]),
                "threshold": float(thr[i]),
            })

    # Also send all scores for UI debugging
    all_scores = [
        {"label": ISSUE_LABELS[i], "prob": float(probs[i]), "threshold": float(thr[i])}
        for i in range(len(ISSUE_LABELS))
    ]

    return {
        "active_labels": labels,
        "all_scores": all_scores,
    }


# ----------------- Routes -----------------
@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "labels": ISSUE_LABELS}


@app.post("/predict")
async def predict(request: Request):
    """
    Single endpoint that accepts either:
      - Content-Type: application/json
        body: { "text": "...", "sensors": "...", "client_id": "flutter_client" }

      - Content-Type: multipart/form-data
        fields:
          text: str
          sensors: str (optional)
          client_id: str (optional)
          image: file (optional)
    """
    global MODEL, TOKENIZER, THRESHOLDS
    if MODEL is None or TOKENIZER is None or THRESHOLDS is None:
        # safety net
        load_model()

    content_type = request.headers.get("content-type", "").lower()

    text = ""
    sensors = ""
    client_id = "flutter_client"
    image_tensor = None

    # ---------- JSON case ----------
    if "application/json" in content_type:
        data = await request.json()
        if not isinstance(data, dict):
            return JSONResponse({"error": "JSON body must be an object"}, status_code=400)

        text = str(data.get("text", "") or "")
        sensors = str(data.get("sensors", "") or "")
        client_id = str(data.get("client_id", "flutter_client") or "")

    # ---------- Multipart / form-data case ----------
    elif "multipart/form-data" in content_type:
        form = await request.form()
        text = str(form.get("text", "") or "")
        sensors = str(form.get("sensors", "") or "")
        client_id = str(form.get("client_id", "flutter_client") or "")

        upload = form.get("image", None)
        if upload is not None:
            image_tensor = preprocess_image(upload)

    else:
        return JSONResponse(
            {"error": f"Unsupported Content-Type: {content_type}"},
            status_code=415,
        )

    if not text and not sensors:
        return JSONResponse({"error": "At least 'text' or 'sensors' must be provided."}, status_code=400)

    combined_text = build_text_with_sensors(text, sensors)

    # Tokenize text
    enc = TOKENIZER(
        combined_text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # Prepare image batch [B,3,H,W] if available
    if image_tensor is not None:
        image_batch = image_tensor.unsqueeze(0).to(DEVICE)
    else:
        image_batch = None

    # Forward
    with torch.no_grad():
        if image_batch is not None:
            out = MODEL(input_ids=input_ids,
                        attention_mask=attention_mask,
                        image=image_batch)
        else:
            out = MODEL(input_ids=input_ids,
                        attention_mask=attention_mask,
                        image=None)
        logits = out.logits  # [1, C]

    result = logits_to_output(combined_text, logits, THRESHOLDS)

    return {
        "client_id": client_id,
        "text_used": combined_text,
        "result": result,
    }


# ----------------- Entry point (for `python server.py`) -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
