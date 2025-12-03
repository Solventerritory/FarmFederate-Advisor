#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — FastAPI backend for FarmFederate-Advisor (multimodal RoBERTa + ViT + LoRA).

Extended features added:
 - /predict (original): JSON or multipart (text + optional image) → performs inference
 - /sensor_upload (POST JSON)                     : accepts sensor telemetry from ESP32 nodes
 - /image_upload  (POST multipart/form-data)      : accepts image uploads (esp32cam/pi) + optional log
 - /publish_command (POST JSON)                   : allow server/frontend to publish an MQTT command to device
 - /upload_adapter (POST multipart/form-data)     : accepts LoRA adapter file from a client (simulate client update)
 - /list_pending_for_training (GET)               : lists saved images/sensors queued for offline training
 - saves uploads under <SAVE_DIR>/ingest/{sensors,images,adapters}
 - publishes MQTT commands to topic farm/<device_id>/commands
 - stores minimal device status info in memory (for quick UI)
"""

import os
import io
import json
import shutil
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# MQTT publish
import paho.mqtt.publish as mqtt_publish

# ---------------------------------------------------------------------
# Import the model + tokenizer + priors + labels from your training file
# ---------------------------------------------------------------------
# NOTE: adapt these imports if your module names differ. I expect
# farm_advisor.py (or farm_advisor_multimodal_full.py) to expose:
#   ISSUE_LABELS, MultiModalModel, build_tokenizer, apply_priors_to_logits
try:
    from farm_advisor import (
        ISSUE_LABELS,
        MultiModalModel,
        build_tokenizer,
        apply_priors_to_logits,
    )
except Exception:
    # fallback: keep names for typing, but server will still run without model.
    ISSUE_LABELS = ["water_stress","nutrient_def","pest_risk","disease_risk","heat_stress"]
    MultiModalModel = None
    build_tokenizer = None
    def apply_priors_to_logits(logits, texts): return logits

# ----------------- Config -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = os.environ.get("SAVE_DIR", "checkpoints_paper")
INGEST_DIR = os.path.join(SAVE_DIR, "ingest")
IMAGES_DIR = os.path.join(INGEST_DIR, "images")
SENSORS_DIR = os.path.join(INGEST_DIR, "sensors")
ADAPTERS_DIR = os.path.join(INGEST_DIR, "adapters")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SENSORS_DIR, exist_ok=True)
os.makedirs(ADAPTERS_DIR, exist_ok=True)

ADAPTER_PATH = os.path.join(SAVE_DIR, "global_lora_text.pt")
THR_PATH = os.path.join(SAVE_DIR, "thresholds.npy")

TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL_NAME", "roberta-base")
VIT_MODEL_NAME = os.environ.get("VIT_MODEL_NAME", "google/vit-base-patch16-224-in21k")
FREEZE_TEXT = True
FREEZE_VISION = False
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LEN = 160
IMG_SIZE = 224

# MQTT broker
MQTT_BROKER = os.environ.get("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

# ----------------- Globals -----------------
app = FastAPI(title="FarmFederate-Advisor Backend", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# transforms
IMAGE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# runtime holders
TOKENIZER = None
MODEL: Optional[nn.Module] = None
THRESHOLDS: Optional[np.ndarray] = None

# simple in-memory device status (can be persisted later)
DEVICE_STATUS: Dict[str, Dict[str, Any]] = {}

# ----------------- Model loading -----------------
def load_model() -> None:
    global TOKENIZER, MODEL, THRESHOLDS
    print("[server] Loading model & tokenizer...")
    try:
        TOKENIZER = build_tokenizer(TEXT_MODEL_NAME)
    except Exception as e:
        print("[server][WARN] build_tokenizer failed:", e)
        TOKENIZER = None

    if MultiModalModel is not None:
        try:
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
            # try load adapters (text LoRA)
            if os.path.exists(ADAPTER_PATH):
                sd = torch.load(ADAPTER_PATH, map_location="cpu")
                try:
                    from peft import set_peft_model_state_dict
                    # assume MODEL.text_encoder exists and is a peft-wrapped module
                    set_peft_model_state_dict(MODEL.text_encoder, sd)
                    print(f"[server] Loaded text LoRA adapters from {ADAPTER_PATH}")
                except Exception as e:
                    print("[server][WARN] set_peft_model_state_dict failed:", e)
            else:
                print(f"[server][WARN] adapter file not found: {ADAPTER_PATH}")

        except Exception as e:
            print("[server][WARN] Building model failed:", e)
            MODEL = None

    # thresholds
    if os.path.exists(THR_PATH):
        THRESHOLDS = np.load(THR_PATH)
        print("[server] thresholds loaded:", THRESHOLDS)
    else:
        THRESHOLDS = np.array([0.5]*len(ISSUE_LABELS), dtype=np.float32)
        print("[server][WARN] thresholds not found, defaulting to 0.5")

# try load at import time
load_model()

# ----------------- Helpers -----------------
def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_sensor(payload: dict) -> str:
    fn = f"sensor_{payload.get('device_id','unknown')}_{_ts()}.json"
    path = os.path.join(SENSORS_DIR, fn)
    with open(path, "w") as f:
        json.dump({"ingested_at": datetime.utcnow().isoformat(), "payload": payload}, f, indent=2)
    return path

def save_image_file(upload: UploadFile, device_id: str, log: str = "") -> str:
    ext = os.path.splitext(upload.filename)[1] or ".jpg"
    fname = f"{device_id}_{_ts()}{ext}"
    outp = os.path.join(IMAGES_DIR, fname)
    with open(outp, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    # also write a small json with log/metadata
    meta = {"filename": fname, "device_id": device_id, "log": log, "ingested_at": datetime.utcnow().isoformat()}
    with open(os.path.join(IMAGES_DIR, fname + ".json"), "w") as m:
        json.dump(meta, m, indent=2)
    return outp

def publish_mqtt_command(device_id: str, command: dict) -> bool:
    topic = f"farm/{device_id}/commands"
    try:
        mqtt_publish.single(topic, payload=json.dumps(command), hostname=MQTT_BROKER, port=MQTT_PORT)
        return True
    except Exception as e:
        print("[server][ERROR] mqtt publish failed:", e)
        return False

def preprocess_image_bytes(path: str):
    try:
        img = Image.open(path).convert("RGB")
        t = IMAGE_TRANSFORM(img)  # [3,H,W]
        return t
    except Exception as e:
        print("[server][WARN] preprocess_image_bytes failed:", e)
        return None

def build_text_with_sensors(text: str, sensors: str) -> str:
    text = (text or "").strip()
    sensors = (sensors or "").strip()
    if sensors:
        sensors_line = sensors if sensors.upper().startswith("SENSORS:") else ("SENSORS: " + sensors)
    else:
        sensors_line = "SENSORS: (not provided)."
    if not text:
        text = "(no free-text log)."
    return f"{sensors_line}\nLOG: {text}"

def logits_to_output(text: str, logits: torch.Tensor, thresholds: np.ndarray) -> Dict[str, Any]:
    logits = apply_priors_to_logits(logits, [text])
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    thr = thresholds.astype(np.float32)
    mask = (probs >= thr).astype(int)
    active = [{"label": ISSUE_LABELS[i], "prob": float(probs[i]), "threshold": float(thr[i])}
              for i in range(len(ISSUE_LABELS)) if mask[i]]
    all_scores = [{"label": ISSUE_LABELS[i], "prob": float(probs[i]), "threshold": float(thr[i])}
                  for i in range(len(ISSUE_LABELS))]
    return {"active_labels": active, "all_scores": all_scores}

# ----------------- Routes -----------------
@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "labels": ISSUE_LABELS}

@app.get("/device_status")
async def device_status():
    # return in-memory device status
    return {"devices": DEVICE_STATUS}

# ---------- Original /predict (keeps behavior) ----------
@app.post("/predict")
async def predict(request: Request):
    global MODEL, TOKENIZER, THRESHOLDS
    if MODEL is None or TOKENIZER is None:
        load_model()

    content_type = request.headers.get("content-type", "").lower()
    text = ""
    sensors = ""
    client_id = "client"
    image_tensor = None

    if "application/json" in content_type:
        data = await request.json()
        text = str(data.get("text", "") or "")
        sensors = str(data.get("sensors", "") or "")
        client_id = str(data.get("client_id", client_id) or client_id)
    elif "multipart/form-data" in content_type:
        form = await request.form()
        text = str(form.get("text", "") or "")
        sensors = str(form.get("sensors", "") or "")
        client_id = str(form.get("client_id", client_id) or client_id)
        upload = form.get("image", None)
        if upload is not None:
            # Upload file is a SpooledTemporaryFile-like UploadFile
            image_bytes = upload.file.read()
            upload.file.seek(0)
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            except Exception as e:
                return JSONResponse({"error": f"bad image: {e}"}, status_code=400)
    else:
        return JSONResponse({"error": f"Unsupported Content-Type: {content_type}"}, status_code=415)

    if not text and not sensors:
        return JSONResponse({"error": "At least 'text' or 'sensors' must be provided."}, status_code=400)

    combined_text = build_text_with_sensors(text, sensors)
    # tokenize
    if TOKENIZER is None:
        return JSONResponse({"error": "tokenizer not available on server"}, status_code=500)
    enc = TOKENIZER(combined_text, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        if image_tensor is not None and MODEL is not None:
            out = MODEL(input_ids=input_ids, attention_mask=attention_mask, image=image_tensor)
        elif MODEL is not None:
            out = MODEL(input_ids=input_ids, attention_mask=attention_mask, image=None)
        else:
            out = None

    if out is None:
        return JSONResponse({"error": "model not available"}, status_code=500)

    logits = out.logits
    result = logits_to_output(combined_text, logits, THRESHOLDS)
    # update device status (lightweight)
    DEVICE_STATUS[client_id] = {"last_seen": datetime.utcnow().isoformat(), "last_result": result}

    return {"client_id": client_id, "text_used": combined_text, "result": result}

# ---------- New hardware endpoints ----------

@app.post("/sensor_upload")
async def sensor_upload(payload: dict):
    """
    Endpoint to be called by resource-constrained devices (ESP32 sensor nodes).
    Expected body (JSON), e.g.:
      {
        "device_id": "esp32-01",
        "soil": 23.5,
        "temp": 33.2,
        "hum": 42.1,
        "vpd": 2.1,
        "raw_log": "irrigation skipped yesterday"
      }
    The server saves the payload for later training and optionally runs light inference.
    """
    try:
        device_id = str(payload.get("device_id", "unknown"))
        path = save_sensor(payload)
        DEVICE_STATUS[device_id] = {"last_seen": datetime.utcnow().isoformat(), "last_sensor_path": path}

        # quick inference: build short combined text & run model if available (non-blocking)
        combined_text = build_text_with_sensors(payload.get("raw_log", ""), payload.get("sensors", ""))
        quick_pred = None
        if MODEL is not None and TOKENIZER is not None:
            enc = TOKENIZER(combined_text, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                out = MODEL(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE), image=None)
                quick_pred = logits_to_output(combined_text, out.logits, THRESHOLDS)
                DEVICE_STATUS[device_id]["last_result"] = quick_pred

        return {"ok": True, "saved": path, "quick_pred": quick_pred}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/image_upload")
async def image_upload(device_id: str = Form(...), log: str = Form(""), image: UploadFile = File(...)):
    """
    Accepts image uploads from esp32cam/pi camera.
    Stores the file + metadata and optionally runs multimodal inference.
    """
    try:
        saved_path = save_image_file(image, device_id, log)
        DEVICE_STATUS[device_id] = {"last_seen": datetime.utcnow().isoformat(), "last_image_path": saved_path}

        # run inference on saved image + log if model present
        image_tensor = preprocess_image_bytes(saved_path)
        combined_text = build_text_with_sensors(log, "")
        inf = None
        if MODEL is not None and TOKENIZER is not None and image_tensor is not None:
            enc = TOKENIZER(combined_text, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                out = MODEL(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE), image=image_tensor.unsqueeze(0).to(DEVICE))
                inf = logits_to_output(combined_text, out.logits, THRESHOLDS)
                DEVICE_STATUS[device_id]["last_result"] = inf

        return {"ok": True, "saved": saved_path, "inference": inf}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/publish_command")
async def publish_command(body: dict):
    """
    Publish a command to a device via MQTT.
    body: {"device_id": "...", "command": {"cmd": "RELAY_ON", "args": {...}}}
    """
    try:
        device_id = body.get("device_id")
        command = body.get("command", {})
        if not device_id:
            return JSONResponse({"ok": False, "error": "device_id required"}, status_code=400)
        ok = publish_mqtt_command(device_id, command)
        return {"ok": ok}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/upload_adapter")
async def upload_adapter(device_id: str = Form(...), adapter_file: UploadFile = File(...)):
    """
    Accept a LoRA adapter file from a client (simulate federated client upload).
    The server saves adapters under ingest/adapters/ for later aggregation/external FedAvg.
    """
    try:
        fname = f"{device_id}_{_ts()}_{adapter_file.filename}"
        outp = os.path.join(ADAPTERS_DIR, fname)
        with open(outp, "wb") as f:
            shutil.copyfileobj(adapter_file.file, f)
        return {"ok": True, "saved": outp}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/list_pending_for_training")
async def list_for_training(limit: int = 200):
    """
    List saved images and sensor json files that are not yet used for training.
    (A later offline job can read these into your dataset).
    """
    imgs = sorted([f for f in os.listdir(IMAGES_DIR) if not f.endswith(".json")], reverse=True)[:limit]
    sensors = sorted(os.listdir(SENSORS_DIR), reverse=True)[:limit]
    adapters = sorted(os.listdir(ADAPTERS_DIR), reverse=True)[:limit]
    return {"images": imgs, "sensors": sensors, "adapters": adapters}

@app.get("/download_image/{fname}")
async def download_image(fname: str):
    path = os.path.join(IMAGES_DIR, fname)
    if not os.path.exists(path):
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg", filename=fname)

# ----------------- Entry point -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
