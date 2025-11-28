# backend/server.py
import os
import json
import shutil
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import uvicorn
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent
MODEL_STORE = ROOT / "model_store" / "multimodal_demo"
ADAPTER_DIR = ROOT / "backend_data" / "federated_llm_adapter"
CLASSIFIER_DIR = ROOT / "backend_data" / "multimodal_classifier"
MANIFEST = MODEL_STORE / "manifest.json"

TELEMETRY_DIR = ROOT / "outputs" / "telemetry"
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

# persistent actions file
ACTIONS_FILE = ROOT / "outputs" / "device_actions.json"
# load existing or init empty dict { device_id: { "pin": "V1", "value": 1, "ts": "..." } }
if ACTIONS_FILE.exists():
    try:
        ACTIONS: Dict[str, Dict[str, Any]] = json.loads(ACTIONS_FILE.read_text())
    except Exception:
        ACTIONS = {}
else:
    ACTIONS = {}

def save_actions():
    ACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIONS_FILE.write_text(json.dumps(ACTIONS, indent=2))

app = FastAPI(title="FarmFederate Backend (no-Blynk)")

CLASSES = ["water_stress","nutrient_def","pest_risk","disease_risk","heat_stress"]

# --- model related code (unchanged from prior) ---
class FusionHead(nn.Module):
    def __init__(self, projection_dim, num_labels=len(CLASSES)):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(projection_dim*2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, num_labels)
        )
    def forward(self, text_emb, image_emb):
        fused = torch.cat([text_emb, image_emb], dim=1)
        return self.fusion(fused)

def download_if_missing(url: Optional[str], out_path: Path):
    if not url:
        return None
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {url} -> {out_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return out_path

# Model globals
MODEL = None
PROCESSOR = None
FUSION = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_models():
    if not MANIFEST.exists():
        raise FileNotFoundError("manifest.json not found in model_store/multimodal_demo")
    m = json.loads(MANIFEST.read_text())
    local_multimodal = MODEL_STORE / "multimodal.pt"
    local_fusion = MODEL_STORE / "fusion_epoch0.pt"
    local_adapter = ADAPTER_DIR / "global_adapter.pt"
    local_classifier = CLASSIFIER_DIR / "model.pt"
    download_if_missing(m.get("multimodal_model"), local_multimodal)
    download_if_missing(m.get("fusion_checkpoint"), local_fusion)
    download_if_missing(m.get("adapter"), local_adapter)
    download_if_missing(m.get("classifier"), local_classifier)
    return local_multimodal, local_fusion, local_adapter, local_classifier

def load_models(model_path, fusion_path, adapter_path):
    global MODEL, PROCESSOR, FUSION, DEVICE
    print("[INFO] Loading CLIP model...")
    MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    proj_dim = MODEL.config.projection_dim
    FUSION = FusionHead(proj_dim, num_labels=len(CLASSES))
    if fusion_path.exists():
        try:
            ck = torch.load(fusion_path, map_location="cpu")
            FUSION.load_state_dict(ck, strict=False)
            print("[INFO] Fusion checkpoint loaded")
        except Exception as e:
            print("[WARN] Fusion load failed:", e)
    if adapter_path.exists():
        try:
            adapter_sd = torch.load(adapter_path, map_location="cpu")
            MODEL.load_state_dict(adapter_sd, strict=False)
            print("[INFO] Adapter loaded (non-strict)")
        except Exception as e:
            print("[WARN] Adapter load failed:", e)
    MODEL.to(DEVICE); FUSION.to(DEVICE); MODEL.eval(); FUSION.eval()

def predict_image_text(image_bytes: bytes, text: str):
    inputs = PROCESSOR(text=[text], images=[image_bytes], return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(DEVICE)
    with torch.no_grad():
        out = MODEL(pixel_values=pixel_values, input_ids=None, return_loss=None, text=[text])
        text_emb = out.text_embeds.to(DEVICE)
        image_emb = out.image_embeds.to(DEVICE)
        logits = FUSION(text_emb, image_emb)
        probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()
    return probs

def predict_from_sensors(sensor_dict):
    sm = float(sensor_dict.get("soil_moisture", 0))
    ah = float(sensor_dict.get("air_humidity", 0))
    temp = float(sensor_dict.get("temp_c", 25))
    flow = float(sensor_dict.get("flow_rate", 0))
    p_water = 0.0
    if sm < 30 or ah < 30:
        p_water = min(1.0, (35 - sm) / 50.0 + (30 - ah) / 80.0)
    p_nutrient = 0.1 if sm < 20 else 0.05
    p_pest = 0.15 if ah > 70 and temp > 25 else 0.05
    p_disease = 0.15 if ah > 80 else 0.05
    p_heat = 0.2 if temp > 33 else 0.05
    return [p_water, p_nutrient, p_pest, p_disease, p_heat]

# --- Device action endpoints (new) ---

class ActionRequest(BaseModel):
    device_id: str
    pin: str   # e.g. "V1" or "relay"
    value: int # 0/1
    reason: Optional[str] = None

@app.post("/set_action")
def set_action(req: ActionRequest):
    """
    Called by frontend (or operator) to request a device action.
    Stored persistently in actions.json until the device polls and acknowledges.
    """
    device = req.device_id
    ACTIONS[device] = {
        "pin": req.pin,
        "value": int(req.value),
        "reason": req.reason or "",
        "ts": datetime.utcnow().isoformat(),
        "ack": False
    }
    save_actions()
    return {"ok": True, "queued": ACTIONS[device]}

@app.get("/poll/{device_id}")
def poll_device(device_id: str):
    """
    Called by the ESP32: returns pending action for the device (if any),
    and does NOT remove it until device calls /ack_action.
    """
    action = ACTIONS.get(device_id)
    if action:
        return {"action": action}
    return {"action": None}

class AckRequest(BaseModel):
    device_id: str
    success: bool
    note: Optional[str] = None

@app.post("/ack_action")
def ack_action(req: AckRequest):
    """
    Device acknowledges that it executed (or failed) the action.
    This clears the queued action.
    """
    device = req.device_id
    if device in ACTIONS:
        record = ACTIONS.pop(device)
        save_actions()
        # store ack record for audit
        audit_dir = ROOT / "outputs" / "action_acks"
        audit_dir.mkdir(parents=True, exist_ok=True)
        fname = audit_dir / f"{device}_{datetime.utcnow().isoformat().replace(':','-')}.json"
        fname.write_text(json.dumps({"device_id": device, "ack": req.success, "note": req.note, "record": record}, indent=2))
        return {"ok": True}
    return {"ok": False, "error": "no action queued for device"}

# --- telemetry & other endpoints ---

@app.post("/telemetry")
async def receive_telemetry(payload: dict):
    if not payload.get("device_id"):
        raise HTTPException(status_code=400, detail="device_id required")
    device = payload["device_id"]
    ts = payload.get("ts", datetime.utcnow().isoformat())
    dest = TELEMETRY_DIR / f"{device}_{ts.replace(':','-')}.json"
    dest.write_text(json.dumps(payload))
    sensor_probs = predict_from_sensors(payload)
    action = {"open_valve": False, "reason": None}
    if sensor_probs[0] > 0.6 and float(payload.get("flow_rate",0)) == 0.0:
        # auto-queue an action for device to open valve
        ACTIONS[device] = {"pin": "relay", "value": 1, "reason": "Auto-open due to water_stress", "ts": datetime.utcnow().isoformat(), "ack": False}
        save_actions()
        action["open_valve"] = True
        action["reason"] = "Auto action queued"
    return {"ok": True, "sensor_probs": sensor_probs, "action": action}

@app.get("/telemetry_latest")
def telemetry_latest(device_id: str):
    """
    Return the latest telemetry JSON for a given device_id (or 404).
    """
    files = sorted(TELEMETRY_DIR.glob(f"{device_id}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"device_id": device_id, "latest": None}
    data = json.loads(files[0].read_text())
    return {"device_id": device_id, "latest": data}

@app.post("/predict")
async def predict_endpoint(text: str = Form(""), file: UploadFile = File(None)):
    if file is None:
        raise HTTPException(status_code=400, detail="File required")
    image_bytes = await file.read()
    probs = predict_image_text(image_bytes, text)
    return {"classes": CLASSES, "probs": probs}

@app.post("/predict_sensors")
async def predict_sensors(payload: dict):
    probs = predict_from_sensors(payload)
    return {"classes": CLASSES, "probs": probs}

@app.get("/manifest")
def get_manifest():
    if not MANIFEST.exists():
        return {"error":"manifest not found"}
    with open(MANIFEST,"r") as f:
        m=json.load(f)
    for k,v in list(m.items()):
        if isinstance(v,str):
            m[k+"_exists"] = (MODEL_STORE / Path(v).name).exists()
    return m

@app.on_event("startup")
def startup():
    try:
        mp, fp, ap, cp = prepare_models()
        load_models(mp, fp, ap)
        print("[INFO] Server ready")
    except Exception as e:
        print("[WARN] Model load / manifest missing:", e)

if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
