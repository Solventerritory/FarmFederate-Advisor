# server.py
"""
Flask server providing /predict endpoint that accepts:
 - form field 'text' (string)
 - file field 'image' (uploaded image)
Returns JSON with predicted labels, probabilities, rationales, advice.
"""

import os, io, json
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from multimodal_model import MultimodalClassifier, set_text_adapter_state_dict, set_image_head_state_dict, ISSUE_LABELS
from PIL import Image

app = Flask("farm_federate_server")
CORS(app)

CHECKPOINT = os.path.join("checkpoints", "global_multimodal.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[server] device:", device)

# Build model skeleton and load saved adapter states if present
model = MultimodalClassifier()
model.to(device)
if os.path.exists(CHECKPOINT):
    ck = torch.load(CHECKPOINT, map_location="cpu")
    if "text_adapter" in ck:
        set_text_adapter_state_dict(model, ck["text_adapter"])
    if "image_head" in ck:
        set_image_head_state_dict(model, ck["image_head"])
    if "fusion" in ck:
        model.fusion.load_state_dict(ck["fusion"])
    print("[server] loaded checkpoint", CHECKPOINT)
else:
    print("[server] checkpoint not found; running with initial adapters")

def preprocess_image(pil_image):
    proc = model.image_processor(images=pil_image, return_tensors="pt")
    return proc["pixel_values"].to(device)

def preprocess_text(text):
    enc = model.tokenizer(text, truncation=True, padding="max_length", max_length=160, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)

ADVICE = {
    "water_stress": "Irrigate earlier; mulch; monitor soil moisture AM/PM.",
    "nutrient_def": "Balance NPK (N focus if older leaves yellow); verify with LCC.",
    "pest_risk": "Inspect undersides; sticky traps; early biocontrol or mild soap.",
    "disease_risk": "Improve airflow; avoid late overhead irrigation; prune infected leaves.",
    "heat_stress": "Provide shade at peak heat; keep moisture stable; ensure K sufficiency.",
}

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "") or ""
    img = None
    if "image" in request.files:
        f = request.files["image"]
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        except Exception:
            img = None

    # Preprocess
    input_ids, attention_mask = preprocess_text(text)
    pixel_values = None
    if img is not None:
        pixel_values = preprocess_image(img)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    # thresholds fixed
    thr = 0.5
    preds = (probs >= thr).astype(int).tolist()
    labels = [ISSUE_LABELS[i] for i,v in enumerate(preds) if v==1]
    # rationales (simple): keyword hits and sensor absent
    rationale = []
    t = text.lower()
    for i,lab in enumerate(ISSUE_LABELS):
        if probs[i] > 0.25:
            reason = f"p={probs[i]:.2f}"
            rationale.append({ "label": lab, "prob": float(probs[i]), "reason": reason })
    advice = [ADVICE[l] for l in labels] if labels else ["Conditions look normal. Continue routine monitoring."]
    return jsonify({
        "labels": labels,
        "probs": [float(x) for x in probs.tolist()],
        "rationales": rationale,
        "advice": advice
    })

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status":"ok","msg":"multimodal server running"})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)
