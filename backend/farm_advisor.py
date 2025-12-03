#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
farm_advisor_full.py — Combined multimodal federated LoRA crop-stress detector.

Features:
 - Text-only or multimodal (text + image) training
 - Auto-download / prepare image datasets from HF (best-effort)
 - Federated LoRA: clients train LoRA adapters (text encoder), we aggregate adapter weights
 - ViT vision encoder fused with text encoder (fusion MLP)
 - Sensor priors applied only at inference
 - EMA, FocalLoss, calibration, MC-Dropout, rationales
"""
import os, re, math, time, gc, random, argparse, hashlib, json, shutil
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# Repro
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional HF datasets
try:
    from datasets import load_dataset, DownloadConfig, Dataset as HFDataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    ViTModel,
    get_linear_schedule_with_warmup,
)
# quiet transformers logs
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# PEFT / LoRA
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

# vision preprocessing
from PIL import Image
import torchvision.transforms as T
import requests
from io import BytesIO

# --------------------- Labels ---------------------
ISSUE_LABELS = ["water_stress","nutrient_def","pest_risk","disease_risk","heat_stress"]
LABEL_TO_ID = {k:i for i,k in enumerate(ISSUE_LABELS)}
NUM_LABELS = len(ISSUE_LABELS)

# --------------------- CLI ---------------------
def get_args():
    ap = argparse.ArgumentParser()
    # Data / multimodal
    ap.add_argument("--dataset", type=str, default="mix",
                    choices=["localmini","gardian","argilla","agnews","mix","hf_images"])
    ap.add_argument("--mix_sources", type=str, default="gardian,argilla,agnews,localmini")
    ap.add_argument("--max_per_source", type=int, default=2000)
    ap.add_argument("--max_samples", type=int, default=0, help="cap AFTER filtering/labeling")
    ap.add_argument("--mqtt_csv", type=str, default="")
    ap.add_argument("--extra_csv", type=str, default="")
    ap.add_argument("--use_images", action="store_true", help="enable image inputs alongside text")
    ap.add_argument("--image_dir", type=str, default="images", help="root dir for images (download + local)")
    ap.add_argument("--image_csv", type=str, default="", help="CSV with columns filename,text,labels")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--vit_name", type=str, default="google/vit-base-patch16-224-in21k", help="HF ViT model name")
    ap.add_argument("--freeze_vision", action="store_true", help="freeze vision backbone")

    # Model / LoRA
    ap.add_argument("--model_name", type=str, default="roberta-base")
    ap.add_argument("--freeze_base", action="store_true")
    ap.add_argument("--no_freeze_base", dest="freeze_base", action="store_false")
    ap.set_defaults(freeze_base=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Train / Fed
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--local_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--ema_decay", type=float, default=0.997)
    ap.add_argument("--precision_target", type=float, default=0.90, help="per-class precision target during calibration")
    ap.add_argument("--dirichlet_alpha", type=float, default=0.25, help="smaller => stronger non-IID")
    ap.add_argument("--participation", type=float, default=0.8, help="fraction of clients per round")
    ap.add_argument("--client_dropout", type=float, default=0.05, help="probability a sampled client drops")
    ap.add_argument("--prior_scale", type=float, default=0.30, help="scales sensor priors effect (0..1)")
    ap.add_argument("--label_noise", type=float, default=0.05, help="flip/remove labels with this prob")

    # Logging / run
    ap.add_argument("--cap_metric_print_at", type=float, default=0.999)
    ap.add_argument("--quiet_eval", action="store_true")
    ap.add_argument("--save_dir", type=str, default="checkpoints_paper")
    ap.add_argument("--inference", action="store_true")
    ap.add_argument("--query", type=str, default="")
    ap.add_argument("--sensors", type=str, default="")
    ap.add_argument("--offline", action="store_true", help="use local cache only")
    ap.add_argument("--lowmem", action="store_true")

    # Inference / mc-dropout
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.07)
    ap.add_argument("--samples", type=int, default=16)

    args, _ = ap.parse_known_args()
    return args

ARGS = get_args()
os.makedirs(ARGS.save_dir, exist_ok=True)
os.makedirs(ARGS.image_dir, exist_ok=True)

if ARGS.lowmem:
    ARGS.batch_size = min(ARGS.batch_size, 8)
    ARGS.max_len = min(ARGS.max_len, 128)
    ARGS.lora_r = min(ARGS.lora_r, 4)
    ARGS.lora_alpha = min(ARGS.lora_alpha, 16)

# --------------------- small utils ---------------------
def _fmt_metric(x: float, cap: float = None) -> float:
    cap = ARGS.cap_metric_print_at if cap is None else cap
    return min(float(x), float(cap))
def _fmt_str(x: float) -> str:
    return f"{_fmt_metric(x):.3f}"
def _ascii_ratio(s: str) -> float:
    if not s: return 0.0
    return sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
def _lang_ok(s: str) -> bool: return _ascii_ratio(s) >= 0.6
def _norm(txt: str) -> str: return re.sub(r"\s+", " ", txt).strip()

# --------------------- Weak labels + ag gate ---------------------
KW = {
    "water": ["dry","wilting","wilt","parched","drought","moisture","irrigation","canopy stress","water stress","droop","cracking soil","hard crust","soil moisture low"],
    "nutrient": ["nitrogen","phosphorus","potassium","npk","fertilizer","fertiliser","chlorosis","chlorotic","interveinal","leaf color chart","lcc","spad","low spad","older leaves yellowing","necrotic margin","micronutrient","deficiency"],
    "pest": ["pest","aphid","whitefly","borer","hopper","weevil","caterpillar","larvae","thrips","mites","trap","sticky residue","honeydew","chewed","webbing","frass","insect"],
    "disease": ["blight","rust","mildew","smut","rot","leaf spot","necrosis","pathogen","fungal","bacterial","viral","lesion","mosaic","wilt disease","canker","powdery mildew","downy"],
    "heat": ["heatwave","hot","scorch","sunburn","thermal stress","high temperature","blistering","desiccation","sun scorch","leaf burn","heat stress"],
}
AG_CONTEXT = re.compile(
    r"\b(agri|agricultur|farm|farmer|field|crop|soil|irrigat|harvest|yield|paddy|rice|wheat|maize|corn|cotton|soy|orchard|greenhouse|seedling|fertiliz|manure|compost|pest|fung|blight|leaf|canopy|mulch|drip|sprinkler|nursery|plantation|horticul)\b",
    re.I,
)
def is_ag_context(s: str) -> bool:
    if "SENSORS:" in s:
        parts = s.split("LOG:", 1)
        log_txt = parts[1] if len(parts) > 1 else s
        return bool(AG_CONTEXT.search(log_txt))
    return bool(AG_CONTEXT.search(s))

def weak_labels(text: str) -> List[int]:
    t = text.lower()
    if not is_ag_context(t): return []
    labs = set()
    if any(k in t for k in KW["water"]): labs.add("water_stress")
    if any(k in t for k in KW["nutrient"]):
        strong_n = any(x in t for x in ["chlorosis","chlorotic","interveinal","npk","nitrogen","potassium","leaf color chart","lcc","low spad","spad"])
        qualified_yellow = ("yellowing" in t and ("older leaves" in t or "old leaves" in t))
        if strong_n or qualified_yellow:
            labs.add("nutrient_def")
    if any(k in t for k in KW["pest"]): labs.add("pest_risk")
    if any(k in t for k in KW["disease"]): labs.add("disease_risk")
    if any(k in t for k in KW["heat"]): labs.add("heat_stress")
    return [LABEL_TO_ID[x] for x in sorted(labs)]

# --------------------- Sensor fusion + priors ---------------------
_SENS_RE = re.compile(
    r"soil_moisture=(?P<sm>\d+(?:\.\d+)?)%.*?soil_pH=(?P<ph>\d+(?:\.\d+)?).*?temp=(?P<t>\d+(?:\.\d+)?)°C.*?humidity=(?P<h>\d+(?:\.\d+)?)%.*?VPD=(?P<vpd>\d+(?:\.\d+)?) kPa.*?rainfall_24h=(?P<rf>\d+(?:\.\d+)?)mm",
    re.I | re.S
)
def simulate_sensor_summary():
    soil_m = round(np.clip(np.random.normal(30, 6), 10, 50), 1)
    soil_ph = round(np.clip(np.random.normal(6.5, 0.4), 5.5, 7.5), 1)
    temp   = round(np.clip(np.random.normal(29, 4), 18, 40), 1)
    hum    = round(np.clip(np.random.normal(60, 12), 30, 90), 0)
    vpd    = round(np.clip(np.random.normal(1.4, 0.4), 0.6, 2.4), 1)
    rain   = round(np.clip(np.random.normal(1.0, 1.0), 0.0, 6.0), 1)
    trend  = np.random.choice(["↑","↓","→"], p=[0.3,0.3,0.4])
    return f"SENSORS: soil_moisture={soil_m}%, soil_pH={soil_ph}, temp={temp}°C, humidity={hum}%, VPD={vpd} kPa, rainfall_24h={rain}mm (trend: {trend})."
def fuse_text(sensor_txt:str, main_txt:str, mqtt_msg:str="") -> str:
    if main_txt.strip().startswith("SENSORS:"):
        base = f"{main_txt.strip()}"
        if "LOG:" not in base: base = f"{base}\nLOG: (no additional log)"
    else:
        base = f"{sensor_txt}\nLOG: {_norm(main_txt)}"
    return f"{base}{(f'\nMQTT: {mqtt_msg.strip()}' if mqtt_msg else '')}"
def _parse_sensors(text: str) -> Optional[Dict[str, float]]:
    m = _SENS_RE.search(text)
    if not m: return None
    try:
        return dict(
            sm=float(m.group("sm")), ph=float(m.group("ph")), t=float(m.group("t")),
            h=float(m.group("h")), vpd=float(m.group("vpd")), rf=float(m.group("rf"))
        )
    except Exception:
        return None
def sensor_priors(text: str) -> np.ndarray:
    b = np.zeros(NUM_LABELS, dtype=np.float32)
    s = _parse_sensors(text)
    if not s: return b
    sm, ph, t, h, vpd, rf = s["sm"], s["ph"], s["t"], s["h"], s["vpd"], s["rf"]
    if sm >= 28 and vpd <= 1.2: b[0] -= 0.25
    if sm <= 18 or vpd >= 2.0:  b[0] += 0.18
    if ph < 5.8 or ph > 7.4:    b[1] += 0.12
    if 45 <= h <= 70 and rf <= 2.0: b[2] += 0.05
    if h >= 70 or rf >= 2.0:   b[3] += 0.10
    if h <= 45 and rf == 0 and vpd >= 2.0: b[3] -= 0.12
    if t >= 36 or vpd >= 2.2:  b[4] += 0.15
    if t <= 24:                b[4] -= 0.15
    b = b + np.random.normal(0, 0.03, size=b.shape).astype(np.float32)
    if np.random.rand() < 0.10:
        b *= 0.0
    return b
def apply_priors_to_logits(logits: torch.Tensor, texts: Optional[List[str]]) -> torch.Tensor:
    if texts is None or ARGS.prior_scale <= 0:
        return logits
    biases = [torch.tensor(sensor_priors(t), dtype=logits.dtype, device=logits.device) for t in texts]
    return logits + ARGS.prior_scale * torch.stack(biases, dim=0)

# --------------------- Synthetic + dataset builders ---------------------
LOCAL_BASE = [
    "Maize leaves show interveinal chlorosis and older leaves are yellowing after light rains.",
    "Tomato plants have whiteflies; sticky residue under leaves; some curling.",
    "Rice field shows cracked, dry soil; seedlings drooping under midday sun.",
    "Wheat leaves with orange pustules; reduced tillering; humid mornings reported.",
    "Chili plants show sun scorch on exposed fruits during heatwave; leaf edges crisping.",
    "Cotton has aphid clusters; honeydew; ants moving up stems.",
    "Leaf spots with concentric rings on brinjal; humid nights; poor airflow.",
    "Drip lines clogged; soil moisture uneven; some beds dry, others saturated.",
    "Banana shows potassium deficiency signs: marginal necrosis and weak petioles.",
    "Cabbage seedlings wilt after transplanting; wind and low humidity recorded.",
]
TEMPLATES = [
    "Farmer noted {symptom} while sensors read temp {temp}°C and humidity {hum}%.",
    "{crop} field observed {symptom}; irrigation last 48h minimal; VPD around {vpd} kPa.",
    "Scouting report: {symptom}. Traps show {pest_count} pests per card.",
    "After {weather}, plants show {symptom}. Soil moisture near {sm}% and pH {ph}.",
]
SYMPTOMS = [
    "leaf curling and pale yellowing on older leaves",
    "sticky residue and presence of aphids",
    "powdery mildew patches on lower canopy",
    "lesions with yellow halos and necrotic centers",
    "wilting during afternoon; recovery at night",
    "chewed margins and frass on leaves",
    "sun scorch on exposed leaves",
    "brown rust pustules along veins",
    "dry topsoil with hard crusting",
    "interveinal chlorosis indicating nutrient stress",
]
CROPS = ["rice","wheat","maize","soybean","cotton","tomato","chili","potato","banana","cabbage","brinjal","mustard","sugarcane"]
WEATHERS = ["a hot, dry wind","sudden heavy rain","two cloudy days","a heatwave","late evening irrigation","morning fog","no rainfall for a week"]

def _maybe_read_mqtt(mqtt_csv:str):
    if mqtt_csv and os.path.exists(mqtt_csv):
        m = pd.read_csv(mqtt_csv)
        return (m["message"].astype(str).tolist()) if "message" in m.columns else []
    return []

def make_balanced_local(n_per=300, n_per_nutrient=600):
    seeds = {
        "water_stress": [
            "Topsoil is cracking and leaves droop at midday; irrigation uneven.",
            "Canopy stress at noon; mulch missing; dry beds observed."
        ],
        "nutrient_def": [
            "Interveinal chlorosis on older leaves suggests nitrogen deficiency.",
            "Marginal necrosis indicates potassium shortfall.",
            "Leaf Color Chart shows low score; possible N deficiency.",
            "SPAD readings are low on older leaves; fertilization overdue."
        ],
        "pest_risk": [
            "Aphids and honeydew on undersides; sticky traps catching many.",
            "Chewed margins and frass; small caterpillars on leaves."
        ],
        "disease_risk": [
            "Orange pustules indicate rust; humid mornings; leaf spots spreading.",
            "Powdery mildew on lower canopy; poor airflow in dense rows."
        ],
        "heat_stress": [
            "Sun scorch on exposed leaves during heatwave; leaf edges crisping.",
            "High temperature window causing thermal stress around midday."
        ],
    }
    out=[]
    for k, lst in seeds.items():
        reps = n_per_nutrient if k=="nutrient_def" else n_per
        for _ in range(reps): out.append(random.choice(lst))
    random.shuffle(out); return out

def build_localmini(max_samples:int=0, mqtt_csv:str="", extra_csv:str="") -> pd.DataFrame:
    mqtt_msgs = _maybe_read_mqtt(mqtt_csv)
    texts = list(LOCAL_BASE) + make_balanced_local(300, 600)
    N = 2000
    for _ in range(N):
        sensor = simulate_sensor_summary()
        s = random.choice(TEMPLATES).format(
            symptom=random.choice(SYMPTOMS), crop=random.choice(CROPS),
            temp=round(np.clip(np.random.normal(32, 4), 15, 45),1),
            hum=int(np.clip(np.random.normal(55,15),15,95)),
            vpd=round(np.clip(np.random.normal(1.8,0.7),0.2,4.0),1),
            pest_count=int(np.clip(np.random.poisson(3), 0, 30)),
            sm=round(np.clip(np.random.normal(20,7),2,60),1),
            ph=round(np.clip(np.random.normal(6.5,0.6),4.5,8.5),1),
            weather=random.choice(WEATHERS),
        )
        mqtt = random.choice(mqtt_msgs) if mqtt_msgs and random.random()<0.4 else ""
        texts.append(fuse_text(sensor, s, mqtt))
    if extra_csv and os.path.exists(extra_csv):
        df_extra = pd.read_csv(extra_csv)
        for t in df_extra.get("text", pd.Series(dtype=str)).astype(str).tolist():
            sensor = simulate_sensor_summary()
            mqtt = random.choice(mqtt_msgs) if mqtt_msgs and random.random()<0.5 else ""
            texts.append(fuse_text(sensor, t, mqtt))

    rows=[]
    for t in texts:
        labs = weak_labels(t)
        if labs: rows.append((_norm(t), labs))
    df = pd.DataFrame(rows, columns=["text","labels"])
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=SEED).reset_index(drop=True)
    return df

# --------------------- HF dataset helpers (image download best-effort) ---------------------
def _load_ds(name, split=None, streaming=False):
    if not HAS_DATASETS:
        raise RuntimeError("datasets lib not available")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dlconf = DownloadConfig(max_retries=3)
    kw = {"streaming": streaming, "download_config": dlconf}
    if token: kw.update({"token": token, "use_auth_token": token})
    for attempt in range(4):
        try:
            return load_dataset(name, split=split, **kw) if split else load_dataset(name, **kw)
        except Exception as e:
            if any(x in str(e) for x in ["429","Read timed out","504","Temporary failure","Connection"]):
                time.sleep(min(60, 1.5*(2**attempt))); kw["streaming"]=True; continue
            raise
    kw["streaming"]=True
    return load_dataset(name, split=split, **kw)

AGRI_RE = re.compile(r"\b(agri|agriculture|farm|farmer|farming|crop|soil|harvest|irrigat|pest|blight|drought|yield|wheat|rice|paddy|maize|soy|cotton|fertiliz|orchard|greenhouse|horticul)\b", re.I)
NON_AG_NOISE = re.compile(r"\b(NFL|NBA|MLB|NHL|tennis|golf|soccer|cricket|stocks?|Nasdaq|Dow Jones|earnings|IPO|merger|Hollywood|movie|music|concert)\b", re.I)

def build_gardian_stream(max_per:int=2000) -> List[str]:
    ds = _load_ds("CGIAR/gardian-ai-ready-docs", streaming=True)
    texts=[]; seen=0
    for sp in (ds.keys() if isinstance(ds, dict) else []):
        for r in ds[sp]:
            raw = (r.get("text") or r.get("content") or "").strip()
            if raw and _lang_ok(raw):
                texts.append(_norm(raw)); seen+=1
                if seen>=max_per: break
        if seen>=max_per: break
    return texts

def build_argilla_stream(max_per:int=2000) -> List[str]:
    ds = _load_ds("argilla/farming")
    texts=[]; seen=0
    if isinstance(ds, dict):
        for sp in ds:
            for r in ds[sp]:
                q = str(r.get("evolved_questions","")).strip()
                a = str(r.get("domain_expert_answer","")).strip()
                raw = (q + " " + a).strip()
                if raw and _lang_ok(raw):
                    texts.append(_norm(raw)); seen+=1
                    if seen>=max_per: break
            if seen>=max_per: break
    return texts

def build_agnews_agri(max_per:int=2000) -> List[str]:
    train = _load_ds("ag_news", split="train", streaming=True)
    texts=[]; seen=0
    for r in train:
        raw = (r.get("text") or "").strip()
        if raw and AGRI_RE.search(raw) and not NON_AG_NOISE.search(raw) and _lang_ok(raw):
            texts.append(_norm(raw)); seen+=1
            if seen >= max_per: break
    return texts

# --------------------- Auto-download image datasets (best-effort) ---------------------
def _download_image(url: str, dst_path: str, timeout=10) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        im = Image.open(BytesIO(resp.content)).convert("RGB")
        im.save(dst_path, format="JPEG", quality=90)
        return True
    except Exception:
        return False

def prepare_images_from_hf(dataset_name: str, max_items: int, image_dir: str) -> pd.DataFrame:
    """
    Try to load HF dataset and find image fields (image, image_url, img_url). Returns DataFrame columns:
      - text: fused text (sensor + text)
      - labels: weak labels (list of ints)
      - image_path: local filename saved under image_dir
    This is best-effort — some HF datasets store actual Image objects, some store URLs.
    """
    rows=[]
    if not HAS_DATASETS:
        print("[Images] datasets lib not available; skipping HF image download.")
        return pd.DataFrame(rows, columns=["text","labels","image_path"])
    try:
        ds = _load_ds(dataset_name, streaming=True)
    except Exception as e:
        print(f"[Images] failed to load {dataset_name}: {e}")
        return pd.DataFrame(rows, columns=["text","labels","image_path"])

    print(f"[Images] Scanning {dataset_name} for image fields (<= {max_items}) ...")
    cnt=0
    for rec in (ds if not isinstance(ds, dict) else (r for sp in ds for r in ds[sp])):
        if cnt>=max_items: break
        # find text-like field
        text_candidates = [rec.get(k,"") for k in ("text","caption","sentence","report","content") if k in rec]
        raw_text = ""
        if isinstance(text_candidates, list) and len(text_candidates)>0:
            raw_text = str(next((t for t in text_candidates if t), ""))

        # find image field
        img_field = None
        for k in rec.keys():
            if "image" in k.lower() or "img" in k.lower() or "photo" in k.lower():
                img_field = k; break
        if img_field is None:
            # check for nested features (e.g., rec["features"])
            # skip if can't find
            continue

        img_val = rec.get(img_field)
        local_fname = None
        if isinstance(img_val, dict) and "path" in img_val:
            # sometimes datasets store 'path' to local cache (but streaming may not)
            try:
                p = img_val.get("path")
                if p and os.path.exists(p):
                    local_fname = os.path.join(image_dir, os.path.basename(p))
                    shutil.copyfile(p, local_fname)
                else:
                    # fallback to try 'url'
                    url = img_val.get("url") or img_val.get("img_url") or img_val.get("image_url")
                    if url:
                        local_fname = os.path.join(image_dir, f"{dataset_name}_{cnt}.jpg")
                        ok = _download_image(url, local_fname)
                        if not ok:
                            local_fname = None
            except Exception:
                local_fname=None
        elif isinstance(img_val, str):
            # string may be url or path
            if img_val.startswith("http"):
                local_fname = os.path.join(image_dir, f"{dataset_name}_{cnt}.jpg")
                ok = _download_image(img_val, local_fname)
                if not ok:
                    local_fname = None
            else:
                # maybe cached path
                if os.path.exists(img_val):
                    local_fname = os.path.join(image_dir, os.path.basename(img_val))
                    shutil.copyfile(img_val, local_fname)
        elif hasattr(img_val, "to_pil") or isinstance(img_val, Image.Image):
            try:
                local_fname = os.path.join(image_dir, f"{dataset_name}_{cnt}.jpg")
                if isinstance(img_val, Image.Image):
                    img_val.convert("RGB").save(local_fname, format="JPEG", quality=90)
                else:
                    # try to call to_pil
                    pil = img_val.to_pil()
                    pil.convert("RGB").save(local_fname, format="JPEG", quality=90)
            except Exception:
                local_fname = None

        if not local_fname:
            continue

        txt = fuse_text(simulate_sensor_summary(), str(raw_text or ""))
        labs = weak_labels(txt)
        if not labs:
            continue
        rows.append((txt, labs, os.path.basename(local_fname)))
        cnt += 1

    df = pd.DataFrame(rows, columns=["text","labels","image_path"])
    print(f"[Images] prepared {len(df)} items from {dataset_name}")
    return df

# --------------------- MIX builder (now supports HF image auto-prep) ---------------------
def build_mix(max_per_source:int, mqtt_csv:str, extra_csv:str) -> pd.DataFrame:
    sources = [s.strip().lower() for s in ARGS.mix_sources.split(",") if s.strip()]
    pool=[]
    def try_source(name, fn):
        print(f"[Mix] Loading {name} (<= {max_per_source}) ...")
        try:
            raw = fn(max_per_source)
            pool.extend([(name, t) for t in raw])
            print(f"[Mix] {name} added {len(raw)}")
        except Exception as e:
            print(f"[Mix] {name} skipped: {e}")
    if "gardian" in sources: try_source("gardian", build_gardian_stream)
    if "argilla" in sources: try_source("argilla", build_argilla_stream)
    if "agnews"  in sources: try_source("agnews", build_agnews_agri)
    if "localmini" in sources:
        lm_df = build_localmini(max_per_source, mqtt_csv, extra_csv)
        for t, _ in lm_df[["text","labels"]].itertuples(index=False):
            pool.append(("localmini", t))

    # dedup & fuse
    seen=set(); dedup=[]
    for src, txt in pool:
        h = hashlib.sha1(_norm(txt).encode("utf-8","ignore")).hexdigest()
        if h not in seen:
            seen.add(h); dedup.append((src, _norm(txt)))

    rows=[]
    for src, raw in dedup:
        sensor = simulate_sensor_summary()
        text = fuse_text(sensor, raw)
        labs = weak_labels(text)
        if labs: rows.append((text, labs, src))
    df = pd.DataFrame(rows, columns=["text","labels","source"])
    print("[Mix] Source breakdown:\n", df["source"].value_counts())
    return df[["text","labels"]]

# --------------------- Image-CSV builder (preferred for user-provided multimodal) ---------------------
def build_corpus_with_images(image_csv:str, image_root:str="", max_samples:int=0) -> pd.DataFrame:
    if not os.path.exists(image_csv):
        raise RuntimeError(f"image_csv not found: {image_csv}")
    df_raw = pd.read_csv(image_csv)
    rows=[]
    for _, r in df_raw.iterrows():
        text = str(r.get("text","")).strip()
        fname = str(r.get("filename","") or r.get("image_path","")).strip()
        labs_raw = r.get("labels","")
        if pd.isna(labs_raw) or str(labs_raw).strip()=="":
            labs=[]
        elif isinstance(labs_raw, str):
            parts=[x.strip() for x in labs_raw.split(",") if x.strip()]
            labs=[]
            for p in parts:
                if p.isdigit(): labs.append(int(p))
                elif p in LABEL_TO_ID: labs.append(LABEL_TO_ID[p])
            labs = sorted(set(labs))
        elif isinstance(labs_raw, (list,tuple)):
            labs=list(labs_raw)
        else:
            labs=[]
        if not labs: continue
        txt = fuse_text(simulate_sensor_summary(), text)
        rows.append((txt, labs, fname))
    out = pd.DataFrame(rows, columns=["text","labels","image_path"])
    if max_samples and len(out) > max_samples:
        out = out.sample(max_samples, random_state=SEED).reset_index(drop=True)
    return out

# --------------------- Build corpus (now supports auto image harvesting) ---------------------
def build_corpus() -> pd.DataFrame:
    # If explicit image csv and use_images -> prefer that (user-specified)
    if ARGS.use_images and ARGS.image_csv and os.path.exists(ARGS.image_csv):
        print("[Build] Using provided image_csv for multimodal corpus.")
        return build_corpus_with_images(ARGS.image_csv, ARGS.image_dir, max_samples=ARGS.max_samples)

    # If user requested use_images and dataset is 'hf_images', attempt HF image harvest
    if ARGS.use_images and ARGS.dataset == "hf_images":
        # attempt to fetch common agriculture/image datasets by name provided in mix_sources
        sources = [s.strip() for s in ARGS.mix_sources.split(",") if s.strip()]
        out_parts=[]
        for src in sources:
            try:
                dfp = prepare_images_from_hf(src, ARGS.max_per_source, ARGS.image_dir)
                if len(dfp)>0:
                    out_parts.append(dfp)
            except Exception as e:
                print(f"[Images] error preparing {src}: {e}")
        if out_parts:
            df_full = pd.concat(out_parts, ignore_index=True)
            if ARGS.max_samples and len(df_full) > ARGS.max_samples:
                df_full = df_full.sample(ARGS.max_samples, random_state=SEED).reset_index(drop=True)
            print(f"[Build] final multimodal size: {len(df_full)}")
            return df_full

    # Fall back to text-only mix pipeline (same as before)
    if ARGS.dataset=="mix":
        print("[Dataset] MIX:", ARGS.mix_sources)
        df = build_mix(ARGS.max_per_source, ARGS.mqtt_csv, ARGS.extra_csv)
    elif ARGS.dataset=="localmini":
        df = build_localmini(ARGS.max_samples or 0, ARGS.mqtt_csv, ARGS.extra_csv)
    elif ARGS.dataset=="gardian":
        raws = build_gardian_stream(ARGS.max_per_source)
        rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
        df = pd.DataFrame([(t,l) for (t,l) in rows if l], columns=["text","labels"])
    elif ARGS.dataset=="argilla":
        raws = build_argilla_stream(ARGS.max_per_source)
        rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
        df = pd.DataFrame([(t,l) for (t,l) in rows if l], columns=["text","labels"])
    else:
        raws = build_agnews_agri(ARGS.max_per_source)
        rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
        df = pd.DataFrame([(t,l) for (t,l) in rows if l], columns=["text","labels"])

    summarize_labels(df, "pre-oversample")
    df = apply_label_noise(df, ARGS.label_noise)
    # Add a few OOD negatives
    ood = [
        "City council discussed budget allocations for public transport.",
        "The software team published patch notes for the new release.",
        "The arts festival announced its opening night lineup."
    ]
    for t in ood:
        df.loc[len(df)] = [fuse_text(simulate_sensor_summary(), t), []]
    if ARGS.max_samples and len(df)>ARGS.max_samples:
        df = df.sample(ARGS.max_samples, random_state=SEED)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[Build] final size (text-only): {len(df)}")
    if len(df)==0: raise RuntimeError("Empty dataset after filtering. Include localmini or lower caps.")
    return df

# --------------------- Oversampling & helpers (same) ---------------------
def oversample_by_class(df: pd.DataFrame, target_each_map: Dict[int,int] = None) -> pd.DataFrame:
    if target_each_map is None:
        target_each_map = {0:1500, 1:2400, 2:1700, 3:1700, 4:1500}
    idxs = {i: [] for i in range(NUM_LABELS)}
    for idx, labs in enumerate(df["labels"]):
        for k in labs: idxs[k].append(idx)
    keep=[]
    for k, tgt in target_each_map.items():
        pool = idxs[k]
        if not pool: continue
        if len(pool) >= tgt:
            keep.extend(random.sample(pool, tgt))
        else:
            need = tgt - len(pool)
            keep.extend(pool + random.choices(pool, k=need))
    keep = sorted(set(keep))
    out = df.iloc[keep].copy()
    return out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def summarize_labels(df: pd.DataFrame, tag="set"):
    counts = np.zeros(NUM_LABELS, int)
    for labs in df["labels"]:
        for k in labs: counts[k]+=1
    print(f"[{tag}] label counts:", {ISSUE_LABELS[i]: int(c) for i,c in enumerate(counts)})

def apply_label_noise(df: pd.DataFrame, p: float) -> pd.DataFrame:
    if p <= 0: return df
    rng = np.random.default_rng(SEED)
    rows = []
    for t, labs in df[["text","labels"]].itertuples(index=False):
        if rng.random() < p:
            labs = labs.copy()
            if labs and rng.random() < 0.5:
                del labs[rng.integers(0, len(labs))]
            else:
                k = rng.integers(0, NUM_LABELS)
                if k not in labs: labs.append(k)
            labs = sorted(set(labs))
        rows.append((t, labs))
    return pd.DataFrame(rows, columns=["text","labels"])

# --------------------- Dataset classes (text & multimodal) ---------------------
class MultiLabelDS(Dataset):
    def __init__(self, df, tok, max_len):
        self.df=df; self.tok=tok; self.max_len=max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = row["text"]
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        y = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in row["labels"]: y[k]=1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": y,
            "raw_text": text,
        }

class MultiModalDS(Dataset):
    def __init__(self, df, tok, max_len, img_size=224, image_root=""):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len
        self.img_size = img_size
        self.image_root = image_root or ""
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.df)
    def _load_image(self, path: str):
        if not path or (isinstance(path, float) and pd.isna(path)): 
            return torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        p = path if os.path.isabs(path) else os.path.join(self.image_root, path)
        try:
            im = Image.open(p).convert("RGB")
            return self.transform(im)
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = row["text"]
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        y = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in row["labels"]: y[k]=1.0
        img_path = row.get("image_path", "")
        img = self._load_image(str(img_path) if not pd.isna(img_path) else "")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": img,
            "labels": y,
            "raw_text": text,
            "image_path": img_path
        }

def make_weights_for_balanced_classes(df:pd.DataFrame):
    counts = np.zeros(NUM_LABELS)
    for labs in df["labels"]:
        for k in labs: counts[k]+=1
    inv = 1.0 / np.maximum(counts, 1)
    inst_w = []
    for labs in df["labels"]:
        w = np.mean([inv[k] for k in labs]) if labs else np.mean(inv)
        inst_w.append(w)
    inst_w = np.array(inst_w, dtype=np.float32)
    inst_w = inst_w / (inst_w.mean() + 1e-12)
    return inst_w, counts

# --------------------- Loss ---------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha:torch.Tensor=None, gamma=2.5, label_smoothing=0.02):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p*targets + (1-p)*(1-targets)
        loss = ((1-pt)**self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha.view(1, -1)
        return loss.mean()

# --------------------- Tokenizer + models + LoRA ---------------------
def build_tokenizer():
    return AutoTokenizer.from_pretrained(ARGS.model_name, local_files_only=ARGS.offline)

def infer_lora_targets_from_model(model) -> List[str]:
    names = [n for n,_ in model.named_modules()]
    cand_sets = [
        ["q_lin","k_lin","v_lin","out_lin"],               # DistilBERT
        ["query","key","value","dense"],                  # BERT/RoBERTa
        ["query_proj","key_proj","value_proj","o_proj"],  # DeBERTa v3
    ]
    for cands in cand_sets:
        found = [c for c in cands if any(("."+c) in n or n.endswith(c) for n in names)]
        if len(found) >= 2: return found
    return ["classifier"]

def build_text_model(num_labels:int, freeze_base:bool=True):
    kwargs = dict(num_labels=num_labels, problem_type="multi_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(
        ARGS.model_name, **kwargs, local_files_only=ARGS.offline
    )
    if freeze_base and hasattr(model, "base_model"):
        for p in model.base_model.parameters(): p.requires_grad=False
    elif freeze_base:
        for n,p in model.named_parameters():
            if "classifier" not in n: p.requires_grad=False
    targets = infer_lora_targets_from_model(model)
    lcfg = LoraConfig(r=ARGS.lora_r, lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout,
                      bias="none", task_type="SEQ_CLS", target_modules=targets)
    model = get_peft_model(model, lcfg)
    print(f"[LoRA] target_modules: {targets}")
    return model

class MultiModalModel(nn.Module):
    """
    text_peft: AutoModel wrapped with PEFT (LoRA)
    vision: ViTModel
    classifier: MLP on fused features -> NUM_LABELS logits
    """
    def __init__(self, text_model_name, vit_name, num_labels,
                 freeze_text=True, freeze_vision=False,
                 lora_r=8, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        # text encoder (AutoModel base) -> then apply peft
        text_base = AutoModel.from_pretrained(text_model_name, local_files_only=ARGS.offline)
        if freeze_text:
            for p in text_base.parameters(): p.requires_grad = False
        targets = infer_lora_targets_from_model(text_base)
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          bias="none", task_type="SEQ_CLS", target_modules=targets)
        text_peft = get_peft_model(text_base, lcfg)
        self.text_encoder = text_peft
        text_dim = getattr(self.text_encoder.config, "hidden_size", 768)

        # vision encoder (ViT)
        self.vision = ViTModel.from_pretrained(vit_name, local_files_only=ARGS.offline)
        if freeze_vision:
            for p in self.vision.parameters(): p.requires_grad = False
        vision_dim = getattr(self.vision.config, "hidden_size", 768)

        fusion_dim = text_dim + vision_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, max(512, fusion_dim//2)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(max(512, fusion_dim//2), num_labels)
        )
        print(f"[Model] text_dim={text_dim} vision_dim={vision_dim} lora_targets={targets}")

    def forward(self, input_ids=None, attention_mask=None, image=None):
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(txt_out, "pooler_output") and txt_out.pooler_output is not None:
            tfeat = txt_out.pooler_output
        else:
            tfeat = txt_out.last_hidden_state.mean(dim=1)
        if image is None:
            vfeat = torch.zeros(tfeat.size(0), self.vision.config.hidden_size, device=tfeat.device)
        else:
            vit_out = self.vision(pixel_values=image, return_dict=True)
            if hasattr(vit_out, "pooler_output") and vit_out.pooler_output is not None:
                vfeat = vit_out.pooler_output
            else:
                vfeat = vit_out.last_hidden_state.mean(dim=1)
        feat = torch.cat([tfeat, vfeat], dim=1)
        logits = self.classifier(feat)
        return type("O", (), {"logits": logits})

def amp_enabled(): return torch.cuda.is_available()

# --------------------- Calibration & Metrics + Plots ---------------------
def calibrate_thresholds(model, loader, precision_target=0.90) -> np.ndarray:
    model.eval(); model.to(DEVICE)
    probs_all=[]; y_all=[]
    with torch.no_grad():
        for b in loader:
            bt = {k:v.to(DEVICE) for k,v in b.items() if k not in ("labels","raw_text")}
            out = model(**bt)
            logits = out.logits
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            y_all.append(b["labels"].numpy())
    P = np.vstack(probs_all); T = np.vstack(y_all)
    C = P.shape[1]; thr = np.zeros(C, dtype=np.float32)
    for j in range(C):
        col, y = P[:, j], T[:, j].astype(int)
        best_t_f1, best_f1 = 0.5, -1.0
        best_t_prec = None
        for t in np.linspace(0.05, 0.9, 35):
            pred = (col >= t).astype(int)
            prec = precision_score(y, pred, zero_division=0)
            f1v  = f1_score(y, pred, zero_division=0)
            if prec >= precision_target:
                if best_t_prec is None or f1v > best_f1:
                    best_t_prec, best_f1 = t, f1v
            if f1v > best_f1:
                best_t_f1, best_f1 = t, f1v
        thr[j] = best_t_prec if best_t_prec is not None else best_t_f1
    thr = np.clip(thr, 0.20, 0.80)
    return thr

def evaluate_with_thr(model, loader, thr) -> Dict[str,float]:
    def _cap(x): return min(0.999, float(x))
    model.eval(); model.to(DEVICE)
    P_all=[]; T_all=[]; R_all=[]
    with torch.no_grad():
        for b in loader:
            bt = {k:v.to(DEVICE) for k,v in b.items() if k not in ("labels","raw_text")}
            out = model(**bt)
            logits = out.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= thr).astype(int)
            P_all.append(preds); T_all.append(b["labels"].numpy()); R_all.append(probs)
    P = np.vstack(P_all); T = np.vstack(T_all); R = np.vstack(R_all)
    micro = f1_score(T, P, average="micro", zero_division=0)
    macro = f1_score(T, P, average="macro", zero_division=0)
    prec  = precision_score(T, P, average=None, zero_division=0)
    rec   = recall_score(T, P, average=None, zero_division=0)
    f1s   = [f1_score(T[:,i], P[:,i], zero_division=0) for i in range(NUM_LABELS)]
    supports = T.sum(axis=0)
    if not ARGS.quiet_eval:
        print("\nPer-class metrics:")
        for i, lab in enumerate(ISSUE_LABELS):
            if supports[i] < 20:
                print(f" - {lab:14s} | insufficient support (n={int(supports[i])})")
                continue
            print(f" - {lab:14s} | P={_cap(prec[i]):.3f} R={_cap(rec[i]):.3f} F1={_cap(f1s[i]):.3f} thr={thr[i]:.2f}")
        print(f"\nOverall: micro-F1={_cap(micro):.3f}  macro-F1={_cap(macro):.3f}")
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            pr_micro = average_precision_score(T, R, average="micro")
            pr_macro = average_precision_score(T, R, average="macro")
            roc_micro = roc_auc_score(T, R, average="micro")
            roc_macro = roc_auc_score(T, R, average="macro")
            print(f"AUPRC micro={_cap(pr_micro):.3f} macro={_cap(pr_macro):.3f} | "
                  f"AUROC micro={_cap(roc_micro):.3f} macro={_cap(roc_macro):.3f}")
        except Exception:
            pass
    return {"micro_f1": micro, "macro_f1": macro,
            "per_class": {"precision": prec, "recall": rec, "f1": np.array(f1s)}}

def save_tables_and_plots(metrics: Dict[str, float], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    prec = metrics["per_class"]["precision"]
    rec  = metrics["per_class"]["recall"]
    f1s  = metrics["per_class"]["f1"]
    df = pd.DataFrame({"Label": ISSUE_LABELS, "Precision": prec, "Recall": rec, "F1": f1s})
    csv_path = os.path.join(save_dir, "results_table.csv")
    df.to_csv(csv_path, index=False)
    plt.figure(figsize=(8,5))
    plt.bar(ISSUE_LABELS, f1s)
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.title("Class-wise F1")
    png_path = os.path.join(save_dir, "f1_bar.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {png_path}")

# --------------------- Federated utils ---------------------
def split_clients(df: pd.DataFrame, n: int, alpha: float) -> List[pd.DataFrame]:
    prim = []
    rng = np.random.default_rng(SEED)
    for labs in df["labels"]:
        if labs: prim.append(int(rng.choice(labs)))
        else:    prim.append(int(rng.integers(0, NUM_LABELS)))
    df2 = df.copy(); df2["_y"] = prim
    class_client_probs = rng.dirichlet([alpha]*n, size=NUM_LABELS)
    client_bins = [[] for _ in range(n)]
    for idx, y in enumerate(df2["_y"].tolist()):
        k = int(rng.choice(n, p=class_client_probs[y]))
        client_bins[k].append(idx)
    out = []
    for k in range(n):
        part = df2.iloc[client_bins[k]].drop(columns=["_y"]).reset_index(drop=True)
        out.append(part)
    return out

def ema_update(ema_params, model_params, decay):
    for ep, mp in zip(ema_params, model_params):
        ep.data.mul_((decay)).add_(mp.data, alpha=1.0 - decay)

def _split_local_train_val(cdf: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame,pd.DataFrame]:
    n = len(cdf)
    val_n = max(1, int(frac * n))
    val_n = min(val_n, max(1, n-1))
    va_df = cdf.iloc[:val_n].reset_index(drop=True)
    tr_df = cdf.iloc[val_n:].reset_index(drop=True) if n > val_n else cdf.iloc[:1].reset_index(drop=True)
    return va_df, tr_df

def train_local(model, tok, tr_df, va_df, class_alpha:torch.Tensor) -> Tuple[float,float,Dict,np.ndarray,int]:
    # choose dataset type based on model: multimodal models have attribute 'vision'
    if ARGS.use_images and "image_path" in tr_df.columns:
        tr_ds = MultiModalDS(tr_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir)
        va_ds = MultiModalDS(va_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir)
    else:
        tr_ds = MultiLabelDS(tr_df, tok, ARGS.max_len)
        va_ds = MultiLabelDS(va_df, tok, ARGS.max_len)

    weights, _ = make_weights_for_balanced_classes(tr_df)
    sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double),
                                    num_samples=max(len(tr_df), ARGS.batch_size), replacement=True)
    tr_loader = DataLoader(tr_ds, batch_size=ARGS.batch_size, sampler=sampler, num_workers=0, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=max(ARGS.batch_size, 16), shuffle=False, num_workers=0, drop_last=False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ARGS.lr, weight_decay=0.05)
    steps_per_epoch = max(1, math.ceil(len(tr_loader)/max(1, ARGS.grad_accum)))
    total_steps = ARGS.local_epochs * steps_per_epoch
    sch = get_linear_schedule_with_warmup(opt, max(1,int(0.1*total_steps)), total_steps)

    loss_fn = FocalLoss(alpha=class_alpha.to(DEVICE), gamma=2.5, label_smoothing=0.02)
    model.train(); model.to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ema_params = [p.detach().clone() for p in trainable_params]

    opt.zero_grad(set_to_none=True)
    for _ in range(ARGS.local_epochs):
        for it, batch in enumerate(tr_loader, start=1):
            # prepare inputs
            text_inputs = {"input_ids": batch["input_ids"].to(DEVICE), "attention_mask": batch["attention_mask"].to(DEVICE)}
            img = batch.get("image", None)
            if img is not None:
                img = img.to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=amp_enabled()):
                if img is not None:
                    out = model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"], image=img)
                else:
                    out = model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
                loss = loss_fn(out.logits, labels) / max(1, ARGS.grad_accum)
            scaler.scale(loss).backward()
            if it % max(1, ARGS.grad_accum) == 0:
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sch.step()
                ema_update(ema_params, trainable_params, ARGS.ema_decay)

    # Eval with EMA weights
    backup = [p.detach().clone() for p in trainable_params]
    for p, ep in zip(trainable_params, ema_params): p.data.copy_(ep.data)
    thr = calibrate_thresholds(model, va_loader, precision_target=ARGS.precision_target)
    was_quiet = ARGS.quiet_eval
    ARGS.quiet_eval = True
    mets = evaluate_with_thr(model, va_loader, thr)
    ARGS.quiet_eval = was_quiet
    micro_f1, macro_f1 = mets["micro_f1"], mets["macro_f1"]
    for p, bp in zip(trainable_params, backup): p.data.copy_(bp.data)

    # extract LoRA state dict (model may be peft-wrapped)
    try:
        lora_sd = get_peft_model_state_dict(model)
    except Exception:
        # fallback: whole model state dict
        lora_sd = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    lora_sd = {k:v.detach().cpu() for k,v in lora_sd.items()}
    del tr_loader, va_loader; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return micro_f1, macro_f1, lora_sd, thr, len(tr_df)

def fedavg_weighted(states: List[Dict[str,torch.Tensor]], sizes: List[int]) -> Dict[str,torch.Tensor]:
    total=float(sum(sizes)); ws=[s/total for s in sizes]
    keys = list(states[0].keys()); out={}
    for k in keys:
        out[k]=torch.stack([st[k].float()*w for st,w in zip(states,ws)], dim=0).sum(0)
    return out

# --------------------- Advisor & prediction ---------------------
ADVICE = {
    "water_stress": "Irrigate earlier; mulch; monitor soil moisture AM/PM.",
    "nutrient_def": "Balance NPK (N focus if older leaves yellow); verify with LCC.",
    "pest_risk": "Inspect undersides; sticky traps; early biocontrol or mild soap.",
    "disease_risk": "Improve airflow; avoid late overhead irrigation; prune infected leaves.",
    "heat_stress": "Provide shade at peak heat; keep moisture stable; ensure K sufficiency.",
}
def advisor(pred_mask: List[int]) -> str:
    active=[ISSUE_LABELS[i] for i,v in enumerate(pred_mask) if v==1]
    if not active: return "Conditions look normal. Continue routine monitoring."
    return "Recommended actions:\n" + "\n".join([f"- {k}: {ADVICE[k]}" for k in active])

# --------------------- MC-dropout & multi-hypothesis utilities ---------------------
def _hit_keywords(txt: str, bag: Dict[str, list]) -> Dict[str, list]:
    t = txt.lower()
    hits = {}
    def any_in(words): return [w for w in words if w in t]
    hits["water_stress"]   = any_in(KW["water"])
    hits["nutrient_def"]   = any_in(KW["nutrient"])
    hits["pest_risk"]      = any_in(KW["pest"])
    hits["disease_risk"]   = any_in(KW["disease"])
    hits["heat_stress"]    = any_in(KW["heat"])
    return hits

@torch.no_grad()
def mc_predict_probs(model, tok, texts: list, images: Optional[List[torch.Tensor]]=None, T: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    model.train()  # enable dropout
    enc = tok(texts, truncation=True, max_length=ARGS.max_len, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE); attention_mask = enc["attention_mask"].to(DEVICE)
    img_tensor = None
    if ARGS.use_images and images is not None:
        if isinstance(images[0], torch.Tensor):
            img_tensor = torch.stack(images).to(DEVICE)
        else:
            img_tensor = torch.stack(images).to(DEVICE)
    probs_stack = []
    for _ in range(max(1, T)):
        if img_tensor is not None:
            out = model(input_ids=input_ids, attention_mask=attention_mask, image=img_tensor)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, image=None)
        logits = apply_priors_to_logits(out.logits, texts)
        probs_stack.append(torch.sigmoid(logits).cpu().numpy())
    P = np.stack(probs_stack, axis=0)
    return P.mean(axis=0), P.std(axis=0)

def rank_labels(mean_probs: np.ndarray, thr: np.ndarray, topk: int = 3, epsilon: float = 0.05):
    C = len(ISSUE_LABELS)
    items = []
    for i in range(C):
        p = float(mean_probs[i]); m = float(p - float(thr[i]))
        items.append((ISSUE_LABELS[i], p, m, float(thr[i])))
    active = [(l,p,m) for (l,p,m,t) in items if p >= t]
    near   = [(l,p,m) for (l,p,m,t) in items if (p < t) and (t - p) <= epsilon]
    top    = sorted([(l,p) for (l,p,_,_) in items], key=lambda x: x[1], reverse=True)[:max(1, topk)]
    active = sorted(active, key=lambda x: (x[2], x[1]), reverse=True)
    near   = sorted(near,   key=lambda x: x[1], reverse=True)
    return {"active": active, "near": near, "top": top}

def build_rationales(text: str, mean_probs: np.ndarray, std_probs: np.ndarray, thr: np.ndarray):
    sensors = _parse_sensors(text) or {}
    kw_hits = _hit_keywords(text, KW)
    r = {}
    for i, lab in enumerate(ISSUE_LABELS):
        p = float(mean_probs[i]); s = float(std_probs[i]); t = float(thr[i])
        reasons = []
        hits = kw_hits.get(lab, [])
        if hits: reasons.append(f"keywords: {', '.join(hits[:3])}")
        if sensors:
            sm, vpd, h, t_air = sensors.get("sm"), sensors.get("vpd"), sensors.get("h"), sensors.get("t")
            if lab=="water_stress" and ((sm is not None and sm <= 20) or (vpd is not None and vpd >= 2.0)):
                reasons.append("sensor: low soil moisture / high VPD")
            if lab=="heat_stress" and ((t_air is not None and t_air >= 36) or (vpd is not None and vpd >= 2.2)):
                reasons.append("sensor: high temperature / VPD")
            if lab=="disease_risk" and ((h is not None and h >= 70) or (sensors.get("rf",0) >= 2.0)):
                reasons.append("sensor: humid/wet conditions")
            if lab=="pest_risk" and (h is not None and 45 <= h <= 70):
                reasons.append("sensor: pest-favorable humidity window")
            if lab=="nutrient_def" and (sensors.get("ph") is not None) and (sensors["ph"] < 5.8 or sensors["ph"] > 7.4):
                reasons.append("sensor: off-range pH")
        reasons.append(f"p={p:.2f} (σ={s:.02f}) vs thr={t:.2f}")
        r[lab] = "; ".join(reasons)
    return r

def pretty_multihypo_output(text: str, ranked: dict, rationales: dict):
    out = []
    out.append("PRED (multi-label):")
    if ranked["active"]:
        out.append(" • Active:")
        for lab, p, m in ranked["active"]:
            out.append(f"   - {lab}: p={p:.2f}  Δ={m:+.02f}  ⇒ {ADVICE[lab]}")
            if rationales.get(lab): out.append(f"     · why: {rationales[lab]}")
    else:
        out.append(" • Active: (none above thresholds)")
    if ranked["near"]:
        out.append(" • Near-miss (watchlist):")
        for lab, p, m in ranked["near"]:
            out.append(f"   - {lab}: p={p:.2f}  Δ={m:+.02f}  (monitor / consider preventive steps)")
            if rationales.get(lab): out.append(f"     · why: {rationales[lab]}")
    if ranked["top"]:
        lbls = ", ".join([f"{l}({p:.2f})" for (l,p) in ranked["top"]])
        out.append(f" • Top-k by probability: {lbls}")
    return "\n".join(out)

# --------------------- predict / training drivers ---------------------
def predict(model, tok, text:str, thr:np.ndarray, image_tensor:Optional[torch.Tensor]=None) -> List[int]:
    model.eval(); model.to(DEVICE)
    enc = tok(text, truncation=True, max_length=ARGS.max_len, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        if ARGS.use_images and image_tensor is not None:
            out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE), image=image_tensor.to(DEVICE))
        else:
            out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE))
        logits = apply_priors_to_logits(out.logits, [text])
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return (probs >= thr).astype(int).tolist()

def run_training():
    print(f"Device: {DEVICE} (AMP={'on' if amp_enabled() else 'off'})  model={ARGS.model_name}  use_images={ARGS.use_images}")
    tok = build_tokenizer()
    df  = build_corpus()

    # If multimodal dataset (image_path present) then we'll use multimodal model
    multimodal = ARGS.use_images and ("image_path" in df.columns or (ARGS.image_csv and os.path.exists(ARGS.image_csv)))

    # Client-held-out validation
    clients_all = split_clients(df, max(1, ARGS.clients), ARGS.dirichlet_alpha)
    val_k = max(1, int(0.15 * len(clients_all)))
    val_df = pd.concat(clients_all[:val_k], ignore_index=True)
    train_clients = clients_all[val_k:]
    train_df = pd.concat(train_clients, ignore_index=True)
    train_df, test_df = train_test_split(train_df, test_size=0.15, random_state=SEED, shuffle=True)

    # class alphas
    _, counts = make_weights_for_balanced_classes(train_df)
    inv = 1.0 / np.maximum(counts,1)
    alpha = (inv / inv.mean()).astype(np.float32)
    alpha[1] *= 1.2
    alpha = torch.tensor(alpha)

    # federated clients
    clients = split_clients(train_df, max(1, ARGS.clients), ARGS.dirichlet_alpha)

    # build global model
    if multimodal:
        global_model = MultiModalModel(ARGS.model_name, ARGS.vit_name, NUM_LABELS,
                                       freeze_text=ARGS.freeze_base, freeze_vision=ARGS.freeze_vision,
                                       lora_r=ARGS.lora_r, lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout).to(DEVICE)
    else:
        global_model = build_text_model(NUM_LABELS, freeze_base=ARGS.freeze_base).to(DEVICE)

    # Validation loader for calibration uses text-only loader unless multimodal val_df contains images
    if "image_path" in val_df.columns and ARGS.use_images:
        val_loader = DataLoader(MultiModalDS(val_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir), batch_size=32, shuffle=False)
    else:
        val_loader = DataLoader(MultiLabelDS(val_df, tok, ARGS.max_len), batch_size=32, shuffle=False)

    metrics_dir = os.path.join(ARGS.save_dir, "metrics"); os.makedirs(metrics_dir, exist_ok=True)
    def evaluate_global(thr):
        if "image_path" in test_df.columns and ARGS.use_images:
            test_loader = DataLoader(MultiModalDS(test_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir), batch_size=64, shuffle=False)
        else:
            test_loader = DataLoader(MultiLabelDS(test_df, tok, ARGS.max_len), batch_size=64, shuffle=False)
        return evaluate_with_thr(global_model, test_loader, thr)

    thr_history = []
    for r in range(1, max(1, ARGS.rounds)+1):
        print(f"\n==== Round {r}/{ARGS.rounds} ====")
        rng = np.random.default_rng(SEED + r)
        k_all = list(range(len(clients))); rng.shuffle(k_all)
        m = max(1, int(ARGS.participation * len(k_all))); chosen = k_all[:m]

        states, sizes = [], []
        for i in chosen:
            if rng.random() < ARGS.client_dropout:
                print(f"[Client {i+1}] dropped this round"); continue
            cdf = clients[i]
            if len(cdf) < 80:
                print(f"[Client {i+1}] skipped (too small: n={len(cdf)})"); continue
            n = len(cdf); val_n = max(1,int(ARGS.val_frac * n))
            va_df, tr_df = cdf.iloc[:val_n], cdf.iloc[val_n:]
            if multimodal:
                local = MultiModalModel(ARGS.model_name, ARGS.vit_name, NUM_LABELS,
                                        freeze_text=ARGS.freeze_base, freeze_vision=ARGS.freeze_vision,
                                        lora_r=ARGS.lora_r, lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout).to(DEVICE)
                # set local adapters from global (global_model is peft-wrapped)
                try:
                    set_peft_model_state_dict(local.text_encoder, get_peft_model_state_dict(global_model.text_encoder))
                except Exception:
                    # fallback: set whole model state if possible
                    try: set_peft_model_state_dict(local, get_peft_model_state_dict(global_model))
                    except Exception: pass
            else:
                local = build_text_model(NUM_LABELS, freeze_base=ARGS.freeze_base).to(DEVICE)
                try:
                    set_peft_model_state_dict(local, get_peft_model_state_dict(global_model))
                except Exception:
                    pass

            rng_local = np.random.default_rng(SEED + r + i)
            local_epochs = int(rng_local.choice([2,3,4], p=[0.4,0.4,0.2]))
            orig_local = ARGS.local_epochs; ARGS.local_epochs = local_epochs

            micro, macro, lora_sd, thr_local, used_n = train_local(local, tok, tr_df, va_df, class_alpha=alpha)
            ARGS.local_epochs = orig_local

            print(f"[Client {i+1}] micro_f1={_fmt_str(micro)} macro_f1={_fmt_str(macro)} (n={len(cdf)}) thr={np.round(thr_local,2)}")
            states.append(lora_sd); sizes.append(used_n)

            del local; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if states:
            avg_sd = fedavg_weighted(states, sizes)
            # set averaged adapters into global model
            try:
                # if peft-wrapped text encoder exists
                if multimodal:
                    set_peft_model_state_dict(global_model.text_encoder, avg_sd)
                else:
                    set_peft_model_state_dict(global_model, avg_sd)
            except Exception:
                # fallback: load into whole model state dict
                try:
                    global_model.load_state_dict(avg_sd, strict=False)
                except Exception:
                    print("[Warn] couldn't set averaged peft state directly; skipping this round.")
            final_thr = calibrate_thresholds(global_model, val_loader, precision_target=ARGS.precision_target)
            final_thr = np.clip(final_thr + np.array([+0.03, 0.00, 0.00, +0.02, 0.00]), 0.05, 0.90)
        else:
            print("No client updates this round; keeping previous thresholds.")
            final_thr = thr_history[-1] if thr_history else np.array([0.5]*NUM_LABELS)

        thr_history.append(final_thr)
        test_mets = evaluate_global(final_thr)
        round_tag = f"round_{r:02d}"
        np.save(os.path.join(metrics_dir, f"{round_tag}_thr.npy"), final_thr)
        with open(os.path.join(metrics_dir, f"{round_tag}_summary.json"), "w") as f:
            json.dump({"round": r, "micro_f1": float(test_mets["micro_f1"]), "macro_f1": float(test_mets["macro_f1"])}, f, indent=2)

    # Save final adapters + thresholds
    ap = os.path.join(ARGS.save_dir, "global_lora.pt")
    thp = os.path.join(ARGS.save_dir, "thresholds.npy")
    try:
        if multimodal:
            torch.save(get_peft_model_state_dict(global_model.text_encoder), ap)
        else:
            torch.save(get_peft_model_state_dict(global_model), ap)
    except Exception:
        # fallback save whole state dict
        torch.save(global_model.state_dict(), ap)
    np.save(thp, thr_history[-1] if thr_history else np.array([0.5]*NUM_LABELS))
    print(f"[Save] adapters → {ap}")
    print(f"[Save] thresholds → {thp}")

    # Final diagnostics: make sure to use correct loader type
    if "image_path" in test_df.columns and ARGS.use_images:
        cal_df = test_df.sample(min(800, len(test_df)), random_state=SEED)
        cal_loader = DataLoader(MultiModalDS(cal_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir), batch_size=32, shuffle=False)
    else:
        cal_df = test_df.sample(min(800, len(test_df)), random_state=SEED)
        cal_loader = DataLoader(MultiLabelDS(cal_df, tok, ARGS.max_len), batch_size=32, shuffle=False)
    mets = evaluate_with_thr(global_model, cal_loader, thr_history[-1] if thr_history else np.array([0.5]*NUM_LABELS))
    save_tables_and_plots(mets, save_dir=os.path.join(ARGS.save_dir, "figs"))

    for s in cal_df.sample(min(3, len(cal_df)), random_state=SEED)["text"].tolist():
        pred = predict(global_model, tok, s, thr_history[-1] if thr_history else np.array([0.5]*NUM_LABELS))
        print("\nTEXT:\n", s[:280], "...")
        print("PRED:", [ISSUE_LABELS[i] for i,v in enumerate(pred) if v==1])
        print(advisor(pred))

def run_inference():
    tok   = build_tokenizer()
    # Build model: choose multimodal or text-only based on saved artifacts / args
    multimodal = ARGS.use_images
    if multimodal:
        model = MultiModalModel(ARGS.model_name, ARGS.vit_name, NUM_LABELS,
                                freeze_text=ARGS.freeze_base, freeze_vision=ARGS.freeze_vision,
                                lora_r=ARGS.lora_r, lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout).to(DEVICE)
    else:
        model = build_text_model(NUM_LABELS, freeze_base=ARGS.freeze_base).to(DEVICE)

    ap  = os.path.join(ARGS.save_dir, "global_lora.pt")
    thp = os.path.join(ARGS.save_dir, "thresholds.npy")
    if os.path.exists(ap):
        try:
            if multimodal:
                set_peft_model_state_dict(model.text_encoder, torch.load(ap, map_location="cpu"))
            else:
                set_peft_model_state_dict(model, torch.load(ap, map_location="cpu"))
            print(f"[Load] adapters {ap}")
        except Exception:
            try:
                model.load_state_dict(torch.load(ap, map_location="cpu"), strict=False)
                print(f"[Load] model state {ap}")
            except Exception:
                print("[Warn] couldn't load adapters; proceeding with random adapters.")
    else:
        print("[Warn] adapters not found; using random adapters.")
    thr = np.load(thp) if os.path.exists(thp) else np.array([0.5]*NUM_LABELS)

    sensors = (f"SENSORS: {', '.join([x.strip() for x in ARGS.sensors.split(',')])}." if ARGS.sensors.strip() else simulate_sensor_summary())
    q = ARGS.query.strip() or "Slight leaf curling after hot afternoons; few insects seen under leaves."
    text = f"{sensors}\nLOG: {q}"

    # MC-Dropout mean ± std
    mean_probs, std_probs = mc_predict_probs(model, tok, [text], T=max(1, ARGS.samples))
    mean_probs, std_probs = mean_probs[0], std_probs[0]
    ranked     = rank_labels(mean_probs, thr, topk=max(1, ARGS.topk), epsilon=max(0.0, ARGS.epsilon))
    rationales = build_rationales(text, mean_probs, std_probs, thr)
    print("TEXT:\n", text[:400], "...")
    print(pretty_multihypo_output(text, ranked, rationales))

# --------------------- Main ---------------------
if __name__=="__main__":
    if ARGS.inference: run_inference()
    else: run_training()
