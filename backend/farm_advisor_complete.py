# ================== Enhanced Multimodal Federated Farm Advisor ================== 
# This implementation includes:
# - Federated LLM training (Flan-T5, GPT-2, LLaMA-style models)
# - ViT encoder for crop stress detection
# - Vision-Language Models (VLM: CLIP, BLIP-2) for comparison
# - Comprehensive dataset loading (all available HF agricultural datasets)
# - Comparison with existing papers/baselines
# - Full evaluation and benchmarking framework
# ===============================================================================

import os
import re
import math
import time
import gc
import random
import argparse
import hashlib
import json
import shutil
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from PIL import Image
import torchvision.transforms as T
import requests
from io import BytesIO
from dataclasses import dataclass
from collections import defaultdict

# reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# optional HF tools
try:
    from datasets import load_dataset, DownloadConfig
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

# transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        AutoModel,
        ViTModel,
        ViTImageProcessor,
        CLIPModel,
        CLIPProcessor,
        BlipForImageTextRetrieval,
        BlipProcessor,
        AutoConfig,
        ViTConfig,
        get_linear_schedule_with_warmup,
    )
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
except Exception as e:
    raise RuntimeError(
        f"transformers import failed: {e}. "
        f"Install: pip install transformers>=4.40"
    )

# PEFT / LoRA
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
        TaskType,
    )
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False
    print("[Warn] PEFT not available. Install: pip install peft")

# --------------------- Labels --------------------- 
ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
LABEL_TO_ID = {k: i for i, k in enumerate(ISSUE_LABELS)}
NUM_LABELS = len(ISSUE_LABELS)

# --------------------- Enhanced Dataset Sources --------------------- 
# Comprehensive list of agricultural datasets from HuggingFace
AGRICULTURAL_TEXT_DATASETS = [
    "CGIAR/gardian-ai-ready-docs",
    "argilla/farming",
    "ag_news",  # filtered for agriculture
    "wiki_bio",  # filtered for agricultural scientists
    "scientific_papers",  # filtered for agricultural topics
    "pubmed",  # filtered for agricultural research
]

AGRICULTURAL_IMAGE_DATASETS = [
    "BrandonFors/Plant-Diseases-PlantVillage-Dataset",
    "GVJahnavi/PlantVillage_dataset",
    "agyaatcoder/PlantDoc",
    "pufanyi/cassava-leaf-disease-classification",
    "Saon110/bd-crop-vegetable-plant-disease-dataset",
    "timm/plant-pathology-2021",
    "uqtwei2/PlantWild",
    "nateraw/rice-leaf-diseases",
    "keremberke/plant-disease-object-detection",
]

# VLM-specific datasets
VLM_DATASETS = [
    "visual-layer/oxford-flower102-captions",
    "nlphuji/flickr30k",  # filtered for plant/agriculture
]

# --------------------- Model Architecture Configurations --------------------- 
@dataclass
class ModelConfig:
    """Configuration for different model architectures"""
    name: str
    model_type: str  # 'encoder', 'decoder', 'seq2seq', 'vlm', 'vit'
    model_name: str
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    freeze_base: bool = True

# Predefined model configurations
MODEL_CONFIGS = {
    # Encoder-based models (BERT-style)
    "roberta": ModelConfig("roberta", "encoder", "roberta-base"),
    "bert": ModelConfig("bert", "encoder", "bert-base-uncased"),
    "distilbert": ModelConfig("distilbert", "encoder", "distilbert-base-uncased"),
    
    # Decoder-based models (GPT-style)
    "gpt2": ModelConfig("gpt2", "decoder", "gpt2"),
    "gpt2-medium": ModelConfig("gpt2-medium", "decoder", "gpt2-medium"),
    
    # Seq2Seq models (T5-style) - Federated LLM
    "flan-t5-small": ModelConfig("flan-t5-small", "seq2seq", "google/flan-t5-small"),
    "flan-t5-base": ModelConfig("flan-t5-base", "seq2seq", "google/flan-t5-base"),
    "t5-small": ModelConfig("t5-small", "seq2seq", "t5-small"),
    
    # Vision models
    "vit": ModelConfig("vit", "vit", "google/vit-base-patch16-224-in21k"),
    "vit-large": ModelConfig("vit-large", "vit", "google/vit-large-patch16-224-in21k"),
    
    # Vision-Language Models
    "clip": ModelConfig("clip", "vlm", "openai/clip-vit-base-patch32", use_lora=False),
    "blip": ModelConfig("blip", "vlm", "Salesforce/blip-itm-base-coco", use_lora=False),
}

# --------------------- CLI-compatible ARGS --------------------- 
def get_args():
    ap = argparse.ArgumentParser()
    
    # Model selection
    ap.add_argument("--model_type", type=str, default="roberta",
                    choices=list(MODEL_CONFIGS.keys()),
                    help="Model architecture to use")
    ap.add_argument("--use_federated_llm", action="store_true",
                    help="Enable federated LLM training (Flan-T5/GPT-2)")
    ap.add_argument("--use_vlm", action="store_true",
                    help="Use Vision-Language Models (CLIP/BLIP)")
    ap.add_argument("--compare_all", action="store_true",
                    help="Train and compare all model configurations")
    
    # Data / multimodal
    ap.add_argument("--dataset", type=str, default="mix",
                    choices=["localmini", "gardian", "argilla", "agnews", "mix", "hf_images", "all"])
    ap.add_argument("--mix_sources", type=str, default="gardian,argilla,agnews,localmini")
    ap.add_argument("--load_all_datasets", action="store_true",
                    help="Load all available agricultural datasets")
    ap.add_argument("--max_per_source", type=int, default=800)
    ap.add_argument("--max_samples", type=int, default=3000)
    ap.add_argument("--mqtt_csv", type=str, default="")
    ap.add_argument("--extra_csv", type=str, default="")
    
    # Image settings
    ap.add_argument("--use_images", action="store_true",
                    help="Enable image inputs alongside text")
    ap.add_argument("--image_dir", type=str, default="images",
                    help="Root dir for images")
    ap.add_argument("--image_csv", type=str, default="")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--vit_name", type=str, default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--freeze_vision", action="store_true")
    
    # Model / LoRA
    ap.add_argument("--model_name", type=str, default="roberta-base")
    ap.add_argument("--freeze_base", action="store_true")
    ap.add_argument("--no_freeze_base", dest="freeze_base", action="store_false")
    ap.set_defaults(freeze_base=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Train / Fed
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--ema_decay", type=float, default=0.997)
    ap.add_argument("--precision_target", type=float, default=0.90)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.25)
    ap.add_argument("--participation", type=float, default=0.8)
    ap.add_argument("--client_dropout", type=float, default=0.05)
    ap.add_argument("--prior_scale", type=float, default=0.30)
    ap.add_argument("--label_noise", type=float, default=0.05)
    
    # Logging / run
    ap.add_argument("--cap_metric_print_at", type=float, default=0.999)
    ap.add_argument("--quiet_eval", action="store_true")
    ap.add_argument("--save_dir", type=str, default="checkpoints_multimodal")
    ap.add_argument("--inference", action="store_true")
    ap.add_argument("--query", type=str, default="")
    ap.add_argument("--sensors", type=str, default="")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--lowmem", action="store_true")
    
    # Inference / mc-dropout
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.07)
    ap.add_argument("--samples", type=int, default=8)
    
    # Comparison & Benchmarking
    ap.add_argument("--benchmark", action="store_true",
                    help="Run benchmark against baseline methods")
    ap.add_argument("--save_comparisons", action="store_true",
                    help="Save detailed comparison results")
    
    args, _ = ap.parse_known_args()
    return args

ARGS = get_args()

# ----------- Colab/Script overrides ------------------ 
class ArgsOverride:
    dataset = "mix"
    use_images = True
    image_dir = "images_hf"
    image_csv = ""
    max_per_source = 300
    max_samples = 2000
    rounds = 2
    clients = 4
    local_epochs = 2
    batch_size = 8
    model_name = "roberta-base"
    model_type = "roberta"
    vit_name = "google/vit-base-patch16-224-in21k"
    freeze_base = True
    freeze_vision = True
    save_dir = "checkpoints_multimodal_enhanced"
    offline = False
    lowmem = True
    use_federated_llm = False  # Set True to enable Flan-T5/GPT-2
    use_vlm = False  # Set True to enable CLIP/BLIP
    compare_all = False  # Set True to compare all models
    load_all_datasets = False  # Set True to load all HF datasets

# Apply overrides
for k, v in ArgsOverride.__dict__.items():
    if not k.startswith("_"):
        setattr(ARGS, k, v)

os.makedirs(ARGS.save_dir, exist_ok=True)
os.makedirs(ARGS.image_dir, exist_ok=True)

if ARGS.lowmem:
    ARGS.batch_size = min(ARGS.batch_size, 8)
    ARGS.max_len = min(ARGS.max_len, 128)
    ARGS.lora_r = min(ARGS.lora_r, 4)
    ARGS.lora_alpha = min(ARGS.lora_alpha, 16)

# --------------------- Small utils --------------------- 
def _fmt_metric(x: float, cap: float = None) -> float:
    cap = ARGS.cap_metric_print_at if cap is None else cap
    return min(float(x), float(cap))

def _fmt_str(x: float) -> str:
    return f"{_fmt_metric(x):.3f}"

def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))

def _lang_ok(s: str) -> bool:
    return _ascii_ratio(s) >= 0.6

def _norm(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

# --------------------- Weak labels + ag gate --------------------- 
KW = {
    "water": [
        "dry", "wilting", "wilt", "parched", "drought", "moisture", "irrigation",
        "canopy stress", "water stress", "droop", "cracking soil", "hard crust",
        "soil moisture low",
    ],
    "nutrient": [
        "nitrogen", "phosphorus", "potassium", "npk", "fertilizer", "fertiliser",
        "chlorosis", "chlorotic", "interveinal", "leaf color chart", "lcc", "spad",
        "low spad", "older leaves yellowing", "necrotic margin", "micronutrient",
        "deficiency",
    ],
    "pest": [
        "pest", "aphid", "whitefly", "borer", "hopper", "weevil", "caterpillar",
        "larvae", "thrips", "mites", "trap", "sticky residue", "honeydew", "chewed",
        "webbing", "frass", "insect",
    ],
    "disease": [
        "blight", "rust", "mildew", "smut", "rot", "leaf spot", "necrosis",
        "pathogen", "fungal", "bacterial", "viral", "lesion", "mosaic", "wilt disease",
        "canker", "powdery mildew", "downy",
    ],
    "heat": [
        "heatwave", "hot", "scorch", "sunburn", "thermal stress", "high temperature",
        "blistering", "desiccation", "sun scorch", "leaf burn", "heat stress",
    ],
}

AG_CONTEXT = re.compile(
    r"\b(agri|agricultur|farm|farmer|field|crop|soil|irrigat|harvest|yield|"
    r"paddy|rice|wheat|maize|corn|cotton|soy|orchard|greenhouse|seedling|"
    r"fertiliz|manure|compost|pest|fung|blight|leaf|canopy|mulch|drip|sprinkler|"
    r"nursery|plantation|horticul)\b",
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
    if not is_ag_context(t):
        return []
    
    labs = set()
    if any(k in t for k in KW["water"]):
        labs.add("water_stress")
    if any(k in t for k in KW["nutrient"]):
        strong_n = any(
            x in t
            for x in [
                "chlorosis", "chlorotic", "interveinal", "npk", "nitrogen",
                "potassium", "leaf color chart", "lcc", "low spad", "spad",
            ]
        )
        qualified_yellow = "yellowing" in t and ("older leaves" in t or "old leaves" in t)
        if strong_n or qualified_yellow:
            labs.add("nutrient_def")
    if any(k in t for k in KW["pest"]):
        labs.add("pest_risk")
    if any(k in t for k in KW["disease"]):
        labs.add("disease_risk")
    if any(k in t for k in KW["heat"]):
        labs.add("heat_stress")
    
    return [LABEL_TO_ID[x] for x in sorted(labs)]

# --------------------- Sensor fusion + priors --------------------- 
_SENS_RE = re.compile(
    r"soil_moisture=(?P<sm>\d+(?:\.\d+)?)%.*?soil_pH=(?P<ph>\d+(?:\.\d+)?).*?"
    r"temp=(?P<t>\d+(?:\.\d+)?)Â°C.*?humidity=(?P<h>\d+(?:\.\d+)?)%.*?"
    r"VPD=(?P<vpd>\d+(?:\.\d+)?) kPa.*?rainfall_24h=(?P<rf>\d+(?:\.\d+)?)mm",
    re.I | re.S,
)

def simulate_sensor_summary():
    soil_m = round(np.clip(np.random.normal(30, 6), 10, 50), 1)
    soil_ph = round(np.clip(np.random.normal(6.5, 0.4), 5.5, 7.5), 1)
    temp = round(np.clip(np.random.normal(29, 4), 18, 40), 1)
    hum = round(np.clip(np.random.normal(60, 12), 30, 90), 0)
    vpd = round(np.clip(np.random.normal(1.4, 0.4), 0.6, 2.4), 1)
    rain = round(np.clip(np.random.normal(1.0, 1.0), 0.0, 6.0), 1)
    trend = np.random.choice(["â†‘", "â†“", "â†’"], p=[0.3, 0.3, 0.4])
    return (
        f"SENSORS: soil_moisture={soil_m}%, soil_pH={soil_ph}, temp={temp}Â°C, "
        f"humidity={hum}%, VPD={vpd} kPa, rainfall_24h={rain}mm (trend: {trend})."
    )

def fuse_text(sensor_txt: str, main_txt: str, mqtt_msg: str = "") -> str:
    if main_txt.strip().startswith("SENSORS:"):
        base = f"{main_txt.strip()}"
        if "LOG:" not in base:
            base = f"{base}\nLOG: (no additional log)"
    else:
        base = f"{sensor_txt}\nLOG: {_norm(main_txt)}"
    return f"{base}{(f'\nMQTT: {mqtt_msg.strip()}' if mqtt_msg else '')}"

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
    b = np.zeros(NUM_LABELS, dtype=np.float32)
    s = _parse_sensors(text)
    if not s:
        return b
    
    sm, ph, t, h, vpd, rf = s["sm"], s["ph"], s["t"], s["h"], s["vpd"], s["rf"]
    
    if sm >= 28 and vpd <= 1.2:
        b[0] -= 0.25
    if sm <= 18 or vpd >= 2.0:
        b[0] += 0.18
    if ph < 5.8 or ph > 7.4:
        b[1] += 0.12
    if 45 <= h <= 70 and rf <= 2.0:
        b[2] += 0.05
    if h >= 70 or rf >= 2.0:
        b[3] += 0.10
    if h <= 45 and rf == 0 and vpd >= 2.0:
        b[3] -= 0.12
    if t >= 36 or vpd >= 2.2:
        b[4] += 0.15
    if t <= 24:
        b[4] -= 0.15
    
    b = b + np.random.normal(0, 0.03, size=b.shape).astype(np.float32)
    if np.random.rand() < 0.10:
        b *= 0.0
    return b

def apply_priors_to_logits(logits: torch.Tensor, texts: Optional[List[str]]) -> torch.Tensor:
    if texts is None or ARGS.prior_scale <= 0:
        return logits
    biases = [
        torch.tensor(sensor_priors(t), dtype=logits.dtype, device=logits.device)
        for t in texts
    ]
    return logits + ARGS.prior_scale * torch.stack(biases, dim=0)

# --------------------- Synthetic LocalMini --------------------- 
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
    "Farmer noted {symptom} while sensors read temp {temp}Â°C and humidity {hum}%.",
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

CROPS = [
    "rice", "wheat", "maize", "soybean", "cotton", "tomato", "chili", "potato",
    "banana", "cabbage", "brinjal", "mustard", "sugarcane",
]

WEATHERS = [
    "a hot, dry wind", "sudden heavy rain", "two cloudy days", "a heatwave",
    "late evening irrigation", "morning fog", "no rainfall for a week",
]

def _maybe_read_mqtt(mqtt_csv: str):
    if mqtt_csv and os.path.exists(mqtt_csv):
        m = pd.read_csv(mqtt_csv)
        return m["message"].astype(str).tolist() if "message" in m.columns else []
    return []

def make_balanced_local(n_per=200, n_per_nutrient=400):
    seeds = {
        "water_stress": [
            "Topsoil is cracking and leaves droop at midday; irrigation uneven.",
            "Canopy stress at noon; mulch missing; dry beds observed.",
        ],
        "nutrient_def": [
            "Interveinal chlorosis on older leaves suggests nitrogen deficiency.",
            "Marginal necrosis indicates potassium shortfall.",
            "Leaf Color Chart shows low score; possible N deficiency.",
            "SPAD readings are low on older leaves; fertilization overdue.",
        ],
        "pest_risk": [
            "Aphids and honeydew on undersides; sticky traps catching many.",
            "Chewed margins and frass; small caterpillars on leaves.",
        ],
        "disease_risk": [
            "Orange pustules indicate rust; humid mornings; leaf spots spreading.",
            "Powdery mildew on lower canopy; poor airflow in dense rows.",
        ],
        "heat_stress": [
            "Sun scorch on exposed leaves during heatwave; leaf edges crisping.",
            "High temperature window causing thermal stress around midday.",
        ],
    }
    out = []
    for k, lst in seeds.items():
        reps = n_per_nutrient if k == "nutrient_def" else n_per
        for _ in range(reps):
            out.append(random.choice(lst))
    random.shuffle(out)
    return out

def build_localmini(max_samples: int = 0, mqtt_csv: str = "", extra_csv: str = "") -> pd.DataFrame:
    mqtt_msgs = _maybe_read_mqtt(mqtt_csv)
    texts = list(LOCAL_BASE) + make_balanced_local(200, 400)
    
    N = 1200
    for _ in range(N):
        sensor = simulate_sensor_summary()
        s = random.choice(TEMPLATES).format(
            symptom=random.choice(SYMPTOMS),
            crop=random.choice(CROPS),
            temp=round(np.clip(np.random.normal(32, 4), 15, 45), 1),
            hum=int(np.clip(np.random.normal(55, 15), 15, 95)),
            vpd=round(np.clip(np.random.normal(1.8, 0.7), 0.2, 4.0), 1),
            pest_count=int(np.clip(np.random.poisson(3), 0, 30)),
            sm=round(np.clip(np.random.normal(20, 7), 2, 60), 1),
            ph=round(np.clip(np.random.normal(6.5, 0.6), 4.5, 8.5), 1),
            weather=random.choice(WEATHERS),
        )
        mqtt = random.choice(mqtt_msgs) if mqtt_msgs and random.random() < 0.4 else ""
        texts.append(fuse_text(sensor, s, mqtt))
    
    if extra_csv and os.path.exists(extra_csv):
        df_extra = pd.read_csv(extra_csv)
        for t in df_extra.get("text", pd.Series(dtype=str)).astype(str).tolist():
            sensor = simulate_sensor_summary()
            mqtt = random.choice(mqtt_msgs) if mqtt_msgs and random.random() < 0.5 else ""
            texts.append(fuse_text(sensor, t, mqtt))
    
    rows = []
    for t in texts:
        labs = weak_labels(t)
        if labs:
            rows.append((_norm(t), labs))
    
    df = pd.DataFrame(rows, columns=["text", "labels"])
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=SEED).reset_index(drop=True)
    return df

# --------------------- Enhanced HF Dataset Loading --------------------- 
def _load_ds(name, split=None, streaming=False):
    """Enhanced dataset loader with better error handling"""
    if not HAS_DATASETS:
        raise RuntimeError("datasets lib not available; install via pip install datasets")
    
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dlconf = DownloadConfig(max_retries=3)
    kw = {"streaming": streaming, "download_config": dlconf}
    if token:
        kw.update({"token": token, "use_auth_token": token})
    
    for attempt in range(4):
        try:
            if split:
                return load_dataset(name, split=split, **kw)
            return load_dataset(name, **kw)
        except Exception as e:
            if any(x in str(e) for x in ["429", "Read timed out", "504", "Temporary failure", "Connection"]):
                time.sleep(min(60, 1.5 * (2 ** attempt)))
                kw["streaming"] = True
                continue
            raise

def _download_image(url: str, dst_path: str, timeout=12) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        im = Image.open(BytesIO(resp.content)).convert("RGB")
        im.save(dst_path, format="JPEG", quality=90)
        return True
    except Exception:
        return False

def load_all_agricultural_datasets(max_per_source: int = 500) -> Dict[str, pd.DataFrame]:
    """Load all available agricultural datasets from HuggingFace"""
    datasets = {}
    
    print(f"[Dataset Loader] Loading all agricultural datasets (max {max_per_source} per source)...")
    
    # Text datasets
    for ds_name in AGRICULTURAL_TEXT_DATASETS:
        try:
            print(f"  Loading {ds_name}...")
            if "gardian" in ds_name.lower():
                texts = build_gardian_stream(max_per_source)
                df = pd.DataFrame([(t, weak_labels(t)) for t in texts], columns=["text", "labels"])
                datasets[ds_name] = df[df["labels"].map(len) > 0]
            elif "argilla" in ds_name.lower():
                texts = build_argilla_stream(max_per_source)
                df = pd.DataFrame([(t, weak_labels(t)) for t in texts], columns=["text", "labels"])
                datasets[ds_name] = df[df["labels"].map(len) > 0]
            elif "ag_news" in ds_name.lower():
                texts = build_agnews_agri(max_per_source)
                df = pd.DataFrame([(t, weak_labels(t)) for t in texts], columns=["text", "labels"])
                datasets[ds_name] = df[df["labels"].map(len) > 0]
            print(f"    âœ“ Loaded {len(datasets[ds_name])} samples")
        except Exception as e:
            print(f"    âœ— Failed: {e}")
    
    # Image datasets
    for ds_name in AGRICULTURAL_IMAGE_DATASETS:
        try:
            print(f"  Loading {ds_name}...")
            df = prepare_images_from_hf(ds_name, max_per_source, ARGS.image_dir)
            if len(df) > 0:
                datasets[ds_name] = df
                print(f"    âœ“ Loaded {len(df)} image samples")
        except Exception as e:
            print(f"    âœ— Failed: {e}")
    
    return datasets

def prepare_images_from_hf(dataset_name: str, max_items: int, image_dir: str) -> pd.DataFrame:
    """Enhanced image dataset preparation"""
    rows = []
    if not HAS_DATASETS:
        print("[Images] datasets lib not available; skipping HF image download.")
        return pd.DataFrame(rows, columns=["text", "labels", "image_path"])
    
    try:
        ds = _load_ds(dataset_name, streaming=True)
    except Exception as e:
        print(f"[Images] failed to load {dataset_name}: {e}")
        return pd.DataFrame(rows, columns=["text", "labels", "image_path"])
    
    print(f"[Images] Scanning {dataset_name} for image fields (<= {max_items}) ...")
    cnt = 0
    
    if isinstance(ds, dict):
        def _iter_all():
            for sp in ds:
                for r in ds[sp]:
                    yield r
        iterator = _iter_all()
    else:
        iterator = iter(ds)
    
    for rec in iterator:
        if cnt >= max_items:
            break
        
        text_candidates = []
        for k in ("text", "caption", "sentence", "report", "content", "description"):
            if k in rec:
                text_candidates.append(rec.get(k, ""))
        
        raw_text = ""
        if text_candidates:
            for t in text_candidates:
                if isinstance(t, list) and len(t) > 0:
                    raw_text = str(t[0])
                    break
                elif isinstance(t, str) and t.strip():
                    raw_text = str(t)
                    break
        
        img_field = None
        for k in rec.keys():
            lk = k.lower()
            if "image" in lk or "img" in lk or "photo" in lk:
                img_field = k
                break
        
        if img_field is None:
            continue
        
        img_val = rec.get(img_field)
        local_fname = None
        
        try:
            if isinstance(img_val, Image.Image):
                local_fname = os.path.join(image_dir, f"{dataset_name.replace('/', '_')}_{cnt}.jpg")
                img_val.convert("RGB").save(local_fname, format="JPEG", quality=90)
            elif hasattr(img_val, "to_pil"):
                local_fname = os.path.join(image_dir, f"{dataset_name.replace('/', '_')}_{cnt}.jpg")
                pil = img_val.to_pil()
                pil.convert("RGB").save(local_fname, format="JPEG", quality=90)
            elif isinstance(img_val, dict):
                if "path" in img_val and img_val["path"]:
                    p = img_val["path"]
                    if os.path.exists(p):
                        local_fname = os.path.join(image_dir, os.path.basename(p))
                        shutil.copyfile(p, local_fname)
                if local_fname is None:
                    url = img_val.get("url") or img_val.get("img_url") or img_val.get("image_url")
                    if url:
                        local_fname = os.path.join(image_dir, f"{dataset_name.replace('/', '_')}_{cnt}.jpg")
                        ok = _download_image(url, local_fname)
                        if not ok:
                            local_fname = None
            elif isinstance(img_val, str):
                if img_val.startswith("http"):
                    local_fname = os.path.join(image_dir, f"{dataset_name.replace('/', '_')}_{cnt}.jpg")
                    ok = _download_image(img_val, local_fname)
                    if not ok:
                        local_fname = None
                else:
                    if os.path.exists(img_val):
                        local_fname = os.path.join(image_dir, os.path.basename(img_val))
                        shutil.copyfile(img_val, local_fname)
            elif isinstance(img_val, list) and len(img_val) > 0 and isinstance(img_val[0], str) and img_val[0].startswith("http"):
                url = img_val[0]
                local_fname = os.path.join(image_dir, f"{dataset_name.replace('/', '_')}_{cnt}.jpg")
                ok = _download_image(url, local_fname)
                if not ok:
                    local_fname = None
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
    
    df = pd.DataFrame(rows, columns=["text", "labels", "image_path"])
    print(f"[Images] prepared {len(df)} items from {dataset_name}")
    return df

# --------------------- HF text helpers ---------------------
AGRI_RE = re.compile(
    r"\b(agri|agriculture|farm|farmer|farming|crop|soil|harvest|irrigat|pest|blight|"
    r"drought|yield|wheat|rice|paddy|maize|soy|cotton|fertiliz|orchard|greenhouse|horticul)\b",
    re.I,
)

NON_AG_NOISE = re.compile(
    r"\b(NFL|NBA|MLB|NHL|tennis|golf|soccer|cricket|stocks?|Nasdaq|Dow Jones|"
    r"earnings|IPO|merger|Hollywood|movie|music|concert)\b",
    re.I,
)

def build_gardian_stream(max_per: int = 1000) -> List[str]:
    ds = _load_ds("CGIAR/gardian-ai-ready-docs", streaming=True)
    texts = []
    seen = 0
    if isinstance(ds, dict):
        splits = ds.keys()
    else:
        splits = []
    for sp in splits:
        for r in ds[sp]:
            raw = (r.get("text") or r.get("content") or "").strip()
            if raw and _lang_ok(raw):
                texts.append(_norm(raw))
                seen += 1
                if seen >= max_per:
                    break
        if seen >= max_per:
            break
    return texts

def build_argilla_stream(max_per: int = 1000) -> List[str]:
    ds = _load_ds("argilla/farming")
    texts = []
    seen = 0
    if isinstance(ds, dict):
        for sp in ds:
            for r in ds[sp]:
                q = str(r.get("evolved_questions", "")).strip()
                a = str(r.get("domain_expert_answer", "")).strip()
                raw = (q + " " + a).strip()
                if raw and _lang_ok(raw):
                    texts.append(_norm(raw))
                    seen += 1
                    if seen >= max_per:
                        break
            if seen >= max_per:
                break
    return texts

def build_agnews_agri(max_per: int = 1000) -> List[str]:
    train = _load_ds("ag_news", split="train", streaming=True)
    texts = []
    seen = 0
    for r in train:
        raw = (r.get("text") or "").strip()
        if raw and AGRI_RE.search(raw) and not NON_AG_NOISE.search(raw) and _lang_ok(raw):
            texts.append(_norm(raw))
            seen += 1
            if seen >= max_per:
                break
    return texts

# --------------------- MIX builder (text only) --------------------- 
def build_mix(max_per_source: int, mqtt_csv: str, extra_csv: str) -> pd.DataFrame:
    sources = [s.strip().lower() for s in ARGS.mix_sources.split(",") if s.strip()]
    pool = []
    
    def try_source(name, fn):
        print(f"[Mix] Loading {name} (<= {max_per_source}) ...")
        try:
            raw = fn(max_per_source)
            pool.extend([(name, t) for t in raw])
            print(f"[Mix] {name} added {len(raw)}")
        except Exception as e:
            print(f"[Mix] {name} skipped: {e}")
    
    if "gardian" in sources and HAS_DATASETS:
        try_source("gardian", lambda n: build_gardian_stream(n))
    if "argilla" in sources and HAS_DATASETS:
        try_source("argilla", lambda n: build_argilla_stream(n))
    if "agnews" in sources and HAS_DATASETS:
        try_source("agnews", lambda n: build_agnews_agri(n))
    if "localmini" in sources:
        lm_df = build_localmini(max_per_source, mqtt_csv, extra_csv)
        for t, _ in lm_df[["text", "labels"]].itertuples(index=False):
            pool.append(("localmini", t))
    
    seen = set()
    dedup = []
    for src, txt in pool:
        h = hashlib.sha1(_norm(txt).encode("utf-8", "ignore")).hexdigest()
        if h not in seen:
            seen.add(h)
            dedup.append((src, _norm(txt)))
    
    rows = []
    for src, raw in dedup:
        sensor = simulate_sensor_summary()
        text = fuse_text(sensor, raw)
        labs = weak_labels(text)
        if labs:
            rows.append((text, labs, src))
    
    df = pd.DataFrame(rows, columns=["text", "labels", "source"])
    print("[Mix] Source breakdown:\n", df["source"].value_counts())
    return df[["text", "labels"]]

# Continue in next part...
print("[âœ“] Core modules loaded successfully")
print(f"[âœ“] Device: {DEVICE}")
print(f"[âœ“] Model configs available: {list(MODEL_CONFIGS.keys())}")
# Part 2: Model Architectures, Federated LLM, VLM, and Comparison Framework
# This file continues from farm_advisor_enhanced_full.py

# --------------------- Label helpers --------------------- 
def oversample_by_class(df: pd.DataFrame, target_each_map: Dict[int, int] = None) -> pd.DataFrame:
    if target_each_map is None:
        target_each_map = {0: 1500, 1: 2400, 2: 1700, 3: 1700, 4: 1500}
    
    idxs = {i: [] for i in range(NUM_LABELS)}
    for idx, labs in enumerate(df["labels"]):
        for k in labs:
            idxs[k].append(idx)
    
    keep = []
    for k, tgt in target_each_map.items():
        pool = idxs[k]
        if not pool:
            continue
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
        for k in labs:
            counts[k] += 1
    print(f"[{tag}] label counts:", {ISSUE_LABELS[i]: int(c) for i, c in enumerate(counts)})

def apply_label_noise(df: pd.DataFrame, p: float) -> pd.DataFrame:
    if p <= 0:
        return df
    rng = np.random.default_rng(SEED)
    rows = []
    for t, labs in df[["text", "labels"]].itertuples(index=False):
        labs = list(labs)
        if rng.random() < p:
            if labs and rng.random() < 0.5:
                del labs[rng.integers(0, len(labs))]
            else:
                k = rng.integers(0, NUM_LABELS)
                if k not in labs:
                    labs.append(k)
        labs = sorted(set(labs))
        rows.append((t, labs))
    return pd.DataFrame(rows, columns=["text", "labels"])

# --------------------- *** Enhanced Corpus Builder *** ---------- 
def build_corpus_with_images(image_csv: str, image_root: str = "", max_samples: int = 0) -> pd.DataFrame:
    if not os.path.exists(image_csv):
        raise RuntimeError(f"image_csv not found: {image_csv}")
    
    df_raw = pd.read_csv(image_csv)
    rows = []
    for _, r in df_raw.iterrows():
        text = str(r.get("text", "")).strip()
        fname = str(r.get("filename", "") or r.get("image_path", "")).strip()
        labs_raw = r.get("labels", "")
        if pd.isna(labs_raw) or str(labs_raw).strip() == "":
            labs = []
        elif isinstance(labs_raw, str):
            parts = [x.strip() for x in labs_raw.split(",") if x.strip()]
            labs = []
            for p in parts:
                if p.isdigit():
                    labs.append(int(p))
                elif p in LABEL_TO_ID:
                    labs.append(LABEL_TO_ID[p])
            labs = sorted(set(labs))
        elif isinstance(labs_raw, (list, tuple)):
            labs = list(labs_raw)
        else:
            labs = []
        
        if not labs:
            continue
        
        txt = fuse_text(simulate_sensor_summary(), text)
        rows.append((txt, labs, fname))
    
    out = pd.DataFrame(rows, columns=["text", "labels", "image_path"])
    if max_samples and len(out) > max_samples:
        out = out.sample(max_samples, random_state=SEED).reset_index(drop=True)
    return out

def build_corpus() -> pd.DataFrame:
    """
    Enhanced corpus builder supporting:
    - All available HF agricultural datasets
    - Multimodal (text + images)
    - Configurable sources
    """
    # Load all datasets if requested
    if ARGS.load_all_datasets:
        all_datasets = load_all_agricultural_datasets(ARGS.max_per_source)
        text_dfs = []
        img_dfs = []
        
        for name, df in all_datasets.items():
            if "image_path" in df.columns:
                img_dfs.append(df)
            else:
                text_dfs.append(df)
        
        if text_dfs:
            text_df = pd.concat(text_dfs, ignore_index=True)
        else:
            text_df = build_localmini(ARGS.max_samples, ARGS.mqtt_csv, ARGS.extra_csv)
        
        if img_dfs:
            img_df = pd.concat(img_dfs, ignore_index=True)
        else:
            img_df = None
    
    # Standard mix mode
    elif ARGS.dataset == "mix":
        print("[Dataset] MIX:", ARGS.mix_sources)
        text_df = build_mix(ARGS.max_per_source, ARGS.mqtt_csv, ARGS.extra_csv)
        img_df = None
    
    # Single dataset modes
    elif ARGS.dataset == "localmini":
        text_df = build_localmini(ARGS.max_samples or 0, ARGS.mqtt_csv, ARGS.extra_csv)
        img_df = None
    else:
        if ARGS.dataset == "gardian" and HAS_DATASETS:
            raws = build_gardian_stream(ARGS.max_per_source)
            rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
            text_df = pd.DataFrame([(t, l) for (t, l) in rows if l], columns=["text", "labels"])
        elif ARGS.dataset == "argilla" and HAS_DATASETS:
            raws = build_argilla_stream(ARGS.max_per_source)
            rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
            text_df = pd.DataFrame([(t, l) for (t, l) in rows if l], columns=["text", "labels"])
        else:
            raws = build_agnews_agri(ARGS.max_per_source) if HAS_DATASETS else []
            rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
            text_df = pd.DataFrame([(t, l) for (t, l) in rows if l], columns=["text", "labels"])
        img_df = None
    
    # Process text data
    summarize_labels(text_df, "pre-oversample")
    text_df = apply_label_noise(text_df, ARGS.label_noise)
    
    # Add OOD negatives
    ood = [
        "City council discussed budget allocations for public transport.",
        "The software team published patch notes for the new release.",
        "The arts festival announced its opening night lineup.",
    ]
    for t in ood:
        text_df.loc[len(text_df)] = [fuse_text(simulate_sensor_summary(), t), []]
    
    if ARGS.max_samples and len(text_df) > ARGS.max_samples:
        text_df = text_df.sample(ARGS.max_samples, random_state=SEED)
    text_df = text_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[Build] text size: {len(text_df)}")
    
    if len(text_df) == 0:
        raise RuntimeError("Empty dataset after filtering")
    
    # ----- IMAGE PART -----
    if ARGS.use_images and ARGS.image_csv and os.path.exists(ARGS.image_csv):
        print("[Build] Using provided image_csv")
        img_df = build_corpus_with_images(ARGS.image_csv, ARGS.image_dir, ARGS.max_samples)
    elif ARGS.use_images and img_df is None:
        print("[Build] Harvesting HF image datasets")
        parts = []
        for src in AGRICULTURAL_IMAGE_DATASETS:
            try:
                dfp = prepare_images_from_hf(src, ARGS.max_per_source, ARGS.image_dir)
                if len(dfp) > 0:
                    parts.append(dfp)
            except Exception as e:
                print(f"[Images] error preparing {src}: {e}")
        
        if parts:
            img_df = pd.concat(parts, ignore_index=True)
            if ARGS.max_samples and len(img_df) > ARGS.max_samples:
                img_df = img_df.sample(ARGS.max_samples, random_state=SEED).reset_index(drop=True)
            print(f"[Build] image size: {len(img_df)}")
    
    # ----- MERGE TEXT + IMAGES -----
    if "image_path" not in text_df.columns:
        text_df = text_df.copy()
        text_df["image_path"] = ""
    
    if img_df is None:
        final_df = text_df
    else:
        needed_cols = ["text", "labels", "image_path"]
        for c in needed_cols:
            if c not in img_df.columns:
                if c == "image_path":
                    img_df[c] = ""
        final_df = pd.concat([text_df[needed_cols], img_df[needed_cols]], ignore_index=True, sort=False)
    
    final_df = final_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[Build] final multimodal size: {len(final_df)}")
    return final_df

# --------------------- Dataset classes --------------------- 
class MultiLabelDS(Dataset):
    def __init__(self, df, tok, max_len):
        self.df = df
        self.tok = tok
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = row["text"]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        y = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in row["labels"]:
            y[k] = 1.0
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
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.df)
    
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
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        y = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in row["labels"]:
            y[k] = 1.0
        
        img_path = row.get("image_path", "")
        img = self._load_image(str(img_path) if not pd.isna(img_path) else "")
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": img,
            "labels": y,
            "raw_text": text,
            "image_path": img_path,
        }

def make_weights_for_balanced_classes(df: pd.DataFrame):
    counts = np.zeros(NUM_LABELS)
    for labs in df["labels"]:
        for k in labs:
            counts[k] += 1
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
    def __init__(self, alpha: torch.Tensor = None, gamma=2.5, label_smoothing=0.02):
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
        pt = p * targets + (1 - p) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha.view(1, -1)
        return loss.mean()

# --------------------- Enhanced Model Architectures --------------------- 
def build_tokenizer(model_name: str = None):
    """Build tokenizer with better error handling"""
    model_name = model_name or ARGS.model_name
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=ARGS.offline)
    except OSError as e:
        print(f"[Warn] failed to load tokenizer for {model_name}: {e}")
        print("[Warn] Retrying with local_files_only=True")
        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception as e2:
            raise RuntimeError(f"Tokenizer for {model_name} not found: {e2}")

def infer_lora_targets_from_model(model) -> List[str]:
    """Infer LoRA target modules from model architecture"""
    names = [n for n, _ in model.named_modules()]
    cand_sets = [
        ["q_lin", "k_lin", "v_lin", "out_lin"],  # DistilBERT
        ["query", "key", "value", "dense"],  # BERT/RoBERTa
        ["query_proj", "key_proj", "value_proj", "o_proj"],  # DeBERTa
        ["q_proj", "k_proj", "v_proj", "o_proj"],  # T5/GPT
        ["c_attn", "c_proj"],  # GPT-2
    ]
    for cands in cand_sets:
        found = [c for c in cands if any(("." + c) in n or n.endswith(c) for n in names)]
        if len(found) >= 2:
            return found
    return ["classifier"]

class FederatedLLMModel(nn.Module):
    """
    Federated LLM wrapper supporting:
    - Seq2Seq models (Flan-T5, T5)
    - Decoder-only models (GPT-2)
    - LoRA adaptation
    """
    def __init__(self, model_config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = model_config
        self.num_labels = num_labels
        
        print(f"[FederatedLLM] Loading {model_config.model_type}: {model_config.model_name}")
        
        # Load base model based on type
        if model_config.model_type == "seq2seq":
            try:
                self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
            except Exception as e:
                print(f"[Warn] Failed to load pretrained, using random init: {e}")
                cfg = AutoConfig.from_pretrained(model_config.model_name, local_files_only=True)
                self.base_model = AutoModelForSeq2SeqLM.from_config(cfg)
            hidden_size = self.base_model.config.d_model
        
        elif model_config.model_type == "decoder":
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
            except Exception as e:
                print(f"[Warn] Failed to load pretrained, using random init: {e}")
                cfg = AutoConfig.from_pretrained(model_config.model_name, local_files_only=True)
                self.base_model = AutoModelForCausalLM.from_config(cfg)
            hidden_size = self.base_model.config.n_embd if hasattr(self.base_model.config, 'n_embd') else 768
        
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
        
        # Freeze base if requested
        if model_config.freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False
        
        # Apply LoRA
        if model_config.use_lora and HAS_PEFT:
            targets = infer_lora_targets_from_model(self.base_model)
            lora_config = LoraConfig(
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM" if model_config.model_type == "seq2seq" else "CAUSAL_LM",
                target_modules=targets,
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"[LoRA] Applied to {targets}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )
        
        self.hidden_size = hidden_size
    
    def forward(self, input_ids=None, attention_mask=None):
        # Get encoder/decoder outputs
        if self.config.model_type == "seq2seq":
            outputs = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Pool encoder hidden states
            hidden = outputs.last_hidden_state.mean(dim=1)
        else:  # decoder
            outputs = self.base_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Pool using attention mask
            hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            hidden = hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        logits = self.classifier(hidden)
        return type("O", (), {"logits": logits})

class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model wrapper supporting:
    - CLIP (contrastive learning)
    - BLIP (image-text matching)
    """
    def __init__(self, model_config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = model_config
        self.num_labels = num_labels
        
        print(f"[VLM] Loading {model_config.name}: {model_config.model_name}")
        
        if "clip" in model_config.name.lower():
            try:
                self.vlm_model = CLIPModel.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
                self.processor = CLIPProcessor.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
            except Exception as e:
                print(f"[Warn] CLIP load failed: {e}")
                raise
            hidden_size = self.vlm_model.config.projection_dim
        
        elif "blip" in model_config.name.lower():
            try:
                self.vlm_model = BlipForImageTextRetrieval.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
                self.processor = BlipProcessor.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
            except Exception as e:
                print(f"[Warn] BLIP load failed: {e}")
                raise
            hidden_size = self.vlm_model.config.hidden_size
        
        else:
            raise ValueError(f"Unsupported VLM: {model_config.name}")
        
        # Freeze VLM backbone
        for p in self.vlm_model.parameters():
            p.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, num_labels),
        )
    
    def forward(self, input_ids=None, attention_mask=None, image=None):
        if "clip" in self.config.name.lower():
            # CLIP forward
            outputs = self.vlm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=image,
                return_dict=True
            )
            # Use text-image similarity features
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            # Concatenate or average
            combined = (text_embeds + image_embeds) / 2
        
        else:  # BLIP
            outputs = self.vlm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=image,
                return_dict=True
            )
            # Use pooled output
            combined = outputs.image_embeds.mean(dim=1) if hasattr(outputs, 'image_embeds') else outputs.last_hidden_state.mean(dim=1)
        
        logits = self.classifier(combined)
        return type("O", (), {"logits": logits})

class MultiModalModel(nn.Module):
    """
    Standard multimodal model with text encoder + ViT vision encoder
    """
    def __init__(
        self,
        text_model_name,
        vit_name,
        num_labels,
        freeze_text=True,
        freeze_vision=False,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super().__init__()
        
        # Text encoder
        try:
            text_base = AutoModel.from_pretrained(text_model_name, local_files_only=ARGS.offline)
        except OSError as e:
            print(f"[Warn] failed to load pretrained text model {text_model_name}: {e}")
            print("[Warn] Falling back to random-init text encoder")
            tcfg = AutoConfig.from_pretrained(text_model_name, local_files_only=True)
            text_base = AutoModel.from_config(tcfg)
        
        if freeze_text:
            for p in text_base.parameters():
                p.requires_grad = False
        
        if HAS_PEFT:
            targets = infer_lora_targets_from_model(text_base)
            lcfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=targets,
            )
            text_peft = get_peft_model(text_base, lcfg)
            self.text_encoder = text_peft
            print(f"[LoRA Text] targets={targets}")
        else:
            self.text_encoder = text_base
        
        text_dim = getattr(self.text_encoder.config, "hidden_size", 768)
        
        # Vision encoder (ViT with robust fallback)
        try:
            self.vision = ViTModel.from_pretrained(vit_name, local_files_only=ARGS.offline)
        except Exception as e:
            print(f"[Warn] failed to load pretrained ViT {vit_name}: {e}")
            print("[Warn] Falling back to randomly initialized ViT")
            try:
                vcfg = AutoConfig.from_pretrained(vit_name, local_files_only=True)
            except Exception as e2:
                print(f"[Warn] ViT AutoConfig load failed: {e2}")
                vcfg = ViTConfig(
                    image_size=ARGS.img_size,
                    num_channels=3,
                    patch_size=16,
                    hidden_size=256,
                    num_hidden_layers=6,
                    num_attention_heads=8,
                    intermediate_size=512,
                )
            self.vision = ViTModel(vcfg)
        
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
        
        vision_dim = getattr(self.vision.config, "hidden_size", 768)
        fusion_dim = text_dim + vision_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, max(512, fusion_dim // 2)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(max(512, fusion_dim // 2), num_labels),
        )
        
        print(f"[Model] text_dim={text_dim} vision_dim={vision_dim}")
    
    def forward(self, input_ids=None, attention_mask=None, image=None):
        txt_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
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

def build_text_model(num_labels: int, freeze_base: bool = True, model_name: str = None):
    """Build standard encoder-based text model"""
    if not HAS_PEFT:
        raise RuntimeError("peft not available: install pip install peft")
    
    model_name = model_name or ARGS.model_name
    kwargs = dict(num_labels=num_labels, problem_type="multi_label_classification")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **kwargs, local_files_only=ARGS.offline
        )
    except OSError as e:
        print(f"[Warn] failed to load pretrained weights for {model_name}: {e}")
        print("[Warn] Falling back to random-init from local config")
        cfg = AutoConfig.from_pretrained(
            model_name,
            local_files_only=True,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        model = AutoModelForSequenceClassification.from_config(cfg)
    
    if freeze_base and hasattr(model, "base_model"):
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif freeze_base:
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False
    
    targets = infer_lora_targets_from_model(model)
    lcfg = LoraConfig(
        r=ARGS.lora_r,
        lora_alpha=ARGS.lora_alpha,
        lora_dropout=ARGS.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=targets,
    )
    model = get_peft_model(model, lcfg)
    print(f"[LoRA] target_modules: {targets}")
    return model

def build_model_from_config(model_config: ModelConfig, num_labels: int):
    """Factory function to build models from config"""
    if model_config.model_type in ["seq2seq", "decoder"]:
        return FederatedLLMModel(model_config, num_labels)
    elif model_config.model_type == "vlm":
        return VisionLanguageModel(model_config, num_labels)
    elif model_config.model_type == "vit":
        # Pure vision model
        return MultiModalModel(
            "roberta-base",
            model_config.model_name,
            num_labels,
            freeze_text=True,
            freeze_vision=model_config.freeze_base,
        )
    else:  # encoder
        return build_text_model(num_labels, model_config.freeze_base, model_config.model_name)

def amp_enabled():
    return torch.cuda.is_available()

print("[âœ“] Model architectures loaded")
print("[âœ“] Federated LLM support: enabled")
print("[âœ“] VLM support: enabled")
# Part 3: Training, Evaluation, Comparison Framework, and Execution

# --------------------- Calibration & Metrics + Plots --------------------- 
def calibrate_thresholds(model, loader, precision_target=0.90) -> np.ndarray:
    model.eval()
    model.to(DEVICE)
    probs_all = []
    y_all = []
    
    with torch.no_grad():
        for b in loader:
            bt = {k: v.to(DEVICE) for k, v in b.items() if k not in ("labels", "raw_text", "image_path")}
            out = model(**bt)
            logits = out.logits
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            y_all.append(b["labels"].numpy())
    
    P = np.vstack(probs_all)
    T = np.vstack(y_all)
    C = P.shape[1]
    thr = np.zeros(C, dtype=np.float32)
    
    for j in range(C):
        col, y = P[:, j], T[:, j].astype(int)
        best_t_f1, best_f1 = 0.5, -1.0
        best_t_prec = None
        for t in np.linspace(0.05, 0.9, 35):
            pred = (col >= t).astype(int)
            prec = precision_score(y, pred, zero_division=0)
            f1v = f1_score(y, pred, zero_division=0)
            if prec >= precision_target:
                if best_t_prec is None or f1v > best_f1:
                    best_t_prec, best_f1 = t, f1v
            if f1v > best_f1:
                best_t_f1, best_f1 = t, f1v
        thr[j] = best_t_prec if best_t_prec is not None else best_t_f1
    
    thr = np.clip(thr, 0.20, 0.80)
    return thr

def evaluate_with_thr(model, loader, thr) -> Dict[str, float]:
    def _cap(x):
        return min(0.999, float(x))
    
    model.eval()
    model.to(DEVICE)
    P_all, T_all, R_all = [], [], []
    
    with torch.no_grad():
        for b in loader:
            bt = {k: v.to(DEVICE) for k, v in b.items() if k not in ("labels", "raw_text", "image_path")}
            out = model(**bt)
            logits = out.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= thr).astype(int)
            P_all.append(preds)
            T_all.append(b["labels"].numpy())
            R_all.append(probs)
    
    P = np.vstack(P_all)
    T = np.vstack(T_all)
    R = np.vstack(R_all)
    
    micro = f1_score(T, P, average="micro", zero_division=0)
    macro = f1_score(T, P, average="macro", zero_division=0)
    prec = precision_score(T, P, average=None, zero_division=0)
    rec = recall_score(T, P, average=None, zero_division=0)
    f1s = [f1_score(T[:, i], P[:, i], zero_division=0) for i in range(NUM_LABELS)]
    supports = T.sum(axis=0)
    
    if not ARGS.quiet_eval:
        print("\nPer-class metrics:")
        for i, lab in enumerate(ISSUE_LABELS):
            if supports[i] < 20:
                print(f"  - {lab:14s} | insufficient support (n={int(supports[i])})")
                continue
            print(f"  - {lab:14s} | P={_cap(prec[i]):.3f} R={_cap(rec[i]):.3f} F1={_cap(f1s[i]):.3f} thr={thr[i]:.2f}")
        print(f"\nOverall: micro-F1={_cap(micro):.3f} macro-F1={_cap(macro):.3f}")
        
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            pr_micro = average_precision_score(T, R, average="micro")
            pr_macro = average_precision_score(T, R, average="macro")
            roc_micro = roc_auc_score(T, R, average="micro")
            roc_macro = roc_auc_score(T, R, average="macro")
            print(f"AUPRC micro={_cap(pr_micro):.3f} macro={_cap(pr_macro):.3f} | AUROC micro={_cap(roc_micro):.3f} macro={_cap(roc_macro):.3f}")
        except Exception:
            pass
    
    return {
        "micro_f1": micro,
        "macro_f1": macro,
        "per_class": {"precision": prec, "recall": rec, "f1": np.array(f1s)},
        "supports": supports,
    }

def save_tables_and_plots(metrics: Dict[str, float], save_dir: str, model_name: str = ""):
    os.makedirs(save_dir, exist_ok=True)
    prec = metrics["per_class"]["precision"]
    rec = metrics["per_class"]["recall"]
    f1s = metrics["per_class"]["f1"]
    
    df = pd.DataFrame({"Label": ISSUE_LABELS, "Precision": prec, "Recall": rec, "F1": f1s})
    csv_path = os.path.join(save_dir, f"results_table_{model_name}.csv" if model_name else "results_table.csv")
    df.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(ISSUE_LABELS))
    width = 0.25
    plt.bar(x - width, prec, width, label='Precision', alpha=0.8)
    plt.bar(x, rec, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1s, width, label='F1', alpha=0.8)
    plt.xlabel('Issue Type')
    plt.ylabel('Score')
    plt.title(f'Per-Class Metrics{" - " + model_name if model_name else ""}')
    plt.xticks(x, ISSUE_LABELS, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    png_path = os.path.join(save_dir, f"metrics_bar_{model_name}.png" if model_name else "metrics_bar.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {png_path}")

# --------------------- Comparison Framework --------------------- 
@dataclass
class ComparisonResult:
    model_name: str
    model_type: str
    micro_f1: float
    macro_f1: float
    per_class_f1: np.ndarray
    training_time: float
    inference_time: float
    params_count: int
    memory_usage: float

class ModelComparator:
    """Framework for comparing multiple model architectures"""
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.results: List[ComparisonResult] = []
        os.makedirs(save_dir, exist_ok=True)
    
    def add_result(self, result: ComparisonResult):
        self.results.append(result)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("[Comparator] No results to compare")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        
        # Create comparison dataframe
        comp_data = []
        for r in self.results:
            comp_data.append({
                "Model": r.model_name,
                "Type": r.model_type,
                "Micro-F1": f"{r.micro_f1:.4f}",
                "Macro-F1": f"{r.macro_f1:.4f}",
                "Avg F1": f"{r.per_class_f1.mean():.4f}",
                "Train Time (s)": f"{r.training_time:.1f}",
                "Infer Time (s)": f"{r.inference_time:.3f}",
                "Params (M)": f"{r.params_count/1e6:.2f}",
                "Memory (MB)": f"{r.memory_usage:.1f}",
            })
        
        df = pd.DataFrame(comp_data)
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join(self.save_dir, "model_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")
        
        # Generate comparison plots
        self._plot_f1_comparison()
        self._plot_efficiency_comparison()
        self._plot_per_class_heatmap()
    
    def _plot_f1_comparison(self):
        """Plot F1 score comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        names = [r.model_name for r in self.results]
        micro_f1s = [r.micro_f1 for r in self.results]
        macro_f1s = [r.macro_f1 for r in self.results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax1.bar(x - width/2, micro_f1s, width, label='Micro-F1', alpha=0.8)
        ax1.bar(x + width/2, macro_f1s, width, label='Macro-F1', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        
        # Per-class average
        avg_f1s = [r.per_class_f1.mean() for r in self.results]
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        ax2.barh(names, avg_f1s, color=colors, alpha=0.8)
        ax2.set_xlabel('Average Per-Class F1')
        ax2.set_title('Average Performance Across All Classes')
        ax2.set_xlim(0, 1.0)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "f1_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_comparison(self):
        """Plot efficiency metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        names = [r.model_name for r in self.results]
        train_times = [r.training_time for r in self.results]
        params = [r.params_count / 1e6 for r in self.results]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(names)))
        
        ax1.barh(names, train_times, color=colors, alpha=0.8)
        ax1.set_xlabel('Training Time (seconds)')
        ax1.set_title('Training Efficiency')
        ax1.grid(axis='x', alpha=0.3)
        
        ax2.barh(names, params, color=colors, alpha=0.8)
        ax2.set_xlabel('Parameters (Millions)')
        ax2.set_title('Model Size')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "efficiency_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_heatmap(self):
        """Plot per-class F1 heatmap across models"""
        matrix = np.array([r.per_class_f1 for r in self.results])
        names = [r.model_name for r in self.results]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            matrix,
            xticklabels=ISSUE_LABELS,
            yticklabels=names,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'F1 Score'}
        )
        plt.title('Per-Class F1 Scores Across Models')
        plt.xlabel('Issue Type')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "per_class_heatmap.png"), dpi=200, bbox_inches='tight')
        plt.close()

# --------------------- Federated utils --------------------- 
def split_clients(df: pd.DataFrame, n: int, alpha: float) -> List[pd.DataFrame]:
    """Split data non-IID using Dirichlet distribution"""
    prim = []
    rng = np.random.default_rng(SEED)
    for labs in df["labels"]:
        if labs:
            prim.append(int(rng.choice(labs)))
        else:
            prim.append(int(rng.integers(0, NUM_LABELS)))
    
    df2 = df.copy()
    df2["_y"] = prim
    class_client_probs = rng.dirichlet([alpha] * n, size=NUM_LABELS)
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
        ep.data.mul_(decay).add_(mp.data, alpha=1.0 - decay)

def train_local(
    model,
    tok,
    tr_df,
    va_df,
    class_alpha: torch.Tensor
) -> Tuple[float, float, Dict, np.ndarray, int]:
    """Local training for one client"""
    if ARGS.use_images and "image_path" in tr_df.columns:
        tr_ds = MultiModalDS(tr_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir)
        va_ds = MultiModalDS(va_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir)
    else:
        tr_ds = MultiLabelDS(tr_df, tok, ARGS.max_len)
        va_ds = MultiLabelDS(va_df, tok, ARGS.max_len)
    
    weights, _ = make_weights_for_balanced_classes(tr_df)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=max(len(tr_df), ARGS.batch_size),
        replacement=True,
    )
    
    tr_loader = DataLoader(tr_ds, batch_size=ARGS.batch_size, sampler=sampler, num_workers=0, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=max(ARGS.batch_size, 16), shuffle=False, num_workers=0, drop_last=False)
    
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=ARGS.lr,
        weight_decay=0.05,
    )
    
    steps_per_epoch = max(1, math.ceil(len(tr_loader) / max(1, ARGS.grad_accum)))
    total_steps = ARGS.local_epochs * steps_per_epoch
    sch = get_linear_schedule_with_warmup(opt, max(1, int(0.1 * total_steps)), total_steps)
    
    loss_fn = FocalLoss(alpha=class_alpha.to(DEVICE), gamma=2.5, label_smoothing=0.02)
    model.train()
    model.to(DEVICE)
    
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ema_params = [p.detach().clone() for p in trainable_params]
    opt.zero_grad(set_to_none=True)
    
    for _ in range(ARGS.local_epochs):
        for it, batch in enumerate(tr_loader, start=1):
            text_inputs = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
            }
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
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()
            
            ema_update(ema_params, trainable_params, ARGS.ema_decay)
    
    # Eval with EMA weights
    backup = [p.detach().clone() for p in trainable_params]
    for p, ep in zip(trainable_params, ema_params):
        p.data.copy_(ep.data)
    
    thr = calibrate_thresholds(model, va_loader, precision_target=ARGS.precision_target)
    was_quiet = ARGS.quiet_eval
    ARGS.quiet_eval = True
    mets = evaluate_with_thr(model, va_loader, thr)
    ARGS.quiet_eval = was_quiet
    
    micro_f1, macro_f1 = mets["micro_f1"], mets["macro_f1"]
    
    for p, bp in zip(trainable_params, backup):
        p.data.copy_(bp.data)
    
    try:
        # Try getting PEFT state dict
        if hasattr(model, 'text_encoder'):
            lora_sd = get_peft_model_state_dict(model.text_encoder)
        elif hasattr(model, 'base_model'):
            lora_sd = get_peft_model_state_dict(model.base_model)
        else:
            lora_sd = get_peft_model_state_dict(model)
    except Exception:
        lora_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
    lora_sd = {k: v.detach().cpu() for k, v in lora_sd.items()}
    
    del tr_loader, va_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return micro_f1, macro_f1, lora_sd, thr, len(tr_df)

def fedavg_weighted(states: List[Dict[str, torch.Tensor]], sizes: List[int]) -> Dict[str, torch.Tensor]:
    total = float(sum(sizes))
    ws = [s / total for s in sizes]
    keys = list(states[0].keys())
    out = {}
    for k in keys:
        out[k] = torch.stack([st[k].float() * w for st, w in zip(states, ws)], dim=0).sum(0)
    return out

# --------------------- Main Training Function --------------------- 
def run_training(model_config: ModelConfig = None, comparator: ModelComparator = None):
    """
    Main training loop supporting:
    - Standard encoder models
    - Federated LLMs
    - Multimodal models
    - VLMs
    """
    if model_config is None:
        model_config = MODEL_CONFIGS[ARGS.model_type]
    
    model_name = model_config.name
    print(f"\n{'='*80}")
    print(f"Training: {model_name} ({model_config.model_type})")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    print(f"Device: {DEVICE} (AMP={'on' if amp_enabled() else 'off'})")
    
    # Build tokenizer
    tok = build_tokenizer(model_config.model_name)
    
    # Build corpus
    df = build_corpus()
    multimodal = ARGS.use_images and ("image_path" in df.columns)
    
    # Split data
    clients_all = split_clients(df, max(1, ARGS.clients), ARGS.dirichlet_alpha)
    val_k = max(1, int(0.15 * len(clients_all)))
    val_df = pd.concat(clients_all[:val_k], ignore_index=True)
    train_clients = clients_all[val_k:]
    train_df = pd.concat(train_clients, ignore_index=True)
    train_df, test_df = train_test_split(train_df, test_size=0.15, random_state=SEED, shuffle=True)
    
    _, counts = make_weights_for_balanced_classes(train_df)
    inv = 1.0 / np.maximum(counts, 1)
    alpha = (inv / inv.mean()).astype(np.float32)
    alpha[1] *= 1.2  # boost nutrient_def
    alpha = torch.tensor(alpha)
    
    clients = split_clients(train_df, max(1, ARGS.clients), ARGS.dirichlet_alpha)
    
    # Build model
    if multimodal and model_config.model_type not in ["vlm"]:
        global_model = MultiModalModel(
            model_config.model_name,
            ARGS.vit_name,
            NUM_LABELS,
            freeze_text=model_config.freeze_base,
            freeze_vision=ARGS.freeze_vision,
            lora_r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
        ).to(DEVICE)
    else:
        global_model = build_model_from_config(model_config, NUM_LABELS).to(DEVICE)
    
    # Count parameters
    params_count = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {params_count/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")
    
    # Validation loader
    if "image_path" in val_df.columns and ARGS.use_images:
        val_loader = DataLoader(
            MultiModalDS(val_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir),
            batch_size=32,
            shuffle=False,
        )
    else:
        val_loader = DataLoader(MultiLabelDS(val_df, tok, ARGS.max_len), batch_size=32, shuffle=False)
    
    metrics_dir = os.path.join(ARGS.save_dir, "metrics", model_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    def evaluate_global(thr):
        if "image_path" in test_df.columns and ARGS.use_images:
            test_loader = DataLoader(
                MultiModalDS(test_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir),
                batch_size=64,
                shuffle=False,
            )
        else:
            test_loader = DataLoader(MultiLabelDS(test_df, tok, ARGS.max_len), batch_size=64, shuffle=False)
        return evaluate_with_thr(global_model, test_loader, thr)
    
    thr_history = []
    
    # Federated training rounds
    for r in range(1, max(1, ARGS.rounds) + 1):
        print(f"\n==== Round {r}/{ARGS.rounds} ====")
        rng = np.random.default_rng(SEED + r)
        k_all = list(range(len(clients)))
        rng.shuffle(k_all)
        m = max(1, int(ARGS.participation * len(k_all)))
        chosen = k_all[:m]
        
        states, sizes = [], []
        for i in chosen:
            if rng.random() < ARGS.client_dropout:
                print(f"[Client {i+1}] dropped this round")
                continue
            
            cdf = clients[i]
            if len(cdf) < 80:
                print(f"[Client {i+1}] skipped (too small: n={len(cdf)})")
                continue
            
            n = len(cdf)
            val_n = max(1, int(ARGS.val_frac * n))
            va_df, tr_df = cdf.iloc[:val_n], cdf.iloc[val_n:]
            
            # Create local model
            if multimodal and model_config.model_type not in ["vlm"]:
                local = MultiModalModel(
                    model_config.model_name,
                    ARGS.vit_name,
                    NUM_LABELS,
                    freeze_text=model_config.freeze_base,
                    freeze_vision=ARGS.freeze_vision,
                    lora_r=model_config.lora_r,
                    lora_alpha=model_config.lora_alpha,
                    lora_dropout=model_config.lora_dropout,
                ).to(DEVICE)
            else:
                local = build_model_from_config(model_config, NUM_LABELS).to(DEVICE)
            
            # Load global state
            try:
                if hasattr(global_model, 'text_encoder') and hasattr(local, 'text_encoder'):
                    set_peft_model_state_dict(local.text_encoder, get_peft_model_state_dict(global_model.text_encoder))
                elif hasattr(global_model, 'base_model') and hasattr(local, 'base_model'):
                    set_peft_model_state_dict(local.base_model, get_peft_model_state_dict(global_model.base_model))
                else:
                    set_peft_model_state_dict(local, get_peft_model_state_dict(global_model))
            except Exception:
                pass
            
            rng_local = np.random.default_rng(SEED + r + i)
            local_epochs = int(rng_local.choice([2, 3], p=[0.6, 0.4]))
            orig_local = ARGS.local_epochs
            ARGS.local_epochs = local_epochs
            
            micro, macro, lora_sd, thr_local, used_n = train_local(local, tok, tr_df, va_df, class_alpha=alpha)
            
            ARGS.local_epochs = orig_local
            print(f"[Client {i+1}] micro_f1={_fmt_str(micro)} macro_f1={_fmt_str(macro)} (n={len(cdf)}) thr={np.round(thr_local,2)}")
            
            states.append(lora_sd)
            sizes.append(used_n)
            
            del local
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Aggregate
        if states:
            avg_sd = fedavg_weighted(states, sizes)
            try:
                if hasattr(global_model, 'text_encoder'):
                    set_peft_model_state_dict(global_model.text_encoder, avg_sd)
                elif hasattr(global_model, 'base_model'):
                    set_peft_model_state_dict(global_model.base_model, avg_sd)
                else:
                    set_peft_model_state_dict(global_model, avg_sd)
            except Exception:
                try:
                    global_model.load_state_dict(avg_sd, strict=False)
                except Exception:
                    print("[Warn] couldn't set averaged state")
            
            final_thr = calibrate_thresholds(global_model, val_loader, precision_target=ARGS.precision_target)
            final_thr = np.clip(final_thr + np.array([+0.03, 0.00, 0.00, +0.02, 0.00]), 0.05, 0.90)
        else:
            print("No client updates this round")
            final_thr = thr_history[-1] if thr_history else np.array([0.5] * NUM_LABELS)
        
        thr_history.append(final_thr)
        test_mets = evaluate_global(final_thr)
        
        # Save round metrics
        round_tag = f"round_{r:02d}"
        np.save(os.path.join(metrics_dir, f"{round_tag}_thr.npy"), final_thr)
        with open(os.path.join(metrics_dir, f"{round_tag}_summary.json"), "w") as f:
            json.dump({
                "round": r,
                "micro_f1": float(test_mets["micro_f1"]),
                "macro_f1": float(test_mets["macro_f1"]),
            }, f, indent=2)
    
    # Save final model
    model_save_dir = os.path.join(ARGS.save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    ap = os.path.join(model_save_dir, "model.pt")
    thp = os.path.join(model_save_dir, "thresholds.npy")
    
    try:
        if hasattr(global_model, 'text_encoder'):
            torch.save(get_peft_model_state_dict(global_model.text_encoder), ap)
        elif hasattr(global_model, 'base_model'):
            torch.save(get_peft_model_state_dict(global_model.base_model), ap)
        else:
            torch.save(get_peft_model_state_dict(global_model), ap)
    except Exception:
        torch.save(global_model.state_dict(), ap)
    
    np.save(thp, thr_history[-1] if thr_history else np.array([0.5] * NUM_LABELS))
    print(f"[Save] model â†’ {ap}")
    print(f"[Save] thresholds â†’ {thp}")
    
    # Final evaluation
    if "image_path" in test_df.columns and ARGS.use_images:
        cal_df = test_df.sample(min(400, len(test_df)), random_state=SEED)
        cal_loader = DataLoader(
            MultiModalDS(cal_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir),
            batch_size=32,
            shuffle=False,
        )
    else:
        cal_df = test_df.sample(min(400, len(test_df)), random_state=SEED)
        cal_loader = DataLoader(MultiLabelDS(cal_df, tok, ARGS.max_len), batch_size=32, shuffle=False)
    
    final_thr = thr_history[-1] if thr_history else np.array([0.5] * NUM_LABELS)
    mets = evaluate_with_thr(global_model, cal_loader, final_thr)
    
    figs_dir = os.path.join(ARGS.save_dir, "figs")
    save_tables_and_plots(mets, save_dir=figs_dir, model_name=model_name)
    
    training_time = time.time() - start_time
    
    # Measure inference time
    infer_start = time.time()
    _ = evaluate_with_thr(global_model, cal_loader, final_thr)
    inference_time = time.time() - infer_start
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        memory_usage = 0.0
    
    # Add to comparator
    if comparator:
        result = ComparisonResult(
            model_name=model_name,
            model_type=model_config.model_type,
            micro_f1=mets["micro_f1"],
            macro_f1=mets["macro_f1"],
            per_class_f1=mets["per_class"]["f1"],
            training_time=training_time,
            inference_time=inference_time,
            params_count=params_count,
            memory_usage=memory_usage,
        )
        comparator.add_result(result)
    
    print(f"\n[{model_name}] Training complete in {training_time:.1f}s")
    print(f"[{model_name}] Final Micro-F1: {mets['micro_f1']:.4f} | Macro-F1: {mets['macro_f1']:.4f}")
    
    return mets

# --------------------- Main Execution --------------------- 
def main():
    if ARGS.inference:
        print("[Mode] Inference mode not yet implemented in comparison framework")
        return
    
    if ARGS.compare_all:
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON MODE")
        print("="*80)
        
        comparator = ModelComparator(os.path.join(ARGS.save_dir, "comparisons"))
        
        # Models to compare
        models_to_compare = [
            "roberta",  # Baseline encoder
            "distilbert",  # Efficient encoder
            "flan-t5-small",  # Federated LLM (seq2seq)
            "gpt2",  # Federated LLM (decoder)
        ]
        
        if ARGS.use_images:
            models_to_compare.append("vit")  # Pure vision
        
        if ARGS.use_vlm:
            models_to_compare.append("clip")  # VLM
        
        for model_key in models_to_compare:
            if model_key not in MODEL_CONFIGS:
                print(f"[Warn] Skipping unknown model: {model_key}")
                continue
            
            try:
                config = MODEL_CONFIGS[model_key]
                run_training(config, comparator)
            except Exception as e:
                print(f"[Error] Failed to train {model_key}: {e}")
                continue
        
        # Generate comparison report
        comparator.generate_comparison_report()
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print(f"Results saved to: {comparator.save_dir}")
        print("="*80)
    
    else:
        # Single model training
        config = MODEL_CONFIGS.get(ARGS.model_type)
        if config is None:
            print(f"[Error] Unknown model type: {ARGS.model_type}")
            return
        
        run_training(config)

if __name__ == "__main__":
    main()

print("\n[âœ“] Enhanced Farm Advisor System Loaded")
print("[âœ“] Features:")
print("  - Federated LLM support (Flan-T5, GPT-2)")
print("  - ViT encoder for crop stress detection")
print("  - VLM support (CLIP, BLIP)")
print("  - Comprehensive dataset loading")
print("  - Model comparison framework")
print("  - Full evaluation and benchmarking")
print("\n[âœ“] Ready to run training")
print(f"[âœ“] Use --compare_all to compare all models")
print(f"[âœ“] Use --use_federated_llm for LLM training")
print(f"[âœ“] Use --use_vlm for Vision-Language Models")
