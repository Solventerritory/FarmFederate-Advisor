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
    r"temp=(?P<t>\d+(?:\.\d+)?)°C.*?humidity=(?P<h>\d+(?:\.\d+)?)%.*?"
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
    trend = np.random.choice(["↑", "↓", "→"], p=[0.3, 0.3, 0.4])
    return (
        f"SENSORS: soil_moisture={soil_m}%, soil_pH={soil_ph}, temp={temp}°C, "
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
            print(f"    ✓ Loaded {len(datasets[ds_name])} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    # Image datasets
    for ds_name in AGRICULTURAL_IMAGE_DATASETS:
        try:
            print(f"  Loading {ds_name}...")
            df = prepare_images_from_hf(ds_name, max_per_source, ARGS.image_dir)
            if len(df) > 0:
                datasets[ds_name] = df
                print(f"    ✓ Loaded {len(df)} image samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
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
print("[✓] Core modules loaded successfully")
print(f"[✓] Device: {DEVICE}")
print(f"[✓] Model configs available: {list(MODEL_CONFIGS.keys())}")
