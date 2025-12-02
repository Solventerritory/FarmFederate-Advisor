#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
datasets_loader.py — text + image dataset builders for FarmFederate.

Text sources (downloaded via Hugging Face `datasets`):
    - CGIAR/gardian-ai-ready-docs
    - argilla/farming
    - ag_news (agri-filtered)
    - synthetic local "LocalMini" agri log-style data

Image source:
    - plantvillage (Hugging Face)  — plant disease images
      (if unavailable, falls back to a dummy single-color image in memory)

All text datasets are mapped to the 5 issue labels using weak, rule-based
labelling (keywords) identical in spirit to farm_advisor.py.

The main entry points are:

    build_text_corpus_mix(...)
    load_plant_images_hf(...)
"""

import os
import re
import time
import random
import hashlib
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from PIL import Image

try:
    from datasets import load_dataset, DownloadConfig
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

# ----------------- core labels -----------------
ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
LABEL_TO_ID = {k: i for i, k in enumerate(ISSUE_LABELS)}
NUM_LABELS = len(ISSUE_LABELS)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

# ----------------- text utils -----------------
def _norm(txt: str) -> str:
    return re.sub(r"\s+", " ", str(txt)).strip()


def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))


def _lang_ok(s: str) -> bool:
    return _ascii_ratio(s) >= 0.6


# ------------- weak labels & ag context -------------
KW = {
    "water": [
        "dry", "wilting", "wilt", "parched", "drought", "moisture", "irrigation",
        "canopy stress", "water stress", "droop", "cracking soil", "hard crust",
        "soil moisture low"
    ],
    "nutrient": [
        "nitrogen", "phosphorus", "potassium", "npk", "fertilizer", "fertiliser",
        "chlorosis", "chlorotic", "interveinal", "leaf color chart", "lcc", "spad",
        "low spad", "older leaves yellowing", "old leaves yellowing",
        "necrotic margin", "micronutrient", "deficiency"
    ],
    "pest": [
        "pest", "aphid", "whitefly", "borer", "hopper", "weevil", "caterpillar",
        "larvae", "thrips", "mites", "trap", "sticky residue", "honeydew",
        "chewed", "webbing", "frass", "insect"
    ],
    "disease": [
        "blight", "rust", "mildew", "smut", "rot", "leaf spot", "necrosis",
        "pathogen", "fungal", "bacterial", "viral", "lesion", "mosaic",
        "wilt disease", "canker", "powdery mildew", "downy"
    ],
    "heat": [
        "heatwave", "hot", "scorch", "sunburn", "thermal stress",
        "high temperature", "blistering", "desiccation", "sun scorch", "leaf burn",
        "heat stress"
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
                "potassium", "leaf color chart", "lcc", "low spad", "spad"
            ]
        )
        qualified_yellow = (
            "yellowing" in t
            and ("older leaves" in t or "old leaves" in t)
        )
        if strong_n or qualified_yellow:
            labs.add("nutrient_def")
    if any(k in t for k in KW["pest"]):
        labs.add("pest_risk")
    if any(k in t for k in KW["disease"]):
        labs.add("disease_risk")
    if any(k in t for k in KW["heat"]):
        labs.add("heat_stress")
    return [LABEL_TO_ID[x] for x in sorted(labs)]


# ----------------- synthetic sensors + fuse -----------------
def simulate_sensor_summary() -> str:
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
    main_txt = _norm(main_txt)
    if main_txt.strip().startswith("SENSORS:"):
        base = main_txt
        if "LOG:" not in base:
            base = f"{base}\nLOG: (no additional log)"
    else:
        base = f"{sensor_txt}\nLOG: {main_txt}"
    if mqtt_msg:
        base = f"{base}\nMQTT: {_norm(mqtt_msg)}"
    return base


# ----------------- synthetic local corpus -----------------
LOCAL_BASE = [
    "Maize leaves show interveinal chlorosis and older leaves are yellowing after light rains.",
    "Tomato plants have whiteflies; sticky residue under leaves; some curling.",
    "Rice field shows cracked, dry soil; seedlings drooping under midday sun.",
    "Wheat leaves with orange pustules; reduced tillering; humid mornings reported.",
    "Chili plants show sun scorch on exposed fruits during heatwave; leaf edges crisping.",
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

CROPS = ["rice", "wheat", "maize", "soybean", "cotton", "tomato", "chili", "potato", "banana", "cabbage"]
WEATHERS = ["a hot, dry wind", "sudden heavy rain", "two cloudy days", "a heatwave", "no rainfall for a week"]

TEMPLATES = [
    "Farmer noted {symptom} while sensors read temp {temp}°C and humidity {hum}%.",
    "{crop} field observed {symptom}; irrigation last 48h minimal; VPD around {vpd} kPa.",
    "After {weather}, plants show {symptom}. Soil moisture near {sm}% and pH {ph}.",
]


def make_balanced_local(n_per: int = 300, n_per_nutrient: int = 600) -> List[str]:
    seeds = {
        "water_stress": [
            "Topsoil is cracking and leaves droop at midday; irrigation uneven.",
            "Canopy stress at noon; mulch missing; dry beds observed.",
        ],
        "nutrient_def": [
            "Interveinal chlorosis on older leaves suggests nitrogen deficiency.",
            "Marginal necrosis indicates potassium shortfall.",
            "Leaf Color Chart shows low score; possible N deficiency.",
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
    texts = []
    for k, lst in seeds.items():
        reps = n_per_nutrient if k == "nutrient_def" else n_per
        for _ in range(reps):
            texts.append(random.choice(lst))
    random.shuffle(texts)
    return texts


def _maybe_read_mqtt(mqtt_csv: str) -> List[str]:
    if mqtt_csv and os.path.exists(mqtt_csv):
        df = pd.read_csv(mqtt_csv)
        if "message" in df.columns:
            return df["message"].astype(str).tolist()
    return []


def build_localmini(max_samples: int = 0, mqtt_csv: str = "", extra_csv: str = "") -> pd.DataFrame:
    mqtt_msgs = _maybe_read_mqtt(mqtt_csv)
    texts = list(LOCAL_BASE) + make_balanced_local(300, 600)

    # synthetic sensor+log samples
    for _ in range(2000):
        s = random.choice(TEMPLATES).format(
            symptom=random.choice(SYMPTOMS),
            crop=random.choice(CROPS),
            temp=round(np.clip(np.random.normal(32, 4), 15, 45), 1),
            hum=int(np.clip(np.random.normal(55, 15), 15, 95)),
            vpd=round(np.clip(np.random.normal(1.8, 0.7), 0.2, 4.0), 1),
            sm=round(np.clip(np.random.normal(20, 7), 2, 60), 1),
            ph=round(np.clip(np.random.normal(6.5, 0.6), 4.5, 8.5), 1),
            weather=random.choice(WEATHERS),
        )
        sensor = simulate_sensor_summary()
        mqtt = random.choice(mqtt_msgs) if mqtt_msgs and random.random() < 0.4 else ""
        texts.append(fuse_text(sensor, s, mqtt))

    # extra CSV of farmer queries (optional)
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


# ----------------- HF loading helpers -----------------
def _load_ds(name, split=None, streaming=False):
    if not HAS_DATASETS:
        raise RuntimeError("The `datasets` library is not installed.")
    dlconf = DownloadConfig(max_retries=3)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kw = {"streaming": streaming, "download_config": dlconf}
    if token:
        kw["token"] = token
        kw["use_auth_token"] = token
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
    kw["streaming"] = True
    return load_dataset(name, split=split, **kw)


AGRI_RE = re.compile(
    r"\b(agri|agriculture|farm|farmer|farming|crop|soil|harvest|irrigat|pest|blight|"
    r"drought|yield|wheat|rice|paddy|maize|soy|cotton|fertiliz|orchard|greenhouse|horticul)\b",
    re.I,
)
NON_AG_NOISE = re.compile(
    r"\b(NFL|NBA|MLB|NHL|tennis|golf|soccer|stocks?|Nasdaq|Dow Jones|Hollywood|movie|concert)\b",
    re.I,
)


def build_gardian_stream(max_per: int = 2000) -> List[str]:
    ds = _load_ds("CGIAR/gardian-ai-ready-docs", streaming=True)
    texts = []
    seen = 0
    if isinstance(ds, dict):
        splits = ds.keys()
    else:
        splits = [None]
    for sp in splits:
        subset = ds[sp] if sp is not None else ds
        for r in subset:
            raw = (r.get("text") or r.get("content") or "").strip()
            if raw and _lang_ok(raw):
                texts.append(_norm(raw))
                seen += 1
                if seen >= max_per:
                    break
        if seen >= max_per:
            break
    return texts


def build_argilla_stream(max_per: int = 2000) -> List[str]:
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


def build_agnews_agri(max_per: int = 2000) -> List[str]:
    train = _load_ds("ag_news", split="train", streaming=True)
    texts = []
    seen = 0
    for r in train:
        raw = (r.get("text") or "").strip()
        if (
            raw
            and AGRI_RE.search(raw)
            and not NON_AG_NOISE.search(raw)
            and _lang_ok(raw)
        ):
            texts.append(_norm(raw))
            seen += 1
            if seen >= max_per:
                break
    return texts


# ----------------- MIX builder for text -----------------
def build_text_corpus_mix(
    mix_sources: str = "gardian,argilla,agnews,localmini",
    max_per_source: int = 2000,
    max_samples: int = 0,
    mqtt_csv: str = "",
    extra_csv: str = "",
) -> pd.DataFrame:
    sources = [s.strip().lower() for s in mix_sources.split(",") if s.strip()]
    pool: List[Tuple[str, str]] = []

    def _try(name: str, fn):
        print(f"[Mix] loading {name} (<= {max_per_source}) ...")
        try:
            texts = fn(max_per_source)
            pool.extend([(name, t) for t in texts])
            print(f"[Mix] {name} added {len(texts)} rows")
        except Exception as e:
            print(f"[Mix] {name} skipped: {e}")

    if "gardian" in sources:
        _try("gardian", build_gardian_stream)
    if "argilla" in sources:
        _try("argilla", build_argilla_stream)
    if "agnews" in sources:
        _try("agnews", build_agnews_agri)
    if "localmini" in sources:
        df_local = build_localmini(max_per_source, mqtt_csv, extra_csv)
        for t, labs in df_local[["text", "labels"]].itertuples(index=False):
            pool.append(("localmini", t))

    # deduplicate by text hash
    seen = set()
    dedup: List[Tuple[str, str]] = []
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
    print("[Mix] source breakdown:")
    print(df["source"].value_counts())

    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=SEED).reset_index(drop=True)
    return df[["text", "labels"]]


# ----------------- Image dataset (PlantVillage via HF) -----------------
def load_plant_images_hf(max_images: int = 4000):
    """
    Returns a Hugging Face dataset with a column `image` of PIL images.

    Uses `plantvillage` if available. If `datasets` is missing or
    download fails, returns None and you should fall back to dummy images.
    """
    if not HAS_DATASETS:
        print("[Images] datasets not installed; no HF images.")
        return None
    try:
        print("[Images] loading plantvillage from HF (this will download once)...")
        ds = load_dataset("plantvillage", "color", split="train")
        if max_images and len(ds) > max_images:
            ds = ds.shuffle(seed=SEED).select(range(max_images))
        print(f"[Images] plantvillage loaded: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"[Images] failed to load plantvillage: {e}")
        return None


# ----------------- Convenience for summaries -----------------
def summarize_labels(df: pd.DataFrame, tag: str = "set"):
    counts = np.zeros(NUM_LABELS, int)
    for labs in df["labels"]:
        for k in labs:
            counts[k] += 1
    print(f"[{tag}] label counts:", {ISSUE_LABELS[i]: int(c) for i, c in enumerate(counts)})
