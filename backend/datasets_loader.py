# datasets_loader.py
"""
Load/download datasets (text + image). Provide an unified mapping
and simple weak-labeling for text-only samples.

Datasets attempted:
 - PlantVillage (images) via 'plant_village' or fallback to placeholder
 - gardian (CGIAR) via 'CGIAR/gardian-ai-ready-docs' (text)
 - argilla farming (if available)
 - ag_news filtered for agriculture mentions (text)
Also provides synthetic text samples if network unavailable.
"""

import os, random, math
from typing import List, Dict
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
import io

# Basic weak labeling keywords (same categories)
KW = {
    "water": ["dry","wilting","wilt","parched","drought","moisture","irrigat","water stress","droop","cracking soil"],
    "nutrient": ["nitrogen","phosphorus","potassium","npk","fertilizer","chlorosis","interveinal","spad","yellowing"],
    "pest": ["pest","aphid","whitefly","borer","hopper","weevil","caterpillar","thrips","mites","insect","frass"],
    "disease": ["blight","rust","mildew","rot","leaf spot","necrosis","fungal","bacterial","viral","lesion","mosaic"],
    "heat": ["heatwave","hot","scorch","sunburn","thermal stress","high temperature","leaf burn"],
}

ISSUE_LABELS = ["water_stress","nutrient_def","pest_risk","disease_risk","heat_stress"]

def text_to_labels(text: str):
    t = text.lower()
    labs = []
    if any(k in t for k in KW["water"]): labs.append(0)
    if any(k in t for k in KW["nutrient"]): labs.append(1)
    if any(k in t for k in KW["pest"]): labs.append(2)
    if any(k in t for k in KW["disease"]): labs.append(3)
    if any(k in t for k in KW["heat"]): labs.append(4)
    return sorted(list(set(labs)))

def download_text_datasets(max_per_source=2000, offline=False):
    out = []
    # gardian
    try:
        ds = load_dataset("CGIAR/gardian-ai-ready-docs", split="train", streaming=not offline)
        cnt = 0
        for r in ds:
            txt = (r.get("text") or r.get("content") or "").strip()
            if txt:
                labs = text_to_labels(txt)
                if labs:
                    out.append({"text": txt, "labels": labs, "image": None})
                    cnt += 1
                    if cnt >= max_per_source: break
    except Exception:
        pass

    # argilla farming
    try:
        ds2 = load_dataset("argilla/farming", split="train", streaming=not offline)
        cnt = 0
        for r in ds2:
            q = str(r.get("evolved_questions","")).strip()
            a = str(r.get("domain_expert_answer","")).strip()
            txt = (q + " " + a).strip()
            if txt:
                labs = text_to_labels(txt)
                if labs:
                    out.append({"text": txt, "labels": labs, "image": None})
                    cnt += 1
                    if cnt >= max_per_source: break
    except Exception:
        pass

    # ag_news filtered
    try:
        ds3 = load_dataset("ag_news", split="train", streaming=not offline)
        cnt = 0
        for r in ds3:
            txt = (r.get("text") or "").strip()
            if txt and ("farm" in txt.lower() or "crop" in txt.lower() or "agri" in txt.lower()):
                labs = text_to_labels(txt)
                if labs:
                    out.append({"text": txt, "labels": labs, "image": None})
                    cnt += 1
                    if cnt >= max_per_source: break
    except Exception:
        pass

    # fallback local synthetic
    if len(out) < 200:
        synth = [
            ("Topsoil is cracking and leaves droop at midday", [0]),
            ("Interveinal chlorosis on older leaves suggests nitrogen deficiency", [1]),
            ("Powdery mildew on lower canopy after humid nights", [3]),
            ("Whitefly and sticky residue under leaves", [2]),
            ("Sun scorch on exposed leaves during heatwave", [4]),
        ]
        for s,l in synth:
            out.append({"text": s, "labels": l, "image": None})
    return out

def download_image_datasets(max_images=2000, offline=False, image_dir="data_real/images"):
    os.makedirs(image_dir, exist_ok=True)
    mapping = []
    # try plant_village (community dataset may not be on HF) — use 'plant_village' if available
    try:
        ds = load_dataset("plant_village", split="train", streaming=not offline)
        cnt = 0
        for r in ds:
            # dataset returns image and label name, we use label text to weak-label
            pil = r.get("image")
            lbl = str(r.get("label", "") or r.get("label_name", "")).lower()
            text_hint = lbl
            labs = text_to_labels(text_hint)
            if not labs:
                # try some heuristics: 'blight' etc
                if "rust" in lbl or "mildew" in lbl or "spot" in lbl: labs=[3]
                elif "healthy" in lbl: labs=[]
            if pil is None:
                continue
            # save image file
            fname = f"{cnt:07d}.jpg"
            path = os.path.join(image_dir, fname)
            if not os.path.exists(path):
                pil.convert("RGB").save(path, format="JPEG")
            mapping.append({"image": path, "text": "", "labels": labs})
            cnt += 1
            if cnt >= max_images: break
    except Exception:
        pass

    # fallback: try 'plantdoc' or image-net style
    try:
        ds2 = load_dataset("plantdoc", split="train", streaming=not offline)
        cnt = len(mapping)
        for r in ds2:
            pil = r.get("image")
            lbl = str(r.get("label", "")).lower()
            labs = text_to_labels(lbl)
            if pil is None: continue
            fname = f"{cnt:07d}.jpg"
            path = os.path.join(image_dir, fname)
            if not os.path.exists(path):
                pil.convert("RGB").save(path, format="JPEG")
            mapping.append({"image": path, "text": "", "labels": labs})
            cnt += 1
            if cnt >= max_images: break
    except Exception:
        pass

    # if no images downloaded, generate placeholders (colored images) for debugging
    if len(mapping) == 0:
        from PIL import Image, ImageDraw
        for i in range(min(200, max_images)):
            img = Image.new("RGB", (224,224), (int(200*(i%3)), int(120*(i%5)), int(80*(i%7))))
            p = os.path.join(image_dir, f"placeholder_{i:05d}.jpg")
            img.save(p)
            mapping.append({"image": p, "text": "", "labels": []})
    return mapping

def build_unified_dataset(max_per_text=2000, max_images=2000, offline=False):
    texts = download_text_datasets(max_per_source=max_per_text, offline=offline)
    images = download_image_datasets(max_images=max_images, offline=offline)
    # combine: images entries may have labels inferred; also create multimodal items by pairing random text with images
    unified = []
    # take text-only
    for t in texts:
        unified.append({"text": t["text"], "image": None, "labels": t["labels"]})
    # image-only
    for im in images:
        unified.append({"text": "", "image": im["image"], "labels": im["labels"]})
    # multimodal pairs: pair some text samples with images (synthetic pairing)
    rand = random.Random(42)
    for i in range(min(500, len(images), len(texts))):
        t = texts[i % len(texts)]
        im = images[i % len(images)]
        # combine labels union
        labs = sorted(list(set(t["labels"] + im["labels"])))
        unified.append({"text": t["text"], "image": im["image"], "labels": labs})
    # shuffle
    rand.shuffle(unified)
    return unified
