#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agri_fed_all_datasets.py

Comprehensive single-file pipeline:
 - Downloads/loads image datasets (PlantVillage / DeepWeeds / PlantDoc) when possible
 - Streams/loads text datasets: CGIAR/gardian-ai-ready-docs, argilla/farming, ag_news
 - Builds synthetic localmini samples and sensor summaries
 - Generates BLIP captions for images (if online)
 - Fuses text+image examples into a multimodal corpus
 - Maps labels into 5 target issue labels
 - Splits clients using Dirichlet non-IID and runs federated LoRA training on a CLIP backbone
 - Saves adapters and thresholds

Usage (quick smoke test):
  pip install -U "torch" torchvision transformers datasets tensorflow-datasets peft accelerate tqdm scikit-learn pandas pillow
  python backend/agri_fed_all_datasets.py --data_dir ./data_all --save_dir ./checkpoints_all --quick

Notes:
 - Use --quick to limit downloads and iterations for a faster test.
 - If offline, provide local dataset files in the expected structure.
"""
import os, sys, time, json, math, random, argparse, gc, hashlib, shutil, re
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------- Defensive seeding ----------------------
SEED = 12345
random.seed(SEED); np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

# Try importing torch and handle CUDA device-side assert fallback
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.manual_seed_all(SEED)
        except Exception as e:
            print("Warning: cuda seed failed, disabling GPUs. Exception:", e)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            # re-import torch to pick up env change (best-effort)
            import importlib
            importlib.reload(torch)
except Exception as e:
    print("Torch import error, disabling CUDA. Exception:", e)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import importlib
    torch = importlib.import_module("torch")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------- HF / libs ----------------------
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# datasets + tfds
try:
    from datasets import load_dataset, DownloadConfig
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    import tensorflow_datasets as tfds
    HAS_TFDS = True
except Exception:
    HAS_TFDS = False

# PEFT/LoRA
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# image transforms
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------- CLI ----------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_all")
    ap.add_argument("--save_dir", type=str, default="checkpoints_all")
    ap.add_argument("--hf_clip", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--hf_blip", type=str, default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--local_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.25)
    ap.add_argument("--participation", type=float, default=0.8)
    ap.add_argument("--client_dropout", type=float, default=0.05)
    ap.add_argument("--prior_scale", type=float, default=0.30)
    ap.add_argument("--label_noise", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true", help="limit downloads and samples for a quick run")
    ap.add_argument("--offline", action="store_true", help="avoid HF downloads (use local files if present)")
    ap.add_argument("--quiet_eval", action="store_true")
    return ap.parse_args()

ARGS = get_args()
os.makedirs(ARGS.data_dir, exist_ok=True)
os.makedirs(ARGS.save_dir, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---------------------- Labels ----------------------
ISSUE_LABELS = ["water_stress","nutrient_def","pest_risk","disease_risk","heat_stress"]
LABEL_TO_ID = {k:i for i,k in enumerate(ISSUE_LABELS)}
NUM_LABELS = len(ISSUE_LABELS)

# ---------------------- Utilities ----------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def sha1_of_file(path):
    import hashlib
    h = hashlib.sha1()
    with open(path,'rb') as f:
        while True:
            b = f.read(8192)
            if not b: break
            h.update(b)
    return h.hexdigest()

# ---------------------- Sensor generator & fusion ----------------------
_SENS_RE = re.compile(
    r"soil_moisture=(?P<sm>\d+(?:\.\d+)?)%.*?soil_pH=(?P<ph>\d+(?:\.\d+)?).*?temp=(?P<t>\d+(?:\.\d+)?)°C.*?humidity=(?P<h>\d+(?:\.\d+)?)%.*?VPD=(?P<vpd>\d+(?:\.\d+)?) kPa.*?rainfall_24h=(?P<rf>\d+(?:\.\d+)?)mm",
    re.I | re.S
)
def simulate_sensor_summary():
    soil_m = round(np.clip(np.random.normal(30, 6), 5, 60), 1)
    soil_ph = round(np.clip(np.random.normal(6.5, 0.5), 4.5, 8.5), 1)
    temp   = round(np.clip(np.random.normal(29, 5), 12, 45), 1)
    hum    = round(np.clip(np.random.normal(60, 15), 20, 95), 0)
    vpd    = round(np.clip(np.random.normal(1.4, 0.5), 0.2, 3.5), 1)
    rain   = round(np.clip(np.random.normal(1.0, 1.0), 0.0, 20.0), 1)
    trend  = np.random.choice(["↑","↓","→"], p=[0.3,0.3,0.4])
    return f"SENSORS: soil_moisture={soil_m}%, soil_pH={soil_ph}, temp={temp}°C, humidity={hum}%, VPD={vpd} kPa, rainfall_24h={rain}mm (trend: {trend})."

def fuse_text(sensor_txt:str, main_txt:str, mqtt_msg:str="") -> str:
    if main_txt.strip().startswith("SENSORS:"):
        base = f"{main_txt.strip()}"
        if "LOG:" not in base:
            base = f"{base}\nLOG: (no additional log)"
    else:
        base = f"{sensor_txt}\nLOG: {re.sub(r'\\s+',' ', main_txt).strip()}"
    return f"{base}{(f'\\nMQTT: {mqtt_msg.strip()}' if mqtt_msg else '')}"

def _parse_sensors(text: str) -> Optional[Dict[str,float]]:
    m = _SENS_RE.search(text)
    if not m: return None
    try:
        return dict(sm=float(m.group("sm")), ph=float(m.group("ph")), t=float(m.group("t")), h=float(m.group("h")), vpd=float(m.group("vpd")), rf=float(m.group("rf")))
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

# ---------------------- Weak-label keywords (from your ref) ----------------------
KW = {
    "water": ["dry","wilting","wilt","parched","drought","moisture","irrigation","canopy stress","water stress","droop","cracking soil","hard crust","soil moisture low"],
    "nutrient": ["nitrogen","phosphorus","potassium","npk","fertilizer","fertiliser","chlorosis","chlorotic","interveinal","leaf color chart","lcc","low spad","older leaves yellowing","necrotic margin","micronutrient","deficiency","yellowing"],
    "pest": ["pest","aphid","whitefly","borer","hopper","weevil","caterpillar","larvae","thrips","mites","trap","sticky residue","honeydew","chewed","webbing","frass","insect"],
    "disease": ["blight","rust","mildew","smut","rot","leaf spot","necrosis","pathogen","fungal","bacterial","viral","lesion","mosaic","wilt disease","canker","powdery mildew","downy"],
    "heat": ["heatwave","hot","scorch","sunburn","thermal stress","high temperature","blistering","desiccation","sun scorch","leaf burn","heat stress"],
}
AG_CONTEXT = re.compile(r"\b(agri|agricultur|farm|farmer|field|crop|soil|irrigat|harvest|yield|paddy|rice|wheat|maize|corn|cotton|soy|orchard|greenhouse|seedling|fertiliz|manure|compost|pest|fung|blight|leaf|canopy|mulch|drip|sprinkler|nursery|plantation|horticul)\b", re.I)

def is_ag_context(s: str) -> bool:
    if not s: return False
    return bool(AG_CONTEXT.search(s))

def weak_labels(text: str) -> List[int]:
    t = (text or "").lower()
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

# ---------------------- BLIP captioner (optional) ----------------------
print("Initializing BLIP captioner...")
if ARGS.offline:
    blip_processor = None
    blip_model = None
    print("Offline mode: BLIP captioning disabled.")
else:
    try:
        blip_processor = BlipProcessor.from_pretrained(ARGS.hf_blip)
        blip_model = BlipForConditionalGeneration.from_pretrained(ARGS.hf_blip).to(DEVICE)
    except Exception as e:
        print("BLIP load failed; captioning disabled. Exception:", e)
        blip_processor = None
        blip_model = None

def caption_image(img_path: str, max_length:int=32) -> str:
    if blip_model is None or blip_processor is None:
        return ""
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = blip_processor(images=img, return_tensors="pt").to(DEVICE)
        out_ids = blip_model.generate(**inputs, max_length=max_length)
        cap = blip_processor.decode(out_ids[0], skip_special_tokens=True)
        return cap
    except Exception as e:
        return ""

# ---------------------- Image dataset downloads (TFDS or repo mirrors) ----------------------
def download_plant_village(data_dir:str, quick:bool=False) -> str:
    pv_dir = os.path.join(data_dir, "plant_village")
    ensure_dir(pv_dir)
    images_dir = ensure_dir(os.path.join(pv_dir, "images"))
    csv_path = os.path.join(pv_dir, "captions.csv")
    rows=[]
    # try TFDS
    if HAS_TFDS:
        try:
            print("Downloading PlantVillage via TFDS (may take time)...")
            ds = tfds.load("plant_village", split="train", shuffle_files=False, download=True)
            it = iter(tfds.as_numpy(ds))
            max_n = 2000 if not quick else 200
            cnt=0
            for ex in it:
                if cnt>=max_n: break
                img = Image.fromarray(ex["image"])
                fn = f"pv_{cnt:06d}.jpg"
                img.save(os.path.join(images_dir, fn))
                label = int(ex.get("label",-1))
                rows.append((fn, f"PlantVillage sample label {label}", label))
                cnt+=1
            pd.DataFrame(rows, columns=["filename","text","orig_label"]).to_csv(csv_path, index=False)
            print(f"Saved {cnt} PlantVillage images")
            return pv_dir
        except Exception as e:
            print("TFDS PlantVillage failed:", e)
    # fallback: try cloning mirror repo
    try:
        mirror_repo = os.path.join(pv_dir, "PlantVillage-Dataset")
        if not os.path.exists(mirror_repo):
            print("Cloning PlantVillage mirror (may be large)...")
            os.system(f"git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset {mirror_repo}")
        raw_dir = os.path.join(mirror_repo, "raw")
        if os.path.exists(raw_dir):
            copied=0
            for root,_,files in os.walk(raw_dir):
                for f in files:
                    if f.lower().endswith((".jpg",".png")):
                        src=os.path.join(root,f)
                        dst=os.path.join(images_dir,f"pv_m_{copied:06d}.jpg")
                        if not os.path.exists(dst):
                            shutil.copy(src,dst)
                            rows.append((os.path.basename(dst),"mirror import","mirror"))
                            copied+=1
                            if quick and copied>=200: break
                            if not quick and copied>=2000: break
                if quick and copied>=200: break
            pd.DataFrame(rows, columns=["filename","text","orig_label"]).to_csv(csv_path, index=False)
            print(f"Copied {copied} images from mirror")
            return pv_dir
    except Exception as e:
        print("PlantVillage mirror fallback failed:", e)
    raise RuntimeError("PlantVillage download failed. Provide dataset manually or enable TFDS.")

def download_deepweeds(data_dir:str, quick:bool=False) -> str:
    dw_dir = os.path.join(data_dir, "deepweeds")
    ensure_dir(dw_dir)
    images_dir = ensure_dir(os.path.join(dw_dir, "images"))
    csv_path = os.path.join(dw_dir, "captions.csv")
    rows=[]
    if HAS_TFDS:
        try:
            print("Downloading DeepWeeds via TFDS...")
            ds = tfds.load("deep_weeds", split="train", download=True)
            it = iter(tfds.as_numpy(ds))
            max_n = 1000 if not quick else 200
            cnt=0
            for ex in it:
                if cnt>=max_n: break
                img = Image.fromarray(ex["image"])
                fn = f"dw_{cnt:06d}.jpg"
                img.save(os.path.join(images_dir, fn))
                rows.append((fn,"DeepWeeds",int(ex.get("label",-1))))
                cnt+=1
            pd.DataFrame(rows, columns=["filename","text","orig_label"]).to_csv(csv_path, index=False)
            print(f"Saved {cnt} DeepWeeds images")
            return dw_dir
        except Exception as e:
            print("TFDS DeepWeeds failed:", e)
    # fallback: try repo clone
    try:
        repo_dir = os.path.join(dw_dir, "DeepWeeds")
        if not os.path.exists(repo_dir):
            os.system(f"git clone --depth 1 https://github.com/AlexOlsen/DeepWeeds {repo_dir}")
        src_images = os.path.join(repo_dir, "images")
        copied=0
        for root,_,files in os.walk(src_images):
            for f in files:
                if f.lower().endswith((".jpg",".png")):
                    src=os.path.join(root,f)
                    dst=os.path.join(images_dir,f"dw_m_{copied:06d}.jpg")
                    shutil.copy(src,dst)
                    rows.append((os.path.basename(dst),"mirror","mirror"))
                    copied+=1
                    if quick and copied>=200: break
                    if not quick and copied>=2000: break
            if quick and copied>=200: break
        pd.DataFrame(rows, columns=["filename","text","orig_label"]).to_csv(csv_path, index=False)
        print(f"Copied {copied} DeepWeeds images")
        return dw_dir
    except Exception as e:
        print("DeepWeeds fallback failed:", e)
    raise RuntimeError("DeepWeeds download failed. Provide dataset manually or enable TFDS.")

def download_plantdoc(data_dir:str, quick:bool=False) -> str:
    pd_dir = os.path.join(data_dir, "plantdoc")
    ensure_dir(pd_dir)
    images_dir = ensure_dir(os.path.join(pd_dir, "images"))
    csv_path = os.path.join(pd_dir, "captions.csv")
    rows=[]
    try:
        repo_dir = os.path.join(pd_dir, "PlantDoc-Dataset")
        if not os.path.exists(repo_dir):
            os.system(f"git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset {repo_dir}")
        copied=0
        for root,_,files in os.walk(repo_dir):
            for f in files:
                if f.lower().endswith((".jpg",".png")):
                    src=os.path.join(root,f)
                    dst=os.path.join(images_dir,f"pd_{copied:06d}.jpg")
                    shutil.copy(src,dst)
                    rows.append((os.path.basename(dst),"PlantDoc import","mirror"))
                    copied+=1
                    if quick and copied>=200: break
                    if not quick and copied>=2000: break
            if quick and copied>=200: break
        pd.DataFrame(rows, columns=["filename","text","orig_label"]).to_csv(csv_path, index=False)
        print(f"Copied {copied} PlantDoc images")
        return pd_dir
    except Exception as e:
        print("PlantDoc clone failed:", e)
    raise RuntimeError("PlantDoc download failed. Provide dataset manually.")

# ---------------------- Text dataset loaders (gardian, argilla, agnews, plus localmini) ----------------------
AGRI_RE = re.compile(r"\b(agri|agriculture|farm|farmer|farming|crop|soil|harvest|irrigat|pest|blight|drought|yield|wheat|rice|paddy|maize|soy|cotton|fertiliz|orchard|greenhouse|horticul)\b", re.I)
NON_AG_NOISE = re.compile(r"\b(NFL|NBA|MLB|NHL|tennis|golf|soccer|cricket|stocks?|Nasdaq|Dow Jones|earnings|IPO|merger|Hollywood|movie|music|concert)\b", re.I)

def build_gardian_stream(max_per:int=2000, quick:bool=False):
    if not HAS_DATASETS:
        raise RuntimeError("datasets library not available for Gardian.")
    print("Loading CGIAR/gardian-ai-ready-docs (streaming)...")
    ds = load_dataset("CGIAR/gardian-ai-ready-docs", split=None, streaming=True)
    texts=[]
    seen=0
    for key in ds.keys():
        for r in ds[key]:
            raw = (r.get("text") or r.get("content") or "").strip()
            if raw and len(raw)>40 and (_ascii_ratio(raw) >= 0.6) and AGRI_RE.search(raw) and not NON_AG_NOISE.search(raw):
                texts.append(raw)
                seen+=1
                if quick and seen>=200: break
                if not quick and seen>=max_per: break
        if quick and seen>=200: break
        if not quick and seen>=max_per: break
    print(f"Gardian: collected {len(texts)}")
    return texts

def build_argilla_stream(max_per:int=2000, quick:bool=False):
    if not HAS_DATASETS:
        raise RuntimeError("datasets library not available for Argilla.")
    print("Loading argilla/farming...")
    ds = load_dataset("argilla/farming", split="train")
    texts=[]
    seen=0
    for r in ds:
        q = str(r.get("evolved_questions","")).strip()
        a = str(r.get("domain_expert_answer","")).strip()
        raw = (q + " " + a).strip()
        if raw and _lang_ok(raw) and AGRI_RE.search(raw) and not NON_AG_NOISE.search(raw):
            texts.append(raw)
            seen+=1
            if quick and seen>=200: break
            if not quick and seen>=max_per: break
    print(f"Argilla: collected {len(texts)}")
    return texts

def build_agnews_agri(max_per:int=2000, quick:bool=False):
    if not HAS_DATASETS:
        raise RuntimeError("datasets library not available for AG News.")
    print("Loading ag_news and filtering for agriculture mentions...")
    ds = load_dataset("ag_news", split="train", streaming=True)
    texts=[]; seen=0
    for r in ds:
        raw = (r.get("text") or "").strip()
        if raw and AGRI_RE.search(raw) and not NON_AG_NOISE.search(raw) and _lang_ok(raw):
            texts.append(raw)
            seen+=1
            if quick and seen>=200: break
            if not quick and seen>=max_per: break
    print(f"AgNews agrified: {len(texts)}")
    return texts

# helper functions from your earlier reference
def _ascii_ratio(s:str)->float:
    if not s: return 0.0
    return sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
def _lang_ok(s:str)->bool: return _ascii_ratio(s) >= 0.6
def _norm(txt:str)->str: return re.sub(r"\s+", " ", txt).strip()

# ---------------------- Build synthetic localmini (text-only) ----------------------
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
    for k,lst in seeds.items():
        reps = n_per_nutrient if k=="nutrient_def" else n_per
        for _ in range(reps): out.append(random.choice(lst))
    random.shuffle(out)
    return out

def build_localmini(max_samples:int=0, mqtt_msgs:List[str]=None):
    texts = list(LOCAL_BASE) + make_balanced_local(300, 600)
    N = 2000
    if not mqtt_msgs: mqtt_msgs=[]
    for _ in range(N):
        sensor = simulate_sensor_summary()
        s = random.choice(TEMPLATES).format(
            symptom=random.choice(SYMPTOMS), crop=random.choice(CROPS),
            temp=round(np.clip(np.random.normal(32, 4), 15, 45),1),
            hum=int(np.clip(np.random.normal(55,15),15,95)),
            vpd=round(np.clip(np.random.normal(1.8,0.7),0.2,4.0),1),
            pest_count=int(np.clip(np.random.poisson(3),0,30)),
            sm=round(np.clip(np.random.normal(20,7),2,60),1),
            ph=round(np.clip(np.random.normal(6.5,0.6),4.5,8.5),1),
            weather=random.choice(WEATHERS),
        )
        mqtt = random.choice(mqtt_msgs) if mqtt_msgs and random.random()<0.4 else ""
        texts.append(fuse_text(sensor, s, mqtt))
    rows=[]
    for t in texts:
        labs = weak_labels(t)
        if labs: rows.append((_norm(t), labs))
    df = pd.DataFrame(rows, columns=["text","labels"])
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=SEED).reset_index(drop=True)
    return df

# ---------------------- Mix builder (text + image fusion) ----------------------
def build_mix(max_per_source:int=2000, quick:bool=False):
    # sources: gardian, argilla, agnews, localmini, plus image-derived captions
    pool=[]
    # text streams
    try:
        if HAS_DATASETS:
            print("Loading gardian (stream)...")
            garden_texts = build_gardian_stream(max_per=min(max_per_source,2000), quick=quick)
            pool.extend([("gardian", t) for t in garden_texts])
    except Exception as e:
        print("Gardian load skipped:", e)
    try:
        if HAS_DATASETS:
            print("Loading argilla...")
            arg_texts = build_argilla_stream(max_per=min(max_per_source,2000), quick=quick)
            pool.extend([("argilla", t) for t in arg_texts])
    except Exception as e:
        print("Argilla load skipped:", e)
    try:
        if HAS_DATASETS:
            print("Loading ag_news agrified...")
            ag_texts = build_agnews_agri(max_per=min(max_per_source,2000), quick=quick)
            pool.extend([("agnews", t) for t in ag_texts])
    except Exception as e:
        print("AgNews load skipped:", e)
    # localmini
    try:
        print("Building LocalMini synthetic texts...")
        lm = build_localmini(max_samples=1000)
        pool.extend([("localmini", t) for t in lm["text"].tolist()])
    except Exception as e:
        print("Localmini build skipped:", e)

    # image datasets: build captions and texts
    image_sources=[]
    try:
        pv_dir = download_plant_village(ARGS.data_dir, quick=quick)
        image_sources.append(("plant_village", pv_dir))
    except Exception as e:
        print("PlantVillage skipped:", e)
    try:
        dw_dir = download_deepweeds(ARGS.data_dir, quick=quick)
        image_sources.append(("deepweeds", dw_dir))
    except Exception as e:
        print("DeepWeeds skipped:", e)
    try:
        pd_dir = download_plantdoc(ARGS.data_dir, quick=quick)
        image_sources.append(("plantdoc", pd_dir))
    except Exception as e:
        print("PlantDoc skipped:", e)

    # process image folders: create caption + sensor fused text and add to pool
    for name, base in image_sources:
        caps_csv = os.path.join(base, "captions.csv")
        imgs_dir = os.path.join(base, "images")
        if not os.path.exists(imgs_dir):
            continue
        # if captions.csv exists, read; else build from files
        rows=[]
        if os.path.exists(caps_csv):
            dfc = pd.read_csv(caps_csv)
            for idx, r in dfc.iterrows():
                fn = r["filename"]
                imgp = os.path.join(imgs_dir, fn)
                if not os.path.exists(imgp): continue
                # caption using BLIP
                caption = caption_image(imgp) if blip_model else r.get("text","")
                sensor = simulate_sensor_summary()
                text = fuse_text(sensor, caption or r.get("text",""))
                # map label heuristics per dataset
                if name=="plant_village":
                    # treat non-zero orig_label as disease
                    orig = r.get("orig_label", None)
                    labs = []
                    try:
                        if int(orig) != 0: labs=[LABEL_TO_ID["disease_risk"]]
                    except Exception:
                        labs=[LABEL_TO_ID["disease_risk"]]
                elif name=="deepweeds":
                    labs=[LABEL_TO_ID["pest_risk"]]
                else:
                    labs=[LABEL_TO_ID["disease_risk"]]
                pool.append((name, text))
        else:
            # fallback: iterate files
            cnt=0
            for f in os.listdir(imgs_dir):
                if not f.lower().endswith((".jpg",".png",".jpeg")): continue
                imgp = os.path.join(imgs_dir, f)
                caption = caption_image(imgp) if blip_model else ""
                sensor = simulate_sensor_summary()
                text = fuse_text(sensor, caption or f)
                pool.append((name, text))
                cnt+=1
                if quick and cnt>=200: break

    # dedup & weak-label
    seen=set(); rows=[]
    for src, txt in pool:
        h = hashlib.sha1(_norm(txt).encode("utf-8")).hexdigest()
        if h in seen: continue
        seen.add(h)
        labs = weak_labels(txt)
        if labs:
            rows.append((src, _norm(txt), labs))
    df = pd.DataFrame(rows, columns=["source","text","labels"])
    print("[Mix] final size:", len(df))
    if len(df)==0:
        raise RuntimeError("Empty final corpus. Provide datasets or disable offline.")
    return df[["text","labels"]]

# ---------------------- Model building (CLIP + LoRA) ----------------------
def build_clip_tokenizer_and_model(model_name:str, freeze_base:bool=True):
    clip = CLIPModel.from_pretrained(model_name)
    proc = CLIPProcessor.from_pretrained(model_name)
    if freeze_base:
        for p in clip.parameters(): p.requires_grad=False
    return clip, proc

def infer_lora_targets(model):
    names = [n for n,_ in model.named_modules()]
    cands = ["q_proj","k_proj","v_proj","out_proj","proj","dense","to_q","to_k","to_v"]
    found=[]
    for c in cands:
        if any(c in n for n in names):
            found.append(c)
    if not found: found=["proj"]
    return found

class MultimodalClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_labels=NUM_LABELS):
        super().__init__()
        self.clip = clip_model
        text_dim = getattr(clip_model.config, "projection_dim", None) or getattr(clip_model.text_model.config, "hidden_size", 512)
        vision_dim = getattr(clip_model.config, "projection_dim", None) or getattr(clip_model.vision_model.config, "hidden_size", 512)
        hidden = max(256, (text_dim + vision_dim)//4)
        self.head = torch.nn.Sequential(torch.nn.Linear(text_dim+vision_dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(hidden, num_labels))
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, return_dict=True):
        out = self.clip(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_embeds = getattr(out, "text_embeds", None)
        image_embeds = getattr(out, "image_embeds", None)
        if text_embeds is None:
            text_embeds = out.text_model_output[0][:,0,:]
        if image_embeds is None:
            image_embeds = out.image_model_output[0][:,0,:]
        feats = torch.cat([text_embeds, image_embeds], dim=-1)
        logits = self.head(feats)
        class R: pass
        r = R(); r.logits = logits
        return r

def build_model(num_labels:int, clip_name:str, freeze_base:bool=True):
    clip, processor = build_clip_tokenizer_and_model(clip_name, freeze_base=freeze_base)
    mm = MultimodalClassifier(clip, num_labels=num_labels)
    targets = infer_lora_targets(clip)
    lcfg = LoraConfig(r=ARGS.lora_r, lora_alpha=ARGS.lora_alpha, lora_dropout=ARGS.lora_dropout, bias="none", task_type="SEQ_CLS", target_modules=targets)
    mm = get_peft_model(mm, lcfg)
    print("[LoRA] target modules:", targets)
    return mm, processor

# ---------------------- Dataset class ----------------------
from torchvision import transforms as T
IMG_TRANSFORM = T.Compose([T.Resize((224,224), interpolation=InterpolationMode.BICUBIC), T.ToTensor()])

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame, processor, max_len:int=160):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        fn = r.get("filename", None)
        text = r["text"]
        if fn and os.path.exists(fn):
            img = Image.open(fn).convert("RGB")
        else:
            # placeholder blank image
            img = Image.new("RGB",(224,224),(128,128,128))
        return {"img": img, "text": text, "labels": torch.tensor([1.0 if i in r["labels"] else 0.0 for i in range(NUM_LABELS)], dtype=torch.float32), "fn": fn or ""}
    @staticmethod
    def collate_fn(batch, processor, max_len=160):
        texts = [b["text"] for b in batch]
        images = [b["img"] for b in batch]
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "pixel_values": enc["pixel_values"], "labels": labels, "raw_texts": texts}

# ---------------------- Loss & metrics ----------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.5, label_smoothing=0.02):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.smooth = label_smoothing
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p*targets + (1-p)*(1-targets)
        loss = ((1-pt)**self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha.view(1,-1)
        return loss.mean()

def make_weights_for_balanced_classes(df:pd.DataFrame):
    counts = np.zeros(NUM_LABELS)
    for labs in df["labels"]:
        for k in labs: counts[k]+=1
    inv = 1.0 / np.maximum(counts, 1)
    inst_w=[]
    for labs in df["labels"]:
        w = np.mean([inv[k] for k in labs]) if labs else np.mean(inv)
        inst_w.append(w)
    inst_w = np.array(inst_w, dtype=np.float32)
    inst_w = inst_w / (inst_w.mean() + 1e-12)
    return inst_w, counts

# ---------------------- Training helpers ----------------------
def train_local(model:torch.nn.Module, processor, tr_df:pd.DataFrame, va_df:pd.DataFrame, class_alpha:torch.Tensor):
    tr_ds = ImageTextDataset(tr_df, processor, max_len=ARGS.local_epochs*10)
    va_ds = ImageTextDataset(va_df, processor, max_len=ARGS.local_epochs*10)
    inst_w, _ = make_weights_for_balanced_classes(tr_df)
    sampler = torch.utils.data.WeightedRandomSampler(weights=torch.tensor(inst_w, dtype=torch.double), num_samples=max(len(tr_df), ARGS.batch_size), replacement=True)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=ARGS.batch_size, sampler=sampler, collate_fn=lambda b: ImageTextDataset.collate_fn(b, processor), num_workers=0)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=max(ARGS.batch_size,8), shuffle=False, collate_fn=lambda b: ImageTextDataset.collate_fn(b, processor), num_workers=0)
    device = DEVICE
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=ARGS.lr, weight_decay=0.01)
    steps = max(1, math.ceil(len(tr_loader)))
    total_steps = ARGS.local_epochs * steps
    from transformers import get_linear_schedule_with_warmup
    sch = get_linear_schedule_with_warmup(opt, max(1,int(0.1*total_steps)), total_steps)
    loss_fn = FocalLoss(alpha=class_alpha.to(device), gamma=2.0, label_smoothing=0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available()))
    model.train().to(device)
    non_frozen = [p for p in model.parameters() if p.requires_grad]
    ema_params = [p.detach().clone() for p in non_frozen]
    for ep in range(ARGS.local_epochs):
        for batch in tr_loader:
            inputs = {k:v.to(device) for k,v in batch.items() if k in ("input_ids","attention_mask","pixel_values")}
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**inputs)
                logits = out.logits
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sch.step()
            for epm, mp in zip(ema_params, non_frozen):
                epm.data.mul_(0.997).add_(mp.data, alpha=1.0-0.997)
    # Eval with EMA
    backup = [p.detach().clone() for p in non_frozen]
    for p, epm in zip(non_frozen, ema_params): p.data.copy_(epm.data)
    thr = calibrate_thresholds(model, va_loader, precision_target=0.90)
    was_quiet = ARGS.quiet_eval
    ARGS.quiet_eval = True
    mets = evaluate_with_thr(model, va_loader, thr)
    ARGS.quiet_eval = was_quiet
    micro, macro = mets["micro_f1"], mets["macro_f1"]
    for p, bkp in zip(non_frozen, backup): p.data.copy_(bkp.data)
    lora_sd = get_peft_model_state_dict(model)
    lora_sd = {k:v.detach().cpu() for k,v in lora_sd.items()}
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return micro, macro, lora_sd, thr, len(tr_df)

def fedavg_weighted(states:List[Dict[str,torch.Tensor]], sizes:List[int]):
    total = float(sum(sizes)); ws=[s/total for s in sizes]
    keys = list(states[0].keys()); out={}
    for k in keys:
        out[k] = torch.stack([st[k].float()*w for st,w in zip(states, ws)], dim=0).sum(0)
    return out

# ---------------------- Calibration & eval ----------------------
def calibrate_thresholds(model, loader, precision_target=0.90):
    model.eval().to(DEVICE)
    P_list=[]; T_list=[]
    with torch.no_grad():
        for b in loader:
            bt = {k:v.to(DEVICE) for k,v in b.items() if k in ("input_ids","attention_mask","pixel_values")}
            out = model(**bt)
            logits = out.logits
            P_list.append(torch.sigmoid(logits).cpu().numpy())
            T_list.append(b["labels"].numpy())
    P = np.vstack(P_list); T = np.vstack(T_list)
    C = P.shape[1]; thr = np.zeros(C, dtype=np.float32)
    for j in range(C):
        col, y = P[:,j], T[:,j].astype(int)
        best_t_f1, best_f1 = 0.5, -1.0
        best_t_prec = None
        for t in np.linspace(0.05,0.9,35):
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

def evaluate_with_thr(model, loader, thr):
    def _cap(x): return min(0.999, float(x))
    model.eval().to(DEVICE)
    P_all=[]; T_all=[]; R_all=[]
    with torch.no_grad():
        for b in loader:
            bt = {k:v.to(DEVICE) for k,v in b.items() if k in ("input_ids","attention_mask","pixel_values")}
            out = model(**bt)
            logits = out.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= thr).astype(int)
            P_all.append(preds); T_all.append(b["labels"].numpy()); R_all.append(probs)
    P = np.vstack(P_all); T = np.vstack(T_all); R = np.vstack(R_all)
    micro = f1_score(T, P, average="micro", zero_division=0)
    macro = f1_score(T, P, average="macro", zero_division=0)
    prec = precision_score(T, P, average=None, zero_division=0)
    rec = recall_score(T, P, average=None, zero_division=0)
    f1s = [f1_score(T[:,i], P[:,i], zero_division=0) for i in range(NUM_LABELS)]
    supports = T.sum(axis=0)
    if not ARGS.quiet_eval:
        print("\nPer-class metrics:")
        for i, lab in enumerate(ISSUE_LABELS):
            if supports[i] < 10:
                print(f" - {lab:14s} | insufficient support (n={int(supports[i])})")
                continue
            print(f" - {lab:14s} | P={_cap(prec[i]):.3f} R={_cap(rec[i]):.3f} F1={_cap(f1s[i]):.3f} thr={thr[i]:.2f}")
        print(f"\nOverall: micro-F1={_cap(micro):.3f}  macro-F1={_cap(macro):.3f}")
        try:
            pr_micro = average_precision_score(T, R, average="micro")
            pr_macro = average_precision_score(T, R, average="macro")
            roc_micro = roc_auc_score(T, R, average="micro")
            roc_macro = roc_auc_score(T, R, average="macro")
            print(f"AUPRC micro={_cap(pr_micro):.3f} macro={_cap(pr_macro):.3f} | AUROC micro={_cap(roc_micro):.3f} macro={_cap(roc_macro):.3f}")
        except Exception:
            pass
    return {"micro_f1": micro, "macro_f1": macro, "per_class": {"precision": prec, "recall": rec, "f1": np.array(f1s)}}

# ---------------------- Build merged corpus and run federated training ----------------------
def build_and_train_all():
    # Build text+image mix
    print("Building mixed multimodal corpus...")
    mix_df = build_mix(max_per_source=ARGS.local_epochs*1000, quick=ARGS.quick)
    # Ensure labels are lists of ints
    mix_df["labels"] = mix_df["labels"].apply(lambda x: x if isinstance(x,list) else list(x))
    print(f"Total corpus size: {len(mix_df)}")
    # Federated split
    clients_all = split_clients(mix_df, max(1, ARGS.clients), ARGS.dirichlet_alpha)
    val_k = max(1, int(0.12 * len(clients_all)))
    val_df = pd.concat(clients_all[:val_k], ignore_index=True) if val_k>0 else pd.DataFrame(columns=mix_df.columns)
    train_clients = clients_all[val_k:]
    train_df = pd.concat(train_clients, ignore_index=True)
    train_df, test_df = train_test_split(train_df, test_size=0.15, random_state=SEED, shuffle=True)
    # class alpha
    _, counts = make_weights_for_balanced_classes(train_df)
    inv = 1.0 / np.maximum(counts, 1)
    class_alpha = (inv / inv.mean()).astype(np.float32)
    class_alpha = torch.tensor(class_alpha)
    # build global model
    global_model, processor = build_model(NUM_LABELS, clip_name=ARGS.hf_clip, freeze_base=True)
    global_model.to(DEVICE)
    # unseen-client validation loader
    val_loader = torch.utils.data.DataLoader(ImageTextDataset(val_df, processor), batch_size=32, shuffle=False, collate_fn=lambda b: ImageTextDataset.collate_fn(b, processor))
    thr_history=[]
    metrics_dir = os.path.join(ARGS.save_dir, "metrics"); ensure_dir(metrics_dir)
    for r in range(1, max(1, ARGS.rounds)+1):
        print(f"\n==== Round {r}/{ARGS.rounds} ====")
        rng = np.random.default_rng(SEED + r)
        k_all = list(range(len(train_clients)))
        rng.shuffle(k_all)
        m = max(1, int(ARGS.participation * len(k_all)))
        chosen = k_all[:m]
        states=[]; sizes=[]
        for i in chosen:
            if rng.random() < ARGS.client_dropout:
                print(f"[Client {i+1}] dropped")
                continue
            cdf = train_clients[i]
            if len(cdf) < 40:
                print(f"[Client {i+1}] skipped (tiny shard n={len(cdf)})")
                continue
            # local split
            n = len(cdf); val_n = max(1, int(0.12 * n))
            va_df, tr_df = cdf.iloc[:val_n], cdf.iloc[val_n:]
            local_model, _ = build_model(NUM_LABELS, clip_name=ARGS.hf_clip, freeze_base=True)
            set_peft_model_state_dict(local_model, get_peft_model_state_dict(global_model))
            local_model.to(DEVICE)
            local_epochs = int(rng.choice([2,3,4], p=[0.4,0.4,0.2]))
            orig_local = ARGS.local_epochs; ARGS.local_epochs = local_epochs
            micro, macro, lora_sd, thr_local, used_n = train_local(local_model, processor, tr_df.reset_index(drop=True), va_df.reset_index(drop=True), class_alpha=class_alpha)
            ARGS.local_epochs = orig_local
            print(f"[Client {i+1}] micro_f1={micro:.3f} macro_f1={macro:.3f} (n={used_n}) thr={np.round(thr_local,2)}")
            states.append(lora_sd); sizes.append(used_n)
            del local_model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if states:
            print("Aggregating client adapters via weighted FedAvg...")
            aggregated = fedavg_weighted(states, sizes)
            set_peft_model_state_dict(global_model, aggregated)
            final_thr = calibrate_thresholds(global_model, val_loader, precision_target=0.90)
            final_thr = np.clip(final_thr + np.array([+0.03, 0.00, 0.00, +0.02, 0.00]), 0.05, 0.90)
        else:
            print("No client updates this round; using previous thresholds or default.")
            final_thr = thr_history[-1] if thr_history else np.array([0.5]*NUM_LABELS)
        thr_history.append(final_thr)
        # evaluate on global test holdout
        test_loader = torch.utils.data.DataLoader(ImageTextDataset(test_df, processor), batch_size=32, shuffle=False, collate_fn=lambda b: ImageTextDataset.collate_fn(b, processor))
        test_mets = evaluate_with_thr(global_model, test_loader, final_thr)
        with open(os.path.join(metrics_dir, f"round_{r:02d}_summary.json"), "w") as f:
            json.dump({"round": r, "micro_f1": float(test_mets["micro_f1"]), "macro_f1": float(test_mets["macro_f1"])}, f, indent=2)
        np.save(os.path.join(metrics_dir, f"round_{r:02d}_thr.npy"), final_thr)
    # save adapters and thresholds
    ap = os.path.join(ARGS.save_dir, "global_lora.pt")
    thp = os.path.join(ARGS.save_dir, "thresholds.npy")
    torch.save(get_peft_model_state_dict(global_model), ap)
    np.save(thp, thr_history[-1] if thr_history else np.array([0.5]*NUM_LABELS))
    print(f"[Saved] adapters -> {ap}")
    print(f"[Saved] thresholds -> {thp}")

# ---------------------- Helper: split_clients (Dirichlet) ----------------------
def split_clients(df:pd.DataFrame, n:int, alpha:float) -> List[pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    prim=[]
    for labs in df["labels"]:
        if labs: prim.append(int(rng.choice(labs)))
        else: prim.append(int(rng.integers(0, NUM_LABELS)))
    df2 = df.copy(); df2["_y"] = prim
    class_client_probs = rng.dirichlet([alpha]*n, size=NUM_LABELS)
    client_bins=[[] for _ in range(n)]
    for idx, y in enumerate(df2["_y"].tolist()):
        k = int(rng.choice(n, p=class_client_probs[y]))
        client_bins[k].append(idx)
    out=[]
    for k in range(n):
        part = df2.iloc[client_bins[k]].drop(columns=["_y"]).reset_index(drop=True)
        out.append(part)
    return out

# ---------------------- Main ----------------------
if __name__ == "__main__":
    print("Starting pipeline. Quick mode:", ARGS.quick, "Offline:", ARGS.offline)
    try:
        build_and_train_all()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Pipeline failed:", e)
        sys.exit(1)
