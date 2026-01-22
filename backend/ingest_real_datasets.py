"""
Ingest real datasets (Kaggle + HuggingFace) and upsert into Qdrant `knowledge_base`.

Usage examples:

# Minimal (requires kaggle CLI and HF access tokens set up):
python backend/ingest_real_datasets.py --kaggle datasetengineer/crop-health-and-environmental-stress-dataset --kaggle ashishpatelresearch/maize-plant-leaf-nutrient-deficiency-dataset --kaggle gauravduttakiit/agricultural-pests-dataset --hf Hemant-Soni/PlantDoc --out data/ingested --qdrant-url :memory: --max-files 100

Notes:
- Kaggle CLI must be configured (KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json in ~/.kaggle)
- HuggingFace `datasets` access may require HF_TOKEN if private
- Installs optional dependencies when necessary
"""

from __future__ import annotations
import os
import sys
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path
import json
from typing import List
import random

# Local project imports
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Attempt to import helpers from backend.qdrant_rag
try:
    from backend.qdrant_rag import init_qdrant_collections, ingest_images_from_dir, Embedders
except Exception:
    init_qdrant_collections = None
    ingest_images_from_dir = None
    Embedders = None

# Recommended default datasets for Qdrant ingestion
DEFAULT_KAGGLE_DATASETS = [
    'datasetengineer/crop-health-and-environmental-stress-dataset',
    'gauravduttakiit/agricultural-pests-dataset',
    'nirmalsankalana/plantdoc-dataset',
    'zoya77/agricultural-water-stress-image-dataset'
]

DEFAULT_HF_DATASETS = [
    'persadian/CropSeek-LLM',
    'sikeaditya/AgriAssist_LLM',
    'CropNet/CropNet'
]


def ensure_package(pkg: str):
    try:
        __import__(pkg)
    except Exception:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def download_kaggle_dataset(dataset_id: str, dest: str, max_files: int = None):
    """Download a Kaggle dataset using kaggle CLI into dest and return dest path."""
    # Ensure kaggle CLI available
    ensure_package('kaggle')
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading Kaggle dataset {dataset_id} into {dest}")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_id, "-p", dest, "--unzip"]
    subprocess.check_call(cmd)

    # Optionally limit files by copying a subset
    if max_files is not None:
        files = []
        for root, _, fns in os.walk(dest):
            for fn in fns:
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    files.append(os.path.join(root, fn))
        if len(files) > max_files:
            print(f"Limiting to {max_files} images for ingestion (found {len(files)})")
            keep = set(files[:max_files])
            tmpdir = Path(dest) / 'subset'
            tmpdir.mkdir(exist_ok=True)
            for p in keep:
                shutil.copy(p, tmpdir / os.path.basename(p))
            return str(tmpdir)
    return dest


def download_hf_dataset(hf_dataset_id: str, dest: str, max_entries: int = None):
    """Download a HuggingFace dataset (text) and write a simple TSV/metadata for ingestion."""
    ensure_package('datasets')
    from datasets import load_dataset
    os.makedirs(dest, exist_ok=True)
    print(f"Loading HuggingFace dataset {hf_dataset_id} (may take time)")
    ds = load_dataset(hf_dataset_id)
    # Pick the first split available
    split = list(ds.keys())[0]
    out_dir = Path(dest) / Path(hf_dataset_id.replace('/', '_'))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    for i, ex in enumerate(ds[split]):
        if max_entries is not None and rows >= max_entries:
            break
        # Save a textual record per example (useful for sentence-transformers ingestion)
        txt = None
        if isinstance(ex, dict):
            # heuristics to find text fields
            for k in ('text', 'description', 'summary', 'caption'):
                if k in ex and isinstance(ex[k], str):
                    txt = ex[k]
                    break
            if txt is None:
                # fallback to JSON dump
                txt = json.dumps(ex)
        else:
            txt = str(ex)
        fn = out_dir / f"rec_{i}.txt"
        fn.write_text(txt, encoding='utf-8')
        rows += 1
    print(f"Wrote {rows} textual records to {out_dir}")
    return str(out_dir)


def ingest_dirs_to_qdrant(client: QdrantClient, dirs: List[str], max_files_per_dir: int = 500, batch_size: int = 32):
    """Use backend.qdrant_rag.Embedders and ingest_images_from_dir when available; otherwise do a minimal embed+upsert flow."""
    # Create collection with named vectors
    try:
        client.recreate_collection(
            collection_name='knowledge_base',
            vectors_config={
                'visual': rest.VectorParams(size=512, distance=rest.Distance.COSINE),
                'semantic': rest.VectorParams(size=384, distance=rest.Distance.COSINE),
            }
        )
    except Exception:
        # fallback
        try:
            client.create_collection(collection_name='knowledge_base', vectors_config={'semantic': rest.VectorParams(size=384, distance=rest.Distance.COSINE)})
        except Exception:
            pass

    if Embedders is not None and ingest_images_from_dir is not None:
        emb = Embedders()
        for d in dirs:
            if os.path.exists(d):
                print(f"Ingesting images from {d} via backend.ingest_images_from_dir")
                ingest_images_from_dir(client, d, emb=emb, source=os.path.basename(d), max_files=max_files_per_dir)
            else:
                print(f"Directory {d} does not exist, skipping")
    else:
        # minimal fallback: use CLIP + sbert directly
        ensure_package('transformers')
        ensure_package('sentence-transformers')
        from transformers import CLIPProcessor, CLIPModel
        from sentence_transformers import SentenceTransformer
        from PIL import Image
        import numpy as np

        clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cpu')
        proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        sbert = SentenceTransformer('all-MiniLM-L6-v2')

        for d in dirs:
            if not os.path.exists(d):
                print(f"Skipping missing dir: {d}")
                continue
            imgs = []
            metas = []
            for root, _, fns in os.walk(d):
                for fn in fns:
                    if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                        imgs.append(os.path.join(root, fn))
                        metas.append({'source': os.path.basename(d), 'filename': fn})
                        if len(imgs) >= max_files_per_dir:
                            break
                if len(imgs) >= max_files_per_dir:
                    break

            print(f"Embedding {len(imgs)} images from {d}")
            # batch embed and upsert
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size]
                pil = [Image.open(p).convert('RGB') for p in batch]
                inputs = proc(images=pil, return_tensors='pt')
                with torch.no_grad():
                    feats = clip.get_image_features(**inputs).numpy()
                sems = sbert.encode([m['filename'] for m in metas[i:i+len(batch)]], convert_to_numpy=True)
                vectors = []
                points = []
                for j, pfile in enumerate(batch):
                    pid = random.getrandbits(63)
                    vis = feats[j].tolist()
                    sem = sems[j].tolist()
                    pts = rest.PointStruct(id=pid, vector={'visual': vis, 'semantic': sem}, payload={'filename': os.path.basename(pfile), 'source': os.path.basename(d)})
                    points.append(pts)
                client.upsert(collection_name='knowledge_base', points=points)


def ingest_texts_to_qdrant(client: QdrantClient, text_dirs: List[str], emb: Optional[Embedders] = None, batch_size: int = 32):
    """Embed text files and upsert as semantic vectors into `knowledge_base`."""
    if emb is None:
        if Embedders is not None:
            emb = Embedders()
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception:
                raise ImportError('sentence-transformers required for text embedding fallback')
            emb = SentenceTransformer('all-MiniLM-L6-v2')

    for d in text_dirs:
        if not os.path.exists(d):
            print(f"Skipping missing text dir: {d}")
            continue
        txt_files = [os.path.join(d, fn) for fn in os.listdir(d) if fn.lower().endswith('.txt')]
        print(f"Embedding {len(txt_files)} text files from {d}")
        batch_texts, metas = [], []
        for i, fp in enumerate(txt_files):
            txt = Path(fp).read_text(encoding='utf-8')
            batch_texts.append(txt)
            metas.append({'source': os.path.basename(d), 'filename': os.path.basename(fp)})
            if len(batch_texts) >= batch_size or i == len(txt_files) - 1:
                # embed
                if hasattr(emb, 'embed_text'):
                    vecs = [emb.embed_text(t) for t in batch_texts]
                else:
                    vecs = emb.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                    vecs = [v.tolist() for v in vecs]

                points = []
                for j, v in enumerate(vecs):
                    pid = random.getrandbits(63)
                    points.append(rest.PointStruct(id=pid, vector={'semantic': v}, payload={'text': batch_texts[j], 'source': metas[j]['source'], 'filename': metas[j]['filename']}))
                client.upsert(collection_name='knowledge_base', points=points)
                batch_texts, metas = [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='append', help='Kaggle dataset ID (e.g. owner/dataset-name)')
    parser.add_argument('--hf', action='append', help='HuggingFace dataset ID (text), e.g. Hemant-Soni/PlantDoc')
    parser.add_argument('--out', type=str, default='data/ingested', help='Output directory for downloaded datasets')
    parser.add_argument('--qdrant-url', type=str, default=':memory:', help='Qdrant client URL or :memory: for local in-memory instance')
    parser.add_argument('--max-files', type=int, default=500, help='Max images per dataset to ingest')
    parser.add_argument('--max-text', type=int, default=1000, help='Max text records per HF dataset')
    parser.add_argument('--use-defaults', action='store_true', help='Use recommended default datasets for Kaggle and HF')

    args = parser.parse_args()

    # Apply defaults if requested
    if args.use_defaults:
        if not args.kaggle:
            args.kaggle = DEFAULT_KAGGLE_DATASETS.copy()
        if not args.hf:
            args.hf = DEFAULT_HF_DATASETS.copy()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(args.qdrant_url)

    # Download Kaggle datasets
    downloaded_dirs = []
    if args.kaggle:
        for k in args.kaggle:
            try:
                tgt = out / Path(k.replace('/', '_'))
                tgt.mkdir(parents=True, exist_ok=True)
                ddir = download_kaggle_dataset(k, str(tgt), max_files=args.max_files)
                downloaded_dirs.append(ddir)
            except Exception as e:
                print(f"Kaggle download failed for {k}: {e}")

    # Download HF datasets (text records)
    if args.hf:
        hf_dirs = []
        for h in args.hf:
            try:
                tgt = out / Path('hf') / Path(h.replace('/', '_'))
                tgt.mkdir(parents=True, exist_ok=True)
                dd = download_hf_dataset(h, str(tgt), max_entries=args.max_text)
                hf_dirs.append(dd)
            except Exception as e:
                print(f"HF dataset download failed for {h}: {e}")
        # Ingest textual records into Qdrant as semantic vectors
        if hf_dirs:
            print("Ingesting HF textual datasets into Qdrant semantic vectors...")
            ingest_texts_to_qdrant(client, hf_dirs)

    # Ingest downloaded image dirs into Qdrant
    ingest_dirs = [d for d in downloaded_dirs if os.path.exists(d)]
    if ingest_dirs:
        ingest_dirs_to_qdrant(client, ingest_dirs, max_files_per_dir=args.max_files)
    else:
        print("No image directories available for ingestion.")

    # Demo hybrid retrieval: pick a random ingested image and perform visual search + semantic lookup
    try:
        from PIL import Image
        emb = Embedders() if Embedders is not None else None
        sample_img = None
        for d in ingest_dirs:
            for root, _, fns in os.walk(d):
                for fn in fns:
                    if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                        sample_img = os.path.join(root, fn); break
                if sample_img: break
            if sample_img: break
        if sample_img and emb is not None:
            print('Running demo hybrid retrieval on:', sample_img)
            img = Image.open(sample_img).convert('RGB')
            vis = emb.embed_image(img)
            res = client.query_points(collection_name='knowledge_base', query=vis, using='visual', limit=3).points
            for r in res:
                print('ID:', r.id, 'score:', getattr(r, 'score', None), 'payload_preview:', {k: r.payload.get(k) for k in ('filename','stress_type','text','source')})
        else:
            print('No demo retrieval performed (missing Embedders or no ingested images)')
    except Exception as e:
        print('Demo retrieval failed:', e)

    print("Ingestion finished. You can query Qdrant via qdrant-client or backend.qdrant_rag.agentic_diagnose()")


if __name__ == '__main__':
    main()
