"""
Qdrant-backed Multimodal RAG utilities for FarmFederate.

Provides:
- init_qdrant_collections: create `crop_health_knowledge` with named vectors: visual (512) and semantic (384)
- ingest_dataset: functions to generate CLIP visual embeddings and semantic embeddings (SentenceTransformers)
  and upsert points into Qdrant (payload includes stress_type, crop_name, severity, source, filename, agronomist_notes)
- agentic_diagnose: given a query image and optional user description, performs visual search, retrieves top-k
  cases, constructs a grounding prompt and returns a treatment plan via a pluggable LLM function.
- session memory helpers: create/append to `farm_session_memory` collection with timestamps and feedback

Notes:
- Requires: qdrant-client, transformers, sentence-transformers, torch, pillow
- This module deliberately keeps LLM/raptor-mini call pluggable: implement `call_llm(prompt, **kwargs)` to integrate Raptor Mini API.

Usage (quick):
from qdrant_rag import init_qdrant_collections, ingest_images_from_dir, agentic_diagnose
client = init_qdrant_collections(url='http://localhost:6333')
ingest_images_from_dir(client, '/data/plantvillage', source='PlantVillage')
plan = agentic_diagnose(client, image_path='example.jpg', user_description='yellowing leaves and spots')
print(plan['treatment'])
"""

from __future__ import annotations
import os
import time
import json
import math
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

# qdrant imports are optional at module import time; guard them in case the package is not installed.
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except Exception:
    QdrantClient = None
    rest = None

from PIL import Image
import numpy as np
import torch
# Transformers / sentence-transformers imports are lazily loaded by `Embedders` to avoid import-time failures
CLIPProcessor = None
CLIPModel = None
SentenceTransformer = None

# Constants
KNOWLEDGE_COLLECTION = "crop_health_knowledge"
SESSION_COLLECTION = "farm_session_memory"

# Default models
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # visual encoder -> 512-d
TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # semantic -> 384-d


def init_qdrant_collections(client: QdrantClient, recreate: bool = False) -> None:
    """Initialize the two collections used by the pipeline.

    - crop_health_knowledge: named vectors `visual` (512, cosine) and `semantic` (384, cosine)
    - farm_session_memory: single `semantic` vector (384, cosine) for session retrieval
    """
    if QdrantClient is None or rest is None:
        raise ImportError("qdrant-client is required to initialize collections. Install with: pip install qdrant-client")

    # Knowledge collection with named vectors
    if recreate and client.get_collection(KNOWLEDGE_COLLECTION, check_response=False) is not None:
        try:
            client.delete_collection(KNOWLEDGE_COLLECTION)
        except Exception:
            pass

    try:
        client.recreate_collection(
            collection_name=KNOWLEDGE_COLLECTION,
            vectors_config={
                "visual": rest.VectorParams(size=512, distance=rest.Distance.COSINE),
                "semantic": rest.VectorParams(size=384, distance=rest.Distance.COSINE),
            },
        )
    except Exception:
        # Fallback for older qdrant-client versions
        try:
            client.create_collection(
                collection_name=KNOWLEDGE_COLLECTION,
                vectors_config={
                    "visual": rest.VectorParams(size=512, distance=rest.Distance.COSINE),
                    "semantic": rest.VectorParams(size=384, distance=rest.Distance.COSINE),
                },
            )
        except Exception as e:
            # If collection exists, ignore
            if 'already exists' in str(e).lower():
                pass
            else:
                raise

    # Ensure payload index exists for common keys
    try:
        client.upsert_payload_filter(
            collection_name=KNOWLEDGE_COLLECTION,
            payload_filter={"must": []},
        )
    except Exception:
        # If API differs, ignore
        pass

    # Session memory collection
    if recreate and client.get_collection(SESSION_COLLECTION, check_response=False) is not None:
        try:
            client.delete_collection(SESSION_COLLECTION)
        except Exception:
            pass
    try:
        client.create_collection(
            collection_name=SESSION_COLLECTION,
            vectors_config={"semantic": rest.VectorParams(size=384, distance=rest.Distance.COSINE)},
        )
    except Exception:
        # already exists or other
        pass


class Embedders:
    """Wrapper for models used for generating embeddings.

    Lazy-loads heavy ML libraries (transformers, sentence-transformers) on demand so the
    module can be imported in environments where these packages are not installed. If you
    call the constructor without the required packages, a clear ImportError will be raised.
    """

    def __init__(self, device: Optional[str] = None, clip_model: str = CLIP_MODEL_NAME, text_model: str = TEXT_EMBED_MODEL):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Lazy imports to avoid module import-time failures in test envs
        try:
            from transformers import CLIPProcessor as _CLIPProcessor, CLIPModel as _CLIPModel
            from sentence_transformers import SentenceTransformer as _SentenceTransformer
        except Exception as e:
            raise ImportError(
                "Embedders requires `transformers` and `sentence-transformers` to be installed. "
                "Install them with: pip install -r backend/requirements-qdrant.txt"
            ) from e

        self.clip = _CLIPModel.from_pretrained(clip_model).to(self.device)
        self.processor = _CLIPProcessor.from_pretrained(clip_model)
        self.text_encoder = _SentenceTransformer(text_model, device=self.device)

    def embed_image(self, image: Image.Image) -> List[float]:
        inputs = self.processor(images=image, return_tensors="pt")
        # move to correct device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        with torch.no_grad():
            outputs = self.clip.get_image_features(**inputs)
            vec = outputs.cpu().numpy()[0]
            # normalize
            vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        vec = self.text_encoder.encode(text)
        # normalize
        vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec
        return vec.tolist()


# Low-level helper to upsert to Qdrant with named vectors
def upsert_point(
    client: QdrantClient,
    collection_name: str,
    point_id: int,
    visual_vector: Optional[List[float]],
    semantic_vector: Optional[List[float]],
    payload: dict,
):
    vectors: Dict[str, List[float]] = {}
    if visual_vector is not None:
        vectors["visual"] = visual_vector
    if semantic_vector is not None:
        vectors["semantic"] = semantic_vector
    client.upsert(
        collection_name=collection_name,
        points=[rest.PointStruct(id=point_id, vector=None if not vectors else vectors, payload=payload)],
    )


def ingest_images_from_dir(
    client: QdrantClient,
    directory: str,
    emb: Optional[Embedders] = None,
    source: Optional[str] = None,
    start_id: int = 0,
    max_files: Optional[int] = None,
) -> int:
    """Ingest images from a directory recursively. Expects that diagnostic label information may be encoded in a sidecar .json or the filename.

    The payload will include: stress_type, crop_name, severity (if available), source, filename, agronomist_notes

    Returns last used id + 1.
    """
    if emb is None:
        emb = Embedders()

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    pid = start_id
    for root, dirs, files in os.walk(directory):
        for fn in files:
            if max_files and pid - start_id >= max_files:
                return pid
            if os.path.splitext(fn)[1].lower() in allowed_ext:
                img_path = os.path.join(root, fn)
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                # Attempt to discover metadata
                meta = {"source": source or "dataset", "filename": fn}
                # Look for sidecar json with same name
                jpath = os.path.join(root, os.path.splitext(fn)[0] + ".json")
                if os.path.exists(jpath):
                    try:
                        with open(jpath, "r", encoding="utf-8") as fh:
                            jm = json.load(fh)
                        meta.update(jm)
                    except Exception:
                        pass
                # fallback: parse filename for label (naive)
                if 'label' not in meta and '_' in fn:
                    parts = fn.split('_')
                    if len(parts) >= 2:
                        meta['stress_type'] = parts[0]
                # Build a text description from available metadata
                desc = "Unknown"
                items = []
                if meta.get('stress_type'):
                    items.append(meta['stress_type'])
                if meta.get('severity'):
                    items.append(f"severity:{meta['severity']}")
                if meta.get('crop_name'):
                    items.append(meta['crop_name'])
                if items:
                    desc = ", ".join(items)
                # Generate embeddings
                try:
                    vis = emb.embed_image(img)
                except Exception as e:
                    print('Image embedding failed for', img_path, e)
                    continue
                sem = emb.embed_text(desc)
                payload = {
                    'stress_type': meta.get('stress_type'),
                    'crop_name': meta.get('crop_name'),
                    'severity': meta.get('severity'),
                    'source': meta.get('source'),
                    'filename': meta.get('filename'),
                    'agronomist_notes': meta.get('agronomist_notes'),
                    'text_description': desc,
                }
                upsert_point(client, KNOWLEDGE_COLLECTION, pid, vis, sem, payload)
                pid += 1
    return pid


def ingest_plantvillage_and_ip102(
    client: QdrantClient,
    plantvillage_dir: str,
    ip102_dir: str,
    emb: Optional[Embedders] = None,
    start_id: int = 0,
) -> int:
    """Convenience ingestion for the two standard datasets. Adjust hooks if dataset layout differs."""
    pid = start_id
    pid = ingest_images_from_dir(client, plantvillage_dir, emb=emb, source='PlantVillage', start_id=pid)
    pid = ingest_images_from_dir(client, ip102_dir, emb=emb, source='IP102', start_id=pid)
    return pid


# Agentic RAG: search + construct prompt
def agentic_diagnose(
    client: QdrantClient,
    image: Image.Image = None,
    image_path: Optional[str] = None,
    user_description: Optional[str] = None,
    emb: Optional[Embedders] = None,
    top_k: int = 3,
    llm_func: Optional[Any] = None,
) -> Dict[str, Any]:
    """Diagnose a new case by retrieving similar historical cases and asking an LLM for a grounded treatment plan.

    Returns: dict with keys 'retrieved', 'prompt', 'treatment'
    """
    if emb is None:
        emb = Embedders()
    if image is None and image_path is None:
        raise ValueError('Provide image or image_path')
    if image is None:
        image = Image.open(image_path).convert('RGB')
    img_v = emb.embed_image(image)
    # Visual search
    try:
        search_res = client.query_points(
            collection_name=KNOWLEDGE_COLLECTION,
            query=img_v,
            limit=top_k,
            using='visual',
            with_payload=True,
        ).points
    except (TypeError, AttributeError):
        # Older client may not support `query_points`; try legacy method
        try:
            search_res = client.search(collection_name=KNOWLEDGE_COLLECTION, query_vector=img_v, limit=top_k, with_payload=True)
        except AttributeError:
            search_res = []

    retrieved = []
    for hit in search_res:
        payload = hit.payload or {}
        retrieved.append({
            'id': hit.id,
            'score': getattr(hit, 'score', None),
            'payload': payload,
        })

    # Build a grounding prompt
    records_text = []
    for r in retrieved:
        p = r['payload']
        rec = f"Record ID: {r['id']}; Stress: {p.get('stress_type')} ; Crop: {p.get('crop_name')} ; Severity: {p.get('severity')} ; Notes: {p.get('agronomist_notes')}"
        records_text.append(rec)
    records_block = "\n".join(records_text)

    user_desc = user_description or "No additional description provided."
    prompt = (
        f"You are an agronomist assistant. Based ONLY on the following historical cases (do not hallucinate), and the user description, provide a clear, prioritized, and actionable treatment plan.\n"
        f"Historical Cases:\n{records_block}\n\n"
        f"Current case description: {user_desc}\n\n"
        f"Please: 1) list the top 3 likely stresses, 2) recommend immediate actions, 3) recommend follow-up monitoring steps, 4) list which historical records you used and why (traceability)."
    )

    treatment = None
    if llm_func:
        treatment = llm_func(prompt)
    else:
        # If no LLM provided, return the prompt so user can call Raptor Mini or other model
        treatment = {'prompt': prompt}

    result = {'retrieved': retrieved, 'prompt': prompt, 'treatment': treatment}
    return result


# Session memory helpers
def store_session_entry(
    client: QdrantClient,
    farm_id: str,
    plant_id: str,
    diagnosis: str,
    treatment: str,
    feedback: Optional[str] = None,
    emb: Optional[Embedders] = None,
) -> int:
    if emb is None:
        emb = Embedders()
    timestamp = datetime.utcnow().isoformat()
    text = f"farm:{farm_id} plant:{plant_id} diag:{diagnosis} treatment:{treatment} feedback:{feedback} ts:{timestamp}"
    vec = emb.embed_text(text)
    payload = {
        'farm_id': farm_id,
        'plant_id': plant_id,
        'diagnosis': diagnosis,
        'treatment': treatment,
        'feedback': feedback,
        'timestamp': timestamp,
    }
    # use a generated id based on time
    pid = int(time.time() * 1000)
    upsert_point(client, SESSION_COLLECTION, pid, None, vec, payload)
    return pid


def retrieve_session_history(client: QdrantClient, farm_id: str, plant_id: str, emb: Optional[Embedders] = None, top_k: int = 10) -> List[Dict[str, Any]]:
    if emb is None:
        emb = Embedders()
    query_text = f"farm:{farm_id} plant:{plant_id}"
    vec = emb.embed_text(query_text)
    try:
        hits = client.query_points(collection_name=SESSION_COLLECTION, query=vec, limit=top_k, using='semantic', with_payload=True).points
    except (TypeError, AttributeError):
        try:
            hits = client.search(collection_name=SESSION_COLLECTION, query_vector=vec, limit=top_k, with_payload=True)
        except AttributeError:
            hits = []
    out = []
    for h in hits:
        out.append({'id': h.id, 'score': getattr(h, 'score', None), 'payload': h.payload})
    return out


# Example pluggable LLM call (user should implement Raptor Mini call here)
def call_llm_with_raptor(prompt: str, model: str = "raptor-mini", **kwargs) -> str:
    """Placeholder: replace with your Raptor Mini API call. Return a string summary as treatment plan."""
    raise NotImplementedError("Integrate your Raptor Mini inference endpoint here (e.g., local server or HF TGI).")


# Convenience small CLI
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true', help='Create collections')
    parser.add_argument('--qdrant', default='http://localhost:6333', help='Qdrant URL')
    parser.add_argument('--pv', default=None, help='PlantVillage dir to ingest')
    parser.add_argument('--ip102', default=None, help='IP102 dir to ingest')
    args = parser.parse_args()
    client = QdrantClient(url=args.qdrant)
    if args.init:
        init_qdrant_collections(client, recreate=False)
        print('Initialized collections.')
    if args.pv or args.ip102:
        emb = Embedders()
        pid = 0
        if args.pv:
            pid = ingest_images_from_dir(client, args.pv, emb=emb, source='PlantVillage', start_id=pid)
        if args.ip102:
            pid = ingest_images_from_dir(client, args.ip102, emb=emb, source='IP102', start_id=pid)
        print('Ingested ending at id', pid)
