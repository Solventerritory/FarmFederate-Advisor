"""
Colab-optimized Qdrant RAG utilities (low-memory defaults).
- Uses in-memory or local-disk Qdrant based on env var USE_QDRANT_LOCAL
- Default small batch sizes, FP16 support when CUDA available
- Safe dry-run and sample ingestion for quick demos
"""
from __future__ import annotations
import os
import time
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

# Optional imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except Exception:
    QdrantClient = None
    rest = None

try:
    import torch
    from PIL import Image
except Exception:
    torch = None
    Image = None

# Lightweight CLIP via transformers
try:
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    CLIPProcessor = None
    CLIPModel = None

# SentenceTransformer for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Defaults
KNOWLEDGE_COLLECTION = "crop_health_knowledge"
SESSION_COLLECTION = "farm_session_memory"
CLIP_MODEL = os.environ.get('RAG_CLIP_MODEL', 'openai/clip-vit-base-patch32')
TEXT_MODEL = os.environ.get('RAG_TEXT_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BATCH_SIZE = int(os.environ.get('RAG_BATCH_SIZE', '8'))
USE_QDRANT_LOCAL = os.environ.get('USE_QDRANT_LOCAL', '0') == '1'
QDRANT_PATH = os.environ.get('QDRANT_PATH', '')
RAG_FP16 = os.environ.get('RAG_FP16', '1') == '1'


def get_qdrant_client():
    if QdrantClient is None:
        raise ImportError('qdrant-client not installed. pip install qdrant-client')
    if USE_QDRANT_LOCAL:
        if QDRANT_PATH:
            return QdrantClient(path=QDRANT_PATH)
        return QdrantClient(':memory:')
    # default remote
    url = os.environ.get('QDRANT_URL', 'http://localhost:6333')
    api_key = os.environ.get('QDRANT_API_KEY')
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)


class ColabEmbedders:
    def __init__(self, device: Optional[str] = None, clip_model: str = CLIP_MODEL, text_model: str = TEXT_MODEL):
        if torch is None:
            raise ImportError('torch is required for Colab embedders')
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if RAG_FP16 and self.device.startswith('cuda'):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError('transformers[torch] is required for CLIP')
        if SentenceTransformer is None:
            raise ImportError('sentence-transformers is required for text embeddings')
        # Load models with low-CPU-memory footprint (use local cache)
        self.clip = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.text_encoder = SentenceTransformer(text_model, device=self.device)

    def embed_image(self, pil_img: Image.Image) -> List[float]:
        inputs = self.processor(images=pil_img, return_tensors='pt')
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            feats = self.clip.get_image_features(**inputs)
            vec = feats.cpu().numpy()[0]
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        vec = self.text_encoder.encode(text)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.tolist()


def init_colab_collections(client: QdrantClient, recreate: bool = False):
    if QdrantClient is None or rest is None:
        raise ImportError('qdrant-client is required')
    if recreate:
        try:
            client.delete_collection(KNOWLEDGE_COLLECTION)
            client.delete_collection(SESSION_COLLECTION)
        except Exception:
            pass
    client.create_collection(collection_name=KNOWLEDGE_COLLECTION, vectors={"visual": rest.VectorParams(size=512, distance=rest.Distance.COSINE), "semantic": rest.VectorParams(size=384, distance=rest.Distance.COSINE)})
    client.create_collection(collection_name=SESSION_COLLECTION, vectors={"semantic": rest.VectorParams(size=384, distance=rest.Distance.COSINE)})


def small_ingest_sample(client: QdrantClient, img_paths: List[str], source: str = 'colab_sample', emb: Optional[ColabEmbedders] = None):
    if emb is None:
        emb = ColabEmbedders()
    pid = int(time.time() * 1000)
    for p in img_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            continue
        vis = emb.embed_image(img)
        sem = emb.embed_text(os.path.basename(p))
        payload = {'filename': os.path.basename(p), 'source': source}
        vectors = {'visual': vis, 'semantic': sem}
        client.upsert(collection_name=KNOWLEDGE_COLLECTION, points=[rest.PointStruct(id=pid, vector=vectors, payload=payload)])
        pid += 1
    return pid


# Minimal agentic diagnose that uses visual search and returns prompt+retrieved
def agentic_diagnose_colab(client: QdrantClient, image, user_description: str = '', emb: Optional[ColabEmbedders] = None, top_k: int =3, llm_func=None):
    if emb is None:
        emb = ColabEmbedders()
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image
    vis = emb.embed_image(img)
    try:
        hits = client.search(collection_name=KNOWLEDGE_COLLECTION, query_vector=vis, limit=top_k, vector_name='visual', with_payload=True)
    except TypeError:
        hits = client.search(collection_name=KNOWLEDGE_COLLECTION, query_vector=vis, limit=top_k, with_payload=True)
    retrieved = [{'id': h.id, 'score': getattr(h,'score',None), 'payload': h.payload} for h in hits]
    records_text = '\n'.join([f"ID:{r['id']} Stress:{r['payload'].get('stress_type')} Notes:{r['payload'].get('agronomist_notes')}" for r in retrieved])
    prompt = f"Based ONLY on these cases:\n{records_text}\nCurrent: {user_description}\nProvide treatment and reference record IDs used."
    out = llm_func(prompt) if llm_func else {'prompt': prompt}
    return {'prompt': prompt, 'retrieved': retrieved, 'treatment': out}


# Small helper for session memory
def store_session_colab(client: QdrantClient, farm_id: str, plant_id: str, diagnosis: str, treatment: str, feedback: str = '', emb: Optional[ColabEmbedders] = None):
    if emb is None:
        emb = ColabEmbedders()
    ts = datetime.utcnow().isoformat()
    text = f"farm:{farm_id} plant:{plant_id} diag:{diagnosis} treatment:{treatment} feedback:{feedback} ts:{ts}"
    vec = emb.embed_text(text)
    pid = int(time.time() * 1000)
    client.upsert(collection_name=SESSION_COLLECTION, points=[rest.PointStruct(id=pid, vector={'semantic': vec}, payload={'farm_id': farm_id, 'plant_id': plant_id, 'diagnosis': diagnosis, 'treatment': treatment, 'feedback': feedback, 'timestamp': ts})])
    return pid


def retrieve_session_history_colab(client: QdrantClient, farm_id: str, plant_id: str, emb: Optional[ColabEmbedders] = None, top_k: int =10):
    if emb is None:
        emb = ColabEmbedders()
    query_text = f"farm:{farm_id} plant:{plant_id}"
    vec = emb.embed_text(query_text)
    try:
        hits = client.search(collection_name=SESSION_COLLECTION, query_vector=vec, limit=top_k, vector_name='semantic', with_payload=True)
    except TypeError:
        hits = client.search(collection_name=SESSION_COLLECTION, query_vector=vec, limit=top_k, with_payload=True)
    return [{'id': h.id, 'score': getattr(h, 'score', None), 'payload': h.payload} for h in hits]
