"""Memory-efficient Qdrant helpers optimized for Colab T4 instances.

Implements:
- MemoryEfficientIngestor: small-batch upsert with optional CUDA cache clearing
- QdrantFarmAgent: lightweight agent wrapper using Qdrant FastEmbed/visual model
- print_traceable_log: helper to render traceable retrieval + reasoning steps

This file is safe to import even when qdrant-client is not present; imports are guarded and clear errors are raised.
"""
from typing import List, Optional, Dict
import os
import time

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except Exception as e:
    QdrantClient = None  # type: ignore
    models = None  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore


class MemoryEfficientIngestor:
    """Processes upserts in small chunks to avoid RAM spikes on T4 instances."""

    def __init__(self, client: 'QdrantClient', batch_size: int = 16):
        if QdrantClient is None:
            raise ImportError('qdrant-client is required for MemoryEfficientIngestor')
        self.client = client
        self.batch_size = int(batch_size)

    def batch_upsert(self, collection_name: str, points: List[models.PointStruct]):
        for i in range(0, len(points), self.batch_size):
            chunk = points[i : i + self.batch_size]
            self.client.upsert(collection_name=collection_name, points=chunk)
            # Clear CUDA cache if using GPU for embeddings
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class QdrantFarmAgent:
    """Lightweight agent wrapper for Qdrant with memory-aware defaults.

    Usage:
        client = QdrantClient(':memory:')
        agent = QdrantFarmAgent(client)
        agent.init_collections()
        agent.ingest_knowledge_batch(images, labels, batch_size=8)
        prompt = agent.agentic_reasoning(farm_id, plant_image, user_desc)

    Notes:
      - This uses Qdrant server-side embedding models when available (fastembed).
      - If client does not support set_model/set_visual_model, the methods still work but
        will require precomputed embeddings provided by the caller.
    """

    def __init__(self, client: 'QdrantClient'):
        if QdrantClient is None or client is None:
            raise ImportError('qdrant-client must be installed and a client instance provided')
        self.client = client
        # Try to enable server-side lightweight embedding models when available
        try:
            if hasattr(self.client, 'set_model'):
                # sentence-transformers/all-MiniLM-L6-v2 is commonly available
                try:
                    self.client.set_model('sentence-transformers/all-MiniLM-L6-v2')
                except Exception:
                    pass
            if hasattr(self.client, 'set_visual_model'):
                try:
                    self.client.set_visual_model('clip-vit-base-patch32')
                except Exception:
                    pass
        except Exception:
            # non-fatal
            pass

    def init_collections(self):
        """Create or recreate collections optimized for visual + semantic search and session memory."""
        if models is None:
            raise ImportError('qdrant-client http models are required to create collections')
        # Knowledge Base: expert cases
        try:
            self.client.recreate_collection(
                collection_name='crop_knowledge',
                vectors_config={
                    'visual': models.VectorParams(size=512, distance=models.Distance.COSINE),
                    'semantic': models.VectorParams(size=384, distance=models.Distance.COSINE),
                },
            )
        except Exception:
            # Fallback single-vector collection
            try:
                self.client.recreate_collection(
                    collection_name='crop_knowledge',
                    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
                )
            except Exception:
                pass

        # Session memory per farm
        try:
            self.client.recreate_collection(
                collection_name='farm_history',
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        except Exception:
            pass

    def ingest_knowledge_batch(self, images: List, labels: List[str], batch_size: int = 8):
        """Ingest images + labels into the `crop_knowledge` collection in small chunks.

        Parameters
        - images: list of image binary/URLs/PIL.Image objects acceptable by the client's visual embedder
        - labels: list of labels/strings of same length
        - batch_size: upsert chunk size
        """
        if len(images) != len(labels):
            raise ValueError('images and labels must have same length')
        if not hasattr(self.client, 'embed_visual'):
            # the client may not provide embed_visual; user must pass precomputed vectors
            raise RuntimeError('Qdrant client does not support `embed_visual`; precompute embeddings and use upsert directly')

        for i in range(0, len(images), batch_size):
            chunk_imgs = images[i : i + batch_size]
            chunk_labels = labels[i : i + batch_size]
            try:
                embeddings = self.client.embed_visual(chunk_imgs)
            except Exception:
                # If visual embedding fails, skip these points rather than crash
                embeddings = [None] * len(chunk_imgs)
            points = []
            base = int(time.time() * 1000) % (10 ** 9)
            for j, emb in enumerate(embeddings):
                pid = base + i + j
                payload = {'label': chunk_labels[j], 'timestamp': int(time.time())}
                if emb is not None:
                    p = models.PointStruct(id=pid, vector={'visual': emb}, payload=payload)
                else:
                    p = models.PointStruct(id=pid, vector=None, payload=payload)
                points.append(p)
            try:
                self.client.upsert(collection_name='crop_knowledge', points=points)
            except Exception as e:
                # Try upserting one by one if bulk fails
                for p in points:
                    try:
                        self.client.upsert(collection_name='crop_knowledge', points=[p])
                    except Exception:
                        continue
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def agentic_reasoning(self, farm_id: str, plant_image, user_desc: str, top_k: int = 3):
        """Construct a traceable prompt by searching global knowledge and farm-specific memory.

        Returns a dict with keys: prompt, global_matches, farm_history
        """
        # 1) Search global knowledge by visual similarity
        global_matches = []
        try:
            if hasattr(self.client, 'embed_visual'):
                qvec = self.client.embed_visual([plant_image])[0]
                global_matches = self.client.search(collection_name='crop_knowledge', query_vector=('visual', qvec), limit=top_k)
            else:
                # if no embedder, empty matches
                global_matches = []
        except Exception:
            global_matches = []

        # 2) Retrieve recent farm history
        farm_history = []
        try:
            scroll = self.client.scroll(collection_name='farm_history', limit=top_k)
            # Filter client-side by farm_id if server-side filter not available
            for batch in scroll:
                for p in batch:
                    try:
                        if p.payload.get('farm_id') == farm_id:
                            farm_history.append(p)
                    except Exception:
                        continue
                if len(farm_history) >= top_k:
                    break
        except Exception:
            # Fallback: empty history
            farm_history = []

        # Build the prompt
        history_context = "\n".join([f"- Past Issue: {h.payload.get('diagnosis', 'unknown')} (t={h.payload.get('timestamp')})" for h in farm_history])
        knowledge_context = "\n".join([f"- Similar Case: {m.payload.get('label', 'unknown')} (score={getattr(m, 'score', None)})" for m in global_matches])

        prompt = f"""
Current Observation: {user_desc}
Farm History for {farm_id}:
{history_context}

Global Expert Matches:
{knowledge_context}

Task: Provide a grounded, traceable recommendation for the farmer. Cite the similar cases and past farm events.
"""

        return {'prompt': prompt, 'global_matches': global_matches, 'farm_history': farm_history}


def print_traceable_log(farm_id: str, query: str, results: List, history: List):
    """Pretty-print a traceable interaction log for a farmer/plot."""
    print(f"--- AGENT LOG for {farm_id} ---")
    print(f"INPUT: {query}")
    print("RECALLING MEMORY...")
    for h in history:
        ts = h.payload.get('timestamp') if hasattr(h, 'payload') else None
        diag = h.payload.get('diagnosis') if hasattr(h, 'payload') else None
        print(f"  [MEMORY FOUND]: {ts} - {diag}")

    print("SEARCHING KNOWLEDGE BASE...")
    for r in results:
        label = r.payload.get('label') if hasattr(r, 'payload') else None
        score = getattr(r, 'score', None)
        print(f"  [EXPERT CASE MATCH]: {label} (Confidence: {score:.2f})")

    print("\nFINAL GROUNDED ACTION GENERATED.")
