"""FarmMemoryAgent

Provides a lightweight wrapper around Qdrant to implement long-term session memory
for ``Crop Stress Analysis Reports``. The agent stores multimodal named vectors
('visual' and 'semantic') and supports per-farm filtering for retrieval.

Usage example:
    from backend.farm_memory_agent import FarmMemoryAgent
    agent = FarmMemoryAgent()
    agent.init_collection()  # creates the `farm_history` collection if missing
    agent.store_report(report_text, image_emb, semantic_emb, farm_id='farm:42')
    hits = agent.retrieve_similar_by_image(image_emb, farm_id='farm:42', top_k=3)
"""
from __future__ import annotations

import os
import uuid
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class FarmMemoryAgent:
    COLLECTION = "farm_history"

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        prefer_grpc: bool = False,
        visual_dim: int = 512,
        semantic_dim: int = 384,
        distance: rest.Distance = rest.Distance.COSINE,
    ):
        """Initialize the FarmMemoryAgent.

        Args:
            qdrant_url: URL of Qdrant server or ':memory:' for in-memory demo. If None,
                will try env var QDRANT_URL and default to ':memory:'.
            prefer_grpc: whether to use gRPC transport (ignored if in-memory).
            visual_dim: dimensionality of image/visual embeddings (ViT/CLIP).
            semantic_dim: dimensionality of semantic/LLM embeddings.
            distance: distance metric for vector search (COSINE by default).
        """
        qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", ":memory:")
        self._qdrant_url = qdrant_url
        self._visual_dim = int(visual_dim)
        self._semantic_dim = int(semantic_dim)
        self._distance = distance

        if qdrant_url == ":memory:":
            # In-memory Qdrant for demos (no auth). Use the special ':memory:' URL
            # which the qdrant-client interprets as an in-memory instance.
            self.client = QdrantClient(":memory:")
        else:
            # For remote Qdrant, trust QDRANT_URL and optional QDRANT_API_KEY
            api_key = os.environ.get("QDRANT_API_KEY")
            # Use HTTP client; the SDK will pick correct transport.
            self.client = QdrantClient(url=qdrant_url, prefer_grpc=prefer_grpc, api_key=api_key)

    def init_collection(self, recreate: bool = False) -> None:
        """Create collection `farm_history` with named vectors 'visual' and 'semantic'.

        Args:
            recreate: if True, drop and recreate the collection.
        """
        if recreate:
            try:
                existing = self.client.get_collection(self.COLLECTION)
                if existing is not None:
                    try:
                        self.client.delete_collection(self.COLLECTION)
                    except Exception:
                        pass
            except Exception:
                # collection doesn't exist; nothing to delete
                pass

        # If already exists, return
        try:
            existing = self.client.get_collection(self.COLLECTION)
            if existing is not None:
                return
        except Exception:
            pass

        vectors_config = {
            "visual": rest.VectorParams(size=self._visual_dim, distance=self._distance),
            "semantic": rest.VectorParams(size=self._semantic_dim, distance=self._distance),
        }

        print(f"Creating collection {self.COLLECTION} with visual_dim={self._visual_dim} semantic_dim={self._semantic_dim}")
        # qdrant-client expects `vectors_config` argument name
        self.client.recreate_collection(collection_name=self.COLLECTION, vectors_config=vectors_config)

        # No initial upsert needed; payload fields are indexed when points are added.
        # Leaving empty collection ready for inserts.
        pass

    def _ensure_collection(self) -> None:
        try:
            self.init_collection(recreate=False)
        except Exception:
            self.init_collection(recreate=True)

    def store_report(
        self,
        report_text: str,
        image_embedding: Iterable[float],
        semantic_embedding: Optional[Iterable[float]],
        farm_id: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a crop stress analysis report and associated embeddings.

        Args:
            report_text: textual summary or diagnosis.
            image_embedding: visual embedding (ViT/CLIP) as a 1D iterable.
            semantic_embedding: semantic embedding (LLM) as a 1D iterable (optional).
            farm_id: identifier for the farm (used as filter key).
            timestamp: epoch timestamp; defaults to now.
            metadata: optional dict with extra structured info (e.g., label, severity).

        Returns:
            id: the internal id of the stored point.
        """
        self._ensure_collection()
        ts = float(timestamp or time.time())
        pid = str(uuid.uuid4())
        payload = {
            "farm_id": farm_id,
            "report": report_text,
            "timestamp": ts,
        }
        if metadata:
            payload.update({"meta": metadata})

        vectors = {"visual": list(np.array(image_embedding).astype(float))}
        if semantic_embedding is not None:
            vectors["semantic"] = list(np.array(semantic_embedding).astype(float))

        point = rest.PointStruct(id=pid, payload=payload, vector=vectors)

        # Upsert point
        self.client.upsert(collection_name=self.COLLECTION, points=[point])
        return pid

    def retrieve_similar_by_image(self, image_embedding: Iterable[float], farm_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top-K visually-similar historical reports for the given farm.

        Args:
            image_embedding: query visual embedding (1D iterable).
            farm_id: restrict search to this farm_id (scoped memory).
            top_k: number of results to return.

        Returns:
            A list of dicts: [{'id', 'score', 'payload'}...]
        """
        self._ensure_collection()
        qv = list(np.array(image_embedding).astype(float))

        f = rest.Filter(must=[rest.FieldCondition(key="farm_id", match=rest.MatchValue(value=farm_id))])

        # qdrant-client uses `query_points` in v1.7+; fall back when necessary
        try:
            hits = self.client.query_points(
                collection_name=self.COLLECTION,
                query=qv,
                using="visual",
                query_filter=f,
                limit=top_k,
                with_payload=True,
            ).points
        except (AttributeError, TypeError):
            # Fall back to scroll if query_points is not available
            try:
                hits = self.client.scroll(
                    collection_name=self.COLLECTION,
                    scroll_filter=f,
                    limit=top_k,
                    with_payload=True,
                )[0]
            except Exception as e:
                raise

        out = []
        for h in hits:
            # ScoredPoint has id, score, payload
            out.append({"id": str(h.id), "score": float(getattr(h, 'score', 0.0) or 0.0), "payload": h.payload})
        return out

        out = []
        for h in hits:
            out.append({"id": str(h.id), "score": float(h.score or 0.0), "payload": h.payload})
        return out

    def get_reports_for_farm(self, farm_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Return up to `limit` stored reports for a farm (non-vector scroll).

        Useful for building timelines / evidence lists.
        """
        self._ensure_collection()
        f = rest.Filter(must=[rest.FieldCondition(key="farm_id", match=rest.MatchValue(value=farm_id))])
        pts, _ = self.client.scroll(collection_name=self.COLLECTION, scroll_filter=f, limit=limit)
        out = []
        for p in pts:
            out.append({"id": str(p.id), "payload": p.payload})
        return out

    def export_reports_to_csv(self, farm_id: str, out_path: str = "farm_reports.csv", limit: int = 1000) -> str:
        """Export reports for a farm to a CSV file. Returns the path of the created file."""
        rows = self.get_reports_for_farm(farm_id, limit=limit)
        if not rows:
            # Create empty CSV with headers
            pd.DataFrame(columns=["id", "farm_id", "report", "timestamp", "meta"]).to_csv(out_path, index=False)
            return out_path

        recs = []
        for r in rows:
            payload = r.get("payload", {})
            recs.append(
                {
                    "id": r.get("id"),
                    "farm_id": payload.get("farm_id"),
                    "report": payload.get("report"),
                    "timestamp": payload.get("timestamp"),
                    "meta": payload.get("meta"),
                }
            )
        pd.DataFrame(recs).to_csv(out_path, index=False)
        return out_path


# Convenience alias for quick demos
def demo_agent() -> FarmMemoryAgent:
    agent = FarmMemoryAgent()
    agent.init_collection()
    return agent
