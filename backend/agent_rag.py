"""Agentic RAG logic for multimodal crop diagnosis.

Functions:
 - rag_diagnose: given image + user_description, retrieve similar cases and call LLM to generate grounded treatment plan
 - store_session_memory: store diagnosis entries for farm/plant
 - check_session_history: fetch recent sessions for the plant

Note: This module uses Qdrant for retrieval and HuggingFace Inference API for calling Raptor Mini (if HF_TOKEN set).
"""
from typing import List, Optional, Dict, Any
import os
import time
import json
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except Exception:
    raise ImportError('qdrant-client is required. pip install qdrant-client')

try:
    import requests
except Exception:
    requests = None

from ingest_qdrant import Embedders


def _format_retrieved(records: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        payload = r.payload or {}
        out.append({
            'id': r.id,
            'score': r.score,
            'label': payload.get('label'),
            'stress_type': payload.get('stress_type'),
            'description': payload.get('description'),
            'source': payload.get('source'),
            'path': payload.get('path'),
            'payload': payload,
        })
    return out


def call_raptor_mini(prompt: str, hf_token: Optional[str] = None, model: str = 'mosaicml/raptor-mini') -> str:
    """Call Raptor Mini via Hugging Face Inference API if HF_TOKEN is provided.
    If no token is provided, return the prompt + a note for offline mode (useful for tests).
    """
    hf_token = hf_token or os.environ.get('HF_TOKEN')
    if not hf_token or requests is None:
        return f"[OFFLINE MODE] Would call Raptor Mini with prompt:\n{prompt}\n\n(Set HF_TOKEN to enable real inference.)"

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256}}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HF inference failed: {resp.status_code} {resp.text}")
    j = resp.json()
    # HF text generation returns list or dict depending on model
    if isinstance(j, dict) and 'error' in j:
        raise RuntimeError('HF error: ' + j['error'])
    # j may be [{'generated_text': '...'}]
    if isinstance(j, list) and len(j) and isinstance(j[0], dict) and 'generated_text' in j[0]:
        return j[0]['generated_text']
    # fallback
    return json.dumps(j)


def rag_diagnose(qdrant_client: QdrantClient, image_path: str, user_description: str, embedders: Optional[Embedders] = None, top_k: int = 3) -> Dict[str, Any]:
    """Perform diagnosis: embed image, retrieve top-k similar cases, and call LLM with grounding.

    Returns dict with:
      - retrieved : list of records
      - prompt: the exact prompt sent to LLM
      - llm_output: model response
    """
    embedders = embedders or Embedders()
    visual_vec = embedders.image_to_visual(image_path)

    search_res = qdrant_client.search(collection_name='crop_health_knowledge', query_vector=visual_vec, limit=top_k, vector_name='visual')
    retrieved = _format_retrieved(search_res)

    # Construct grounding block
    grounding = '\n\n'.join([f"Record {i+1}: label={r['label']}, stress_type={r['stress_type']}, description={r['description']}, source={r['source']}, path={r['path']}" for i, r in enumerate(retrieved)])

    prompt = (
        "You are a crop diagnostic assistant. Use the retrieved historical cases below to provide a personalized, actionable treatment plan. "
        "Be specific and cite the records that influenced each recommendation. Do NOT hallucinate beyond the retrieved content.\n\n"
        f"Retrieved historical cases:\n{grounding}\n\n"
        f"Current observation: {user_description}\n\n"
        "Provide a short diagnosis, a prioritized treatment plan (3 steps max), and what to monitor next with suggested timeframe. For each recommendation, reference which retrieved record(s) support it."
    )

    llm_output = call_raptor_mini(prompt)

    return {
        'retrieved': retrieved,
        'prompt': prompt,
        'llm_output': llm_output,
    }


def store_session_memory(qdrant_client: QdrantClient, farm_id: str, plant_id: str, diagnosis_text: str, feedback: Optional[str] = None, embedders: Optional[Embedders] = None):
    """Store a session entry in 'farm_session_memory' collection."""
    embedders = embedders or Embedders()
    sem_vec = embedders.text_to_semantic(diagnosis_text + (" Feedback: " + feedback if feedback else ""))
    timestamp = int(time.time())
    payload = {
        'farm_id': farm_id,
        'plant_id': plant_id,
        'diagnosis': diagnosis_text,
        'feedback': feedback,
        'timestamp': timestamp,
    }
    point_id = f"session::{farm_id}::{plant_id}::{timestamp}"
    p = rest.PointStruct(id=point_id, vector={'session': sem_vec}, payload=payload)
    qdrant_client.upsert(collection_name='farm_session_memory', points=[p])
    return payload


def check_session_history(qdrant_client: QdrantClient, farm_id: str, plant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent session memory for farm_id+plant_id ordered by timestamp (descending)."""
    # Use simple scroll + filter by payload
    from qdrant_client.http import models as rest_models
    flt = rest_models.Filter(must=[rest_models.FieldCondition(key='farm_id', match=rest_models.MatchValue(value=farm_id)), rest_models.FieldCondition(key='plant_id', match=rest_models.MatchValue(value=plant_id))])
    hits = qdrant_client.search(collection_name='farm_session_memory', query_vector=None, limit=limit, filter=flt, vector_name='session')
    out = []
    for r in hits:
        out.append({
            'id': r.id,
            'score': r.score,
            'payload': r.payload,
        })
    # sort by timestamp desc
    out_sorted = sorted(out, key=lambda x: x['payload'].get('timestamp', 0), reverse=True)
    return out_sorted


if __name__ == '__main__':
    # Simple CLI demo
    from qdrant_utils import get_qdrant_client, initialize_crop_health_collection, initialize_farm_session_collection
    client = get_qdrant_client()
    initialize_crop_health_collection(client)
    initialize_farm_session_collection(client)
    print('Collections ready. Run ingestion or use the functions programmatically.')
