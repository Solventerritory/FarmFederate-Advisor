from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from PIL import Image
import base64
import io

app = FastAPI(title="FarmFederate VLM Service")

# Configurable via environment
QDRANT_URL = None
QDRANT_INMEM = True
COLLECTION = 'crop_stress_knowledge'

# Simple in-process client (in-memory by default)
qdrant = QdrantClient(':memory:')
text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = SentenceTransformer('clip-ViT-B-32')

class QueryPayload(BaseModel):
    text: Optional[str] = None
    image_b64: Optional[str] = None
    top_k: int = 5
    stress_filter: Optional[str] = None

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.post('/query')
async def query(payload: QueryPayload):
    if not payload.text and not payload.image_b64:
        raise HTTPException(status_code=400, detail='Provide text or image_b64')

    results = []
    search_filter = None
    if payload.stress_filter:
        search_filter = Filter(must=[FieldCondition(key='label', match=MatchValue(value=payload.stress_filter))])

    try:
        if payload.text:
            emb = text_embedder.encode(payload.text).tolist()
            res = qdrant.query_points(collection_name=COLLECTION, query=emb, using='semantic', limit=payload.top_k, query_filter=search_filter).points
            for r in res:
                results.append({'id': r.id, 'score': r.score, **r.payload})
        if payload.image_b64:
            img_bytes = base64.b64decode(payload.image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            emb = clip_model.encode(img).tolist()
            res = qdrant.query_points(collection_name=COLLECTION, query=emb, using='visual', limit=payload.top_k, query_filter=search_filter).points
            for r in res:
                results.append({'id': r.id, 'score': r.score, **r.payload})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # deduplicate by id, keep highest score
    seen = {}
    for r in results:
        if r['id'] not in seen or r['score'] > seen[r['id']]['score']:
            seen[r['id']] = r
    out = sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:payload.top_k]
    return {'results': out}

# Note: To run this server in production, the user should set up a persistent Qdrant instance
# and set QDRANT_URL accordingly. This file is a minimal example for local / dev usage.