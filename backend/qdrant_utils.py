"""Qdrant utils for FarmFederate RAG system

Provides helpers to initialize Qdrant collections for:
 - crop_health_knowledge (visual + semantic named vectors)
 - farm_session_memory (semantic vectors for session history)

Requires: qdrant-client
"""
from typing import Optional
import os

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except Exception as e:
    raise ImportError("qdrant-client is required. Install with `pip install qdrant-client`.")


def get_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None, prefer_grpc: bool = False) -> QdrantClient:
    """Create a Qdrant client. Uses env vars QDRANT_URL and QDRANT_API_KEY if not provided."""
    url = url or os.environ.get('QDRANT_URL', 'http://localhost:6333')
    api_key = api_key or os.environ.get('QDRANT_API_KEY')
    # QdrantClient takes url and prefer_grpc flag
    return QdrantClient(url=url, prefer_grpc=prefer_grpc, api_key=api_key)


def initialize_crop_health_collection(client: QdrantClient, collection_name: str = 'crop_health_knowledge') -> None:
    """Create 'crop_health_knowledge' collection with named vectors 'visual' and 'semantic',
    and payload indexes for stress_type, crop_name, severity.

    - visual: size 512 (CLIP image embeddings)
    - semantic: size 384 (text embeddings)
    """
    # Check if exists and exit early
    existing = client.get_collections().collections
    if any(col.name == collection_name for col in existing):
        print(f"Collection '{collection_name}' already exists. Skipping creation.")
        return

    vectors_config = {
        'visual': rest.VectorParams(size=512, distance=rest.Distance.COSINE),
        'semantic': rest.VectorParams(size=384, distance=rest.Distance.COSINE),
    }

    client.recreate_collection(collection_name, vectors_config=vectors_config)
    print(f"Created collection '{collection_name}' with vectors: {list(vectors_config.keys())}")

    # Create payload indexes (for filters & faster queries)
    try:
        client.create_payload_index(collection_name, payload_key='stress_type', field_schema=rest.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, payload_key='crop_name', field_schema=rest.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, payload_key='severity', field_schema=rest.PayloadSchemaType.INTEGER)
        print(f"Created payload indexes for 'stress_type','crop_name','severity' on '{collection_name}'")
    except Exception as e:
        print('Warning: creating payload indexes failed:', e)


def initialize_farm_session_collection(client: QdrantClient, collection_name: str = 'farm_session_memory') -> None:
    """Create a collection to store session memory summarized as semantic vectors (size 384)."""
    existing = client.get_collections().collections
    if any(col.name == collection_name for col in existing):
        print(f"Collection '{collection_name}' already exists. Skipping creation.")
        return

    vectors_config = {
        'session': rest.VectorParams(size=384, distance=rest.Distance.COSINE),
    }
    client.recreate_collection(collection_name, vectors_config=vectors_config)
    print(f"Created collection '{collection_name}' with vector 'session'")

    try:
        client.create_payload_index(collection_name, payload_key='farm_id', field_schema=rest.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, payload_key='plant_id', field_schema=rest.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, payload_key='timestamp', field_schema=rest.PayloadSchemaType.INTEGER)
        print(f"Created payload indexes for 'farm_id','plant_id','timestamp' on '{collection_name}'")
    except Exception as e:
        print('Warning: creating payload indexes failed:', e)
