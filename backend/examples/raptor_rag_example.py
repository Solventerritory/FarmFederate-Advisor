"""Example usage of backend/qdrant_rag.py

This script demonstrates:
- initializing Qdrant collections
- (optionally) ingesting image directories (dry-run recommended)
- performing an agentic diagnosis call that returns a constructed prompt
- storing a session memory entry

Note: This is a small demonstration. It will not call Raptor Mini by default; integrate your LLM by providing `llm_func` to `agentic_diagnose`.
"""
from qdrant_client import QdrantClient
from qdrant_rag import (
    init_qdrant_collections,
    ingest_images_from_dir,
    agentic_diagnose,
    store_session_entry,
    retrieve_session_history,
    Embedders,
)
from PIL import Image


def demo(qdrant_url='http://localhost:6333'):
    client = QdrantClient(url=qdrant_url)
    print('Initializing collections...')
    init_qdrant_collections(client)

    emb = Embedders()

    # Example: ingest small folder (dry run) - POINT TO your dataset path to actually ingest
    pv_dir = '/path/to/sample/PlantVillage_small'
    # Uncomment to ingest if you have a small sample set
    # ingest_images_from_dir(client, pv_dir, emb=emb, source='PlantVillage', max_files=10)

    # Example diagnostic call (without an actual model, returns a prompt)
    test_img = Image.new('RGB', (224, 224), color='green')
    res = agentic_diagnose(client, image=test_img, user_description='Yellowing leaves with small spots', emb=emb)
    print('Constructed prompt:\n')
    print(res['prompt'])

    # Store a sample session entry
    sid = store_session_entry(client, farm_id='farm_001', plant_id='plant_123', diagnosis='nutrient_def', treatment='apply N-rich fertilizer', feedback='plant recovered partially', emb=emb)
    print('Stored session entry id', sid)

    hist = retrieve_session_history(client, farm_id='farm_001', plant_id='plant_123', emb=emb)
    print('Session history (len):', len(hist))


if __name__ == '__main__':
    demo()
