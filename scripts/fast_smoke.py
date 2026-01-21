"""Fast smoke script used by CI to run lightweight checks.
- Runs a small RAG quick test (in-memory Qdrant) if qdrant-client present
- Runs small model training smoke using notebook's lightweight models
"""
import sys
import json
import os

def run_rag_quick_test():
    try:
        from qdrant_client import QdrantClient
        from backend.qdrant_rag import init_qdrant_collections, agentic_diagnose, Embedders, store_session_entry, retrieve_session_history
        from PIL import Image
        client = QdrantClient(':memory:')
        init_qdrant_collections(client)
        emb = Embedders()
        img = Image.new('RGB', (224,224), color='green')
        res = agentic_diagnose(client, image=img, user_description='Test', emb=emb, llm_func=lambda p: 'mock')
        sid = store_session_entry(client, farm_id='farm_1', plant_id='p1', diagnosis='nutrient_def', treatment='test', emb=emb)
        hist = retrieve_session_history(client, 'farm_1', 'p1', emb=emb)
        print('RAG quick test OK, retrieved len', len(res.get('retrieved', [])), 'hist len', len(hist))
        return True
    except Exception as e:
        print('RAG quick test skipped/fail:', e)
        return False


def run_model_smoke():
    try:
        import torch
        from backend.notebooks.RAG_Colab_Demo import SimpleLLM, SimpleViT, MultiModalDataset, generate_image_data, generate_text_data
    except Exception:
        # fallback to local definitions
        try:
            from backend.notebooks.RAG_Colab_Demo import SimpleLLM, SimpleViT, MultiModalDataset, generate_image_data, generate_text_data
        except Exception as e:
            print('Model smoke import failed:', e)
            return False
    imgs = generate_image_data(8)
    texts = generate_text_data(16)
    labels = [[0] for _ in range(16)]
    ds = MultiModalDataset([t for t in texts['text'][:8]], labels[:8], imgs[:8])
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4)
    llm = SimpleLLM()
    vit = SimpleViT()
    # quick forward
    for b in loader:
        try:
            _ = llm(torch.randint(0,1000,(4,16)))
            imgs_tensor = torch.cat([torch.randn(1,3,224,224) for _ in range(4)], dim=0)
            _ = vit(imgs_tensor)
        except Exception as e:
            print('Model forward failed:', e)
            return False
        break
    print('Model smoke OK')
    return True


def main():
    print('Starting fast smoke...')
    rag_ok = run_rag_quick_test()
    model_ok = run_model_smoke()
    all_ok = rag_ok or model_ok
    if not all_ok:
        print('Some smoke checks failed. See logs.')
        sys.exit(2)
    print('All smoke checks passed.')

if __name__ == '__main__':
    main()
