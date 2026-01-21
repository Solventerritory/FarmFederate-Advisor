import unittest
from unittest.mock import MagicMock
from backend.qdrant_rag import init_qdrant_collections, agentic_diagnose


class DummyClient:
    def __init__(self):
        self.collections = {}
    def create_collection(self, collection_name, vectors=None):
        self.collections[collection_name] = vectors
    def get_collection(self, collection_name, check_response=True):
        return self.collections.get(collection_name)
    def search(self, collection_name, query_vector, limit=3, vector_name=None, with_payload=False):
        # Return dummy hits
        class Hit:
            def __init__(self):
                self.id = 1
                self.payload = {'stress_type': 'water_stress', 'crop_name': 'wheat', 'severity': 'mild', 'agronomist_notes': 'water deficit'}
                self.score = 0.98
        return [Hit() for _ in range(limit)]

class DummyEmb:
    def embed_image(self, img):
        return [0.0]*512
    def embed_text(self, text):
        return [0.0]*384

class TestQdrantRAG(unittest.TestCase):
    def test_init_collections(self):
        import backend.qdrant_rag as qr
        if qr.QdrantClient is None or qr.rest is None:
            self.skipTest('qdrant-client not installed; skipping collection init test')
        c = DummyClient()
        init_qdrant_collections(c, recreate=True)
        self.assertIn('crop_health_knowledge', c.collections)
        self.assertIn('farm_session_memory', c.collections)

    def test_agentic_prompt_construction(self):
        c = DummyClient()
        emb = DummyEmb()
        # Use a small green image
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='green')
        res = agentic_diagnose(c, image=img, user_description='leaves yellow', emb=emb)
        self.assertIn('Historical Cases', res['prompt'])


if __name__ == '__main__':
    unittest.main()
