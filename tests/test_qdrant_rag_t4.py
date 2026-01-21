import pytest

try:
    from qdrant_client import QdrantClient
    from backend.qdrant_rag_t4 import QdrantFarmAgent
except Exception:
    QdrantClient = None
    QdrantFarmAgent = None


@pytest.mark.skipif(QdrantClient is None or QdrantFarmAgent is None, reason='qdrant-client not installed')
def test_qdrant_farmagent_init_and_collections():
    client = QdrantClient(':memory:')
    agent = QdrantFarmAgent(client)
    # Should not raise
    agent.init_collections()


@pytest.mark.skipif(QdrantClient is None or QdrantFarmAgent is None, reason='qdrant-client not installed')
def test_ingest_and_agentic_reasoning_smoke():
    client = QdrantClient(':memory:')
    agent = QdrantFarmAgent(client)
    agent.init_collections()
    # Use a very small synthetic set
    from PIL import Image
    imgs = [Image.new('RGB', (224,224), color='green'), Image.new('RGB', (224,224), color='yellow')]
    labels = ['case_a', 'case_b']
    # Ingest (should not raise)
    agent.ingest_knowledge_batch(imgs, labels, batch_size=1)
    res = agent.agentic_reasoning('farm_1', imgs[0], 'Yellowing spots')
    assert 'prompt' in res
