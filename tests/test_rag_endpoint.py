from fastapi.testclient import TestClient
import pytest

from backend import server

try:
    from qdrant_client import QdrantClient
    from backend.qdrant_rag import init_qdrant_collections
except Exception:
    QdrantClient = None

@pytest.mark.skipif(QdrantClient is None, reason='qdrant-client not installed')
def test_rag_endpoint_in_memory():
    # create in-memory client and initialize collections
    client_mem = QdrantClient(':memory:')
    init_qdrant_collections(client_mem, recreate=True)

    # attach to server globals
    server.QDRANT_CLIENT = client_mem
    server.app.state.qdrant_client = client_mem

    with TestClient(server.app) as tc:
        resp = tc.post('/rag', json={'description': 'test rag'})
        assert resp.status_code == 200
        data = resp.json()
        assert 'result' in data
        # result should have prompt and retrieved list
        assert isinstance(data['result'], dict)
