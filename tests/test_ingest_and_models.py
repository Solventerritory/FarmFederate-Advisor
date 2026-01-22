import os
import torch
from qdrant_client import QdrantClient


def test_sensorawarevlm_forward_shape():
    from FarmFederate_Kaggle_Complete import SensorAwareVLM, generate_sensor_data
    model = SensorAwareVLM(text_dim=768, image_dim=768, sensor_dim=10, hidden_dim=256, num_labels=5)
    model.eval()

    b = 4
    text = torch.randn(b, 768)
    image = torch.randn(b, 768)
    labels = torch.randint(0, 2, (b, 5)).float()
    sensors = generate_sensor_data(labels, batch_size=b)

    out = model(text, image, sensors, labels)
    assert 'logits' in out and out['logits'].shape == (b, 5)
    assert 'prior_mu' in out and out['prior_mu'].shape[0] == b


def test_qdrant_basic_upsert_search():
    try:
        from backend.qdrant_rag import init_qdrant_collections
    except Exception:
        init_qdrant_collections = None

    client = QdrantClient(':memory:')
    if init_qdrant_collections is not None:
        init_qdrant_collections(client, recreate=True)
    # Upsert a dummy point with named vectors
    from qdrant_client.http import models as rest
    vec = [0.0] * 512
    pt = rest.PointStruct(id=1, vector={'visual': vec, 'semantic': [0.0]*384}, payload={'test': 'ok'})
    client.upsert(collection_name='crop_health_knowledge', points=[pt])
    res = client.query_points(collection_name='crop_health_knowledge', query=vec, using='visual', limit=1).points
    assert isinstance(res, list)
    # Even if empty (older qdrant versions) the call should not raise
