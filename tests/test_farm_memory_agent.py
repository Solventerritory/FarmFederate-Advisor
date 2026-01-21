import os
import tempfile

import pytest

# Skip these tests if qdrant-client is not installed in the current environment.
pytest.importorskip("qdrant_client")

from backend.farm_memory_agent import FarmMemoryAgent


def test_store_and_retrieve_similar_by_image():
    agent = FarmMemoryAgent(qdrant_url=":memory:", visual_dim=4, semantic_dim=3)
    agent.init_collection(recreate=True)

    img_emb1 = [0.1, 0.2, 0.3, 0.4]
    sem_emb1 = [0.01, 0.02, 0.03]
    pid1 = agent.store_report("Severe water stress observed", img_emb1, sem_emb1, farm_id="farm-A", metadata={"severity": "severe"})

    img_emb2 = [0.9, 0.8, 0.7, 0.6]
    sem_emb2 = [0.7, 0.6, 0.5]
    pid2 = agent.store_report("Early nutrient deficiency", img_emb2, sem_emb2, farm_id="farm-A")

    # Retrieve using same embedding as first point
    hits = agent.retrieve_similar_by_image(img_emb1, farm_id="farm-A", top_k=2)
    assert len(hits) >= 1
    assert hits[0]["payload"]["farm_id"] == "farm-A"

    # Test export to CSV
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    out = agent.export_reports_to_csv("farm-A", out_path=path)
    assert out == path
    assert os.path.exists(path)
    # Basic CSV content check
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    assert "Severe water stress" in content or "Early nutrient" in content
    os.remove(path)
