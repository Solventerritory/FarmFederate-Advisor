import os
import csv
import pytest

pytest.importorskip('qdrant_client')

from backend.farm_memory_agent import FarmMemoryAgent


def test_demo_exports_csv(tmp_path):
    """Smoke test: create demo reports via FarmMemoryAgent and verify CSV export."""
    agent = FarmMemoryAgent(qdrant_url=':memory:')
    agent.init_collection(recreate=True)

    # Store three reports for demo-farm-1
    emb_base = [0.1] * agent._visual_dim
    ids = []
    ids.append(agent.store_report('Rep A', emb_base, [0.2] * agent._semantic_dim, farm_id='demo-farm-1'))
    ids.append(agent.store_report('Rep B', emb_base, [0.3] * agent._semantic_dim, farm_id='demo-farm-1'))
    ids.append(agent.store_report('Rep C', emb_base, [0.4] * agent._semantic_dim, farm_id='demo-farm-1'))

    out_csv = tmp_path / 'demo_farm_1_history.csv'
    path = agent.export_reports_to_csv('demo-farm-1', out_path=str(out_csv))

    assert os.path.exists(path), 'CSV not created'

    # Basic content checks
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) >= 3, f'Expected >=3 reports rows, got {len(rows)}'
    for r in rows:
        assert r.get('farm_id') == 'demo-farm-1'
        assert r.get('report') is not None

    # CSV header exactness
    assert set(reader.fieldnames) == set(['id', 'farm_id', 'report', 'timestamp', 'meta'])
