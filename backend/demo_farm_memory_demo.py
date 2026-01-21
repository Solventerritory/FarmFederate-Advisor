"""Small demo script to exercise FarmMemoryAgent for CI and smoke tests.

This script uses in-memory Qdrant to store three sample reports and performs a
retrieve operation scoped to a farm_id. It exits with non-zero status when an
error occurs to fail CI.
"""
import os
import sys
# Ensure repository root is on sys.path so this script can be run directly
# (python backend/demo_farm_memory_demo.py) or as a module (python -m backend.demo_farm_memory_demo)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.farm_memory_agent import FarmMemoryAgent
import numpy as np
import pprint
import sys


def main():
    try:
        print('Starting FarmMemoryAgent demo (in-memory Qdrant)')
        agent = FarmMemoryAgent(qdrant_url=':memory:')
        agent.init_collection(recreate=True)

        emb_base = np.random.RandomState(42).randn(agent._visual_dim)
        ids = []
        ids.append(agent.store_report('Severe water stress - leaves drooping', emb_base.tolist(), np.random.rand(agent._semantic_dim).tolist(), farm_id='demo-farm-1', metadata={'severity':'severe'}))
        ids.append(agent.store_report('Early signs of nutrient deficiency - yellowing', (emb_base*0.9).tolist(), np.random.rand(agent._semantic_dim).tolist(), farm_id='demo-farm-1', metadata={'severity':'medium'}))
        ids.append(agent.store_report('Possible pest risk - small holes on leaves', (emb_base*1.1).tolist(), np.random.rand(agent._semantic_dim).tolist(), farm_id='demo-farm-1', metadata={'severity':'low'}))

        print('\nStored sample reports with ids:', ids)

        hits = agent.retrieve_similar_by_image(emb_base.tolist(), farm_id='demo-farm-1', top_k=3)
        print('\nRetrieved hits:')
        pprint.pprint(hits)

        out_csv = agent.export_reports_to_csv('demo-farm-1', out_path='results/demo_farm_1_history.csv')
        print('\nExported CSV to', out_csv)
        print('\nFarmMemoryAgent demo completed successfully')
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('Demo FAILED:', e)
        return 2


if __name__ == '__main__':
    sys.exit(main())
