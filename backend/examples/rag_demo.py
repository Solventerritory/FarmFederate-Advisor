"""Small demo to initialize collections, perform a small ingestion (dry-run) and run a diagnosis.

Usage:
    python rag_demo.py --init --ingest --image IMAGE_PATH --desc "yellow spots on leaves" --farm FARM1 --plant PLANT1
"""
import argparse
from qdrant_utils import get_qdrant_client, initialize_crop_health_collection, initialize_farm_session_collection
from ingest_qdrant import ingest_datasets, Embedders
from agent_rag import rag_diagnose, store_session_memory, check_session_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--ingest', action='store_true')
    parser.add_argument('--image', type=str)
    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--farm', type=str, default='demo_farm')
    parser.add_argument('--plant', type=str, default='plant_001')
    args = parser.parse_args()

    client = get_qdrant_client()
    if args.init:
        initialize_crop_health_collection(client)
        initialize_farm_session_collection(client)

    emb = Embedders()
    if args.ingest:
        ingest_datasets(client, collection_name='crop_health_knowledge', data_roots=['data/PlantVillage', 'data/IP102'], embedders=emb)

    if args.image:
        out = rag_diagnose(client, args.image, args.desc, embedders=emb)
        print('\nLLM output:\n', out['llm_output'])
        # store session
        payload = store_session_memory(client, farm_id=args.farm, plant_id=args.plant, diagnosis_text=out['llm_output'])
        print('Stored session:', payload)
        print('Recent history:', check_session_history(client, args.farm, args.plant))

if __name__ == '__main__':
    main()
