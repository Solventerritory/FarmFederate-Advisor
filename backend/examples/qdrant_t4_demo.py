"""Small demo showing QdrantFarmAgent usage (T4/Colab-friendly).

Run in Colab after installing dependencies:
!pip install -q qdrant-client pillow

Then in Python:
from qdrant_client import QdrantClient
from backend.qdrant_rag_t4 import QdrantFarmAgent, print_traceable_log
from PIL import Image

client = QdrantClient(':memory:')
agent = QdrantFarmAgent(client)
agent.init_collections()

# Ingest a tiny synthetic knowledge base
images = [Image.new('RGB', (224,224), color=c) for c in ['green','yellow','brown']]
labels = ['healthy_case','nitrogen_deficiency','water_stress']
agent.ingest_knowledge_batch(images, labels, batch_size=2)

# Store a simple farm history record using client upsert
from qdrant_client.http import models
p = models.PointStruct(id=1, vector=None, payload={'farm_id': 'KGP-25', 'diagnosis': 'water_stress', 'timestamp': 1234567890})
client.upsert(collection_name='farm_history', points=[p])

# Agentic reasoning
prompt_res = agent.agentic_reasoning('KGP-25', images[1], 'Leaves turning yellow with tips browning')
print(prompt_res['prompt'])

# Print a traceable log
print_traceable_log('KGP-25', 'Leaves turning yellow with tips browning', prompt_res['global_matches'], prompt_res['farm_history'])
"""

from qdrant_client import QdrantClient

try:
    from backend.qdrant_rag_t4 import QdrantFarmAgent, print_traceable_log
except Exception as e:
    raise

from PIL import Image


def demo():
    client = QdrantClient(':memory:')
    agent = QdrantFarmAgent(client)
    agent.init_collections()

    images = [Image.new('RGB', (224,224), color=c) for c in ['green', 'yellow', 'brown']]
    labels = ['healthy_case', 'nitrogen_deficiency', 'water_stress']

    print('Ingesting tiny knowledge base (memory-efficient batches)...')
    agent.ingest_knowledge_batch(images, labels, batch_size=2)

    print('Adding a farm history entry...')
    try:
        from qdrant_client.http import models
        p = models.PointStruct(id=1, vector=None, payload={'farm_id': 'KGP-25', 'diagnosis': 'water_stress', 'timestamp': 1234567890})
        client.upsert(collection_name='farm_history', points=[p])
    except Exception:
        print('Failed to write farm_history entry (client may not support PointStruct in this env)')

    print('Running agentic_reasoning to construct a traceable prompt...')
    res = agent.agentic_reasoning('KGP-25', images[1], 'Leaves turning yellow with tips browning')
    print('\n--- PROMPT ---\n')
    print(res['prompt'])

    print('\n--- TRACEABLE LOG ---\n')
    print_traceable_log('KGP-25', 'Leaves turning yellow with tips browning', res['global_matches'], res['farm_history'])


if __name__ == '__main__':
    demo()
