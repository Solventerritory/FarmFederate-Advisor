from qdrant_client import QdrantClient
import inspect
print('QdrantClient.recreate_collection signature:')
print(inspect.signature(QdrantClient.recreate_collection))
print('\nSource (first 200 lines):')
import inspect
src = inspect.getsource(QdrantClient.recreate_collection)
print('\n'.join(src.splitlines()[:200]))
