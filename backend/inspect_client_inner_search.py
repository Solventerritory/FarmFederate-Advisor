import inspect
from qdrant_client import QdrantClient
c = QdrantClient(':memory:')
print('search signature:', inspect.signature(c._client.search))
print('\nSource snippet:\n')
src = inspect.getsource(c._client.search)
print('\n'.join(src.splitlines()[:200]))
