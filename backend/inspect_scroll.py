import inspect
from qdrant_client import QdrantClient
c = QdrantClient(':memory:')
print('scroll signature:', inspect.signature(c.scroll))
print('\nSource snippet:\n')
src = inspect.getsource(c.scroll)
print('\n'.join(src.splitlines()[:200]))
