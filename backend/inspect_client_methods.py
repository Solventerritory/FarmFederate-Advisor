from qdrant_client import QdrantClient
c = QdrantClient(':memory:')
print('has search:', hasattr(c, 'search'))
print('has search_points:', hasattr(c, 'search_points'))
print('dir methods (filtered):', [m for m in dir(c) if any(k in m for k in ('search','upsert','scroll','read'))])
print('\nfull dir snippet:\n', '\n'.join(dir(c)[:60]))
