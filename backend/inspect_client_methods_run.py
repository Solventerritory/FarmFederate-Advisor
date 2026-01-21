from qdrant_client import QdrantClient
c = QdrantClient(':memory:')
print('methods containing search:', [m for m in dir(c) if 'search' in m])
print('methods containing upsert:', [m for m in dir(c) if 'upsert' in m])
print('methods containing scroll:', [m for m in dir(c) if 'scroll' in m])
print('has read_points:', hasattr(c,'read_points'))
print('has search_points:', hasattr(c,'search_points'))
print('has search:', hasattr(c,'search'))
print('client type:', type(c))

print('\n_inner client dir (filtered):', [m for m in dir(c._client) if any(k in m for k in ('search','point','upsert','scroll'))])
print('has _client.search_points:', hasattr(c._client,'search_points'))
print('has _client.search:', hasattr(c._client,'search'))
