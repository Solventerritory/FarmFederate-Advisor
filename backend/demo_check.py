from backend.farm_memory_agent import FarmMemoryAgent
print('Imported FarmMemoryAgent')
try:
    a = FarmMemoryAgent(qdrant_url=':memory:')
    print('Instantiated')
    a.init_collection(recreate=True)
    print('init_collection succeeded')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Error:', e)
