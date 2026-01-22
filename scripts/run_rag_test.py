import sys
import os
# Ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from fastapi.testclient import TestClient
from backend import server

client = TestClient(server.app)

print('Calling /rag (JSON, no image)')
resp = client.post('/rag', json={'description': 'yellow leaves'})
print('status', resp.status_code)
print(resp.json())

print('\nCalling /predict (text)')
resp2 = client.post('/predict', json={'text': 'leaves yellowing'})
print('status', resp2.status_code)
print(resp2.json())
