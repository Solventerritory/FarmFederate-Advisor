import os, json, requests
from datetime import datetime
ROOT = os.path.dirname(os.path.dirname(__file__))
EXT = os.path.join(ROOT, 'external_repos')
OUT = os.path.join(ROOT, 'results', 'github_clone_summary.json')

print('Scanning external_repos for clones...')
clones = []
if os.path.exists(EXT):
    for name in os.listdir(EXT):
        p = os.path.join(EXT, name)
        if os.path.isdir(p):
            print('Found clone dir:', name)
            clones.append({'name': name, 'path': p})

# Load API suggestions if present
api_file = os.path.join(ROOT, 'results', 'github_api_query_keys.json')
suggestions = {}
if os.path.exists(api_file):
    print('Loading API suggestions from', api_file)
    with open(api_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for q in data.get('queries', []):
        suggestions[q['key']] = q.get('suggestions', [])

# Check reachability of suggestion URLs
for key, items in suggestions.items():
    print('Checking suggestions for', key)
    for it in items:
        url = it['html_url']
        try:
            r = requests.head(url, allow_redirects=True, timeout=6)
            it['status'] = r.status_code
            print(' ', url, it['status'])
        except Exception as e:
            print(' ', url, 'ERR', e)
            it['status'] = 'ERR'

out = {'timestamp': datetime.utcnow().isoformat() + 'Z', 'cloned': clones, 'suggestions': suggestions}

os.makedirs(os.path.join(ROOT,'results'), exist_ok=True)
print('Writing summary to', OUT)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('Wrote', OUT)
