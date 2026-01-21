import ast, os, requests, json, re, time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'FarmFederate_Kaggle_Complete.py')
OUT = os.path.join(ROOT, 'results', 'github_api_suggestions.json')

s = open(SRC, 'r', encoding='utf-8').read()
mod = ast.parse(s)
fallbacks = None
for node in mod.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if getattr(t, 'id', None) == 'FALLBACK_SOURCES':
                fallbacks = ast.literal_eval(node.value)
                break
    if fallbacks is not None:
        break
if fallbacks is None:
    raise SystemExit('FALLBACK_SOURCES not found')

results = {'timestamp': datetime.utcnow().isoformat() + 'Z', 'queries': []}

print('Beginning GitHub API suggestion queries...')
for key, urls in fallbacks.items():
    print('Processing key:', key)
    for url in urls:
        if 'github.com' not in url:
            continue
        base = re.match(r'https?://github.com/([^/]+/[^/]+)', url)
        base_url = base.group(0) if base else url
        try:
            r = requests.head(base_url, allow_redirects=True, timeout=6)
            status = r.status_code
        except Exception as e:
            status = f'ERR:{type(e).__name__}'
        if not str(status).startswith('2'):
            # search GitHub API using key and last path component
            name_part = (url.split('/')[-1] or key)
            query = f'{key} {name_part}'
            api_url = f'https://api.github.com/search/repositories?q={requests.utils.requote_uri(query)}&per_page=5'
            try:
                time.sleep(0.5)
                r2 = requests.get(api_url, timeout=8, headers={'Accept':'application/vnd.github.v3+json','User-Agent':'FarmFederateBot/1.0'})
                if r2.status_code == 200:
                    items = r2.json().get('items', [])
                    suggestions = [{'full_name': it['full_name'], 'html_url': it['html_url'], 'stargazers_count': it.get('stargazers_count',0)} for it in items]
                else:
                    suggestions = []
            except Exception as e:
                suggestions = []
            results['queries'].append({'key': key, 'orig': base_url, 'status': status, 'api_query': query, 'suggestions': suggestions})

os.makedirs(os.path.join(ROOT,'results'), exist_ok=True)
with open(OUT,'w',encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('Wrote', OUT)
