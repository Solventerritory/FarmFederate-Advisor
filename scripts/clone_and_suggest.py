import ast, requests, subprocess, os, json, time
from datetime import datetime

MAIN = r'c:/Users/USER_HP/Desktop/FarmFederate/FarmFederate_Kaggle_Complete.py'
OUT_MANIFEST = 'results/dataset_discovery_manifest.json'
SUGGESTIONS_OUT = 'results/fallback_suggestions.json'
DEST_ROOT = 'external_repos'

# Load FALLBACK_SOURCES from main file
s = open(MAIN,'r',encoding='utf-8').read()
mod = ast.parse(s)
fs = None
for node in mod.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if getattr(t,'id',None)=='FALLBACK_SOURCES':
                fs = ast.literal_eval(node.value)
                break
    if fs is not None:
        break
if fs is None:
    raise SystemExit('FALLBACK_SOURCES not found')

os.makedirs('results', exist_ok=True)
manifest = {}
suggestions = {}

print('Scanning FALLBACK_SOURCES:')
for key, urls in fs.items():
    for url in urls:
        entry = {'timestamp': datetime.utcnow().isoformat()+'Z','url': url}
        try:
            r = requests.head(url, allow_redirects=True, timeout=8)
            entry['status_code'] = r.status_code
        except Exception as e:
            entry['status_code'] = 'ERR'
            entry['error'] = str(e)
        manifest.setdefault(key, []).append(entry)

# Try cloning reachable GitHub repos
os.makedirs(DEST_ROOT, exist_ok=True)
for key, records in manifest.items():
    for r in records:
        url = r['url']
        code = r.get('status_code')
        if isinstance(code,int) and code == 200 and 'github.com' in url:
            # parse owner/repo
            import re
            m = re.match(r'https?://github.com/([^/]+/[^/]+)', url)
            if not m:
                continue
            repo_root = 'https://github.com/' + m.group(1)
            repo_name = m.group(1).replace('/','_')
            target_dir = os.path.join(DEST_ROOT, repo_name)
            if os.path.exists(target_dir) and os.listdir(target_dir):
                print(f'Skipping already-present {repo_root}')
                manifest[key].append({'timestamp': datetime.utcnow().isoformat()+'Z','url': repo_root, 'status': 'already_present','detail': target_dir})
                continue
            print(f'Cloning {repo_root} -> {target_dir}')
            cmd = ['git','clone','--depth','1',repo_root,target_dir]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print('clone success')
                manifest[key].append({'timestamp': datetime.utcnow().isoformat()+'Z','url': repo_root, 'status': 'clone_success', 'detail': target_dir})
            else:
                manifest[key].append({'timestamp': datetime.utcnow().isoformat()+'Z','url': repo_root, 'status': 'clone_failed', 'detail': res.stderr[:1000]})

# For 404 github URLs, try GitHub search to find likely replacements
GITHUB_API = 'https://api.github.com/search/repositories'
for key, records in list(manifest.items()):
    for r in records:
        url = r['url']
        code = r.get('status_code')
        if ('github.com' in url) and (code == 404):
            # extract repo name component
            try:
                tgt = url.rstrip('/').split('/')[-1]
                q = f'{tgt} in:name'
                print(f"Searching GitHub for '{tgt}' (key={key})")
                resp = requests.get(GITHUB_API, params={'q': q, 'per_page': 5}, timeout=8)
                if resp.status_code == 200:
                    items = resp.json().get('items', [])
                    sug = []
                    for it in items[:3]:
                        sug.append(it['html_url'])
                    if sug:
                        print(f' -> Found suggestions: {sug}')
                        suggestions.setdefault(key, []).extend(sug)
                else:
                    print(f'GitHub search failed: {resp.status_code}')
            except Exception as e:
                print('GitHub search exception', e)

# For 404 non-GitHub URLs try some heuristics (e.g., search for last path component)
for key, records in list(manifest.items()):
    for r in records:
        url = r['url']
        code = r.get('status_code')
        if (not 'github.com' in url) and (code == 404 or str(code).startswith('ERR')):
            # extract last token
            tok = url.rstrip('/').split('/')[-1]
            # try GH search for token as fallback
            try:
                q = f'{tok} in:name'
                resp = requests.get(GITHUB_API, params={'q': q, 'per_page': 5}, timeout=8)
                if resp.status_code == 200:
                    items = resp.json().get('items', [])
                    sug = [it['html_url'] for it in items[:3]]
                    if sug:
                        suggestions.setdefault(key, []).extend(sug)
            except Exception:
                pass

# Write outputs
with open(OUT_MANIFEST,'w',encoding='utf-8') as mf:
    json.dump(manifest,mf,indent=2)
with open(SUGGESTIONS_OUT,'w',encoding='utf-8') as sf:
    json.dump(suggestions,sf,indent=2)

print('Done. Wrote', OUT_MANIFEST, 'and', SUGGESTIONS_OUT)
print('Suggestions summary:')
for k,v in suggestions.items():
    print(k, '->', v[:3])
