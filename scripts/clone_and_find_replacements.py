import ast, os, subprocess, requests, json, re, time
from datetime import datetime
from urllib.parse import quote_plus

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'FarmFederate_Kaggle_Complete.py')
OUT = os.path.join(ROOT, 'results', 'github_clone_and_suggestions.json')
os.makedirs(os.path.join(ROOT, 'external_repos'), exist_ok=True)
result = {'timestamp': datetime.utcnow().isoformat() + 'Z', 'cloned': [], 'clone_failures': [], 'suggestions': {}}

# Parse FALLBACK_SOURCES from the script using ast
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

print('Found', sum(len(v) for v in fallbacks.values()), 'URLs across', len(fallbacks), 'keys')

HEAD_TIMEOUT = 6
for key, urls in fallbacks.items():
    for url in urls:
        if 'github.com' not in url:
            continue
        # Normalize owner/repo if possible
        m = re.match(r'https?://github.com/([^/]+/[^/]+)', url)
        base = m.group(0) if m else url
        try:
            r = requests.head(base, allow_redirects=True, timeout=HEAD_TIMEOUT)
            status = r.status_code
        except Exception as e:
            status = f'ERR:{type(e).__name__}'
        print(key, status, base)
        if str(status).startswith('2'):
            # attempt clone shallow
            repo_root = base.rstrip('/')
            repo_name = repo_root.split('github.com/')[-1].replace('/', '_')
            target_dir = os.path.join(ROOT, 'external_repos', repo_name)
            if os.path.exists(target_dir) and os.listdir(target_dir):
                print('Already cloned', target_dir)
                result['cloned'].append({'key': key, 'url': repo_root, 'target': target_dir, 'status': 'already_present'})
                continue
            cmd = ['git', 'clone', '--depth', '1', repo_root, target_dir]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if res.returncode == 0:
                    print('Cloned', repo_root, '->', target_dir)
                    result['cloned'].append({'key': key, 'url': repo_root, 'target': target_dir, 'status': 'cloned'})
                else:
                    print('Clone failed for', repo_root, res.stderr[:400])
                    result['clone_failures'].append({'key': key, 'url': repo_root, 'error': res.stderr[:2000]})
            except Exception as e:
                print('Clone exception for', repo_root, e)
                result['clone_failures'].append({'key': key, 'url': repo_root, 'error': str(e)})
        else:
            # record suggestion search for 404 / errors
            query = quote_plus(key + ' ' + (url.split('/')[-1] if '/' in url else ''))
            search_url = f'https://github.com/search?q={query}&type=repositories'
            print('Searching GitHub for replacements:', search_url)
            try:
                time.sleep(1)  # avoid aggressive scraping
                r = requests.get(search_url, timeout=10, headers={'User-Agent':'FarmFederateBot/1.0'})
                if r.status_code == 200:
                    # extract repository links using regex fallback
                    links = re.findall(r'href="/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"', r.text)
                    unique = []
                    for l in links:
                        if l not in unique:
                            unique.append(l)
                    suggestions = ['https://github.com/' + u for u in unique[:6]]
                    print('Found suggestions:', suggestions[:3])
                    result['suggestions'].setdefault(key, []).append({'orig': base, 'status': status, 'suggestions': suggestions})
                else:
                    result['suggestions'].setdefault(key, []).append({'orig': base, 'status': status, 'suggestions': [], 'http_status': r.status_code})
            except Exception as e:
                result['suggestions'].setdefault(key, []).append({'orig': base, 'status': status, 'suggestions': [], 'error': str(e)})

# Save
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print('\nWrote', OUT)
