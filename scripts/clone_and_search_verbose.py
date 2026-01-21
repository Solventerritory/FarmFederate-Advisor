import ast, os, subprocess, requests, json, re, time, traceback
from datetime import datetime
from urllib.parse import quote_plus

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'FarmFederate_Kaggle_Complete.py')
OUT = os.path.join(ROOT, 'results', 'github_clone_and_suggestions_verbose.json')
os.makedirs(os.path.join(ROOT, 'external_repos'), exist_ok=True)
result = {'timestamp': datetime.utcnow().isoformat() + 'Z', 'entries': []}

try:
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
                print('Skipping non-github url for cloning check', key, url)
                continue
            entry = {'key': key, 'orig_url': url, 'head': None, 'clone': None, 'suggestions': []}
            try:
                m = re.match(r'https?://github.com/([^/]+/[^/]+)', url)
                base = m.group(0) if m else url
                print('\nChecking', key, base)
                r = requests.head(base, allow_redirects=True, timeout=HEAD_TIMEOUT)
                status = r.status_code
            except Exception as e:
                status = f'ERR:{type(e).__name__}'
                print('HEAD error for', base, e)
            entry['head'] = status

            if str(status).startswith('2'):
                repo_root = base.rstrip('/')
                repo_name = repo_root.split('github.com/')[-1].replace('/', '_')
                target_dir = os.path.join(ROOT, 'external_repos', repo_name)
                if os.path.exists(target_dir) and os.listdir(target_dir):
                    print('Already cloned:', target_dir)
                    entry['clone'] = {'status': 'already_present', 'target': target_dir}
                else:
                    cmd = ['git', 'clone', '--depth', '1', repo_root, target_dir]
                    print('Cloning', repo_root, '->', target_dir)
                    try:
                        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                        if res.returncode == 0:
                            print('Clone succeeded')
                            entry['clone'] = {'status': 'cloned', 'target': target_dir}
                        else:
                            print('Clone failed:', res.stderr[:300])
                            entry['clone'] = {'status': 'failed', 'error': res.stderr[:2000]}
                    except Exception as e:
                        print('Clone exception:', e)
                        entry['clone'] = {'status': 'exception', 'error': str(e)}
            else:
                # Search GitHub for replacements
                query = quote_plus(key + ' ' + (url.split('/')[-1] if '/' in url else ''))
                search_url = f'https://github.com/search?q={query}&type=repositories'
                print('Searching GitHub for replacements:', search_url)
                try:
                    time.sleep(1)  # polite pause
                    r = requests.get(search_url, timeout=10, headers={'User-Agent':'FarmFederateBot/1.0'})
                    if r.status_code == 200:
                        links = re.findall(r'href="/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"', r.text)
                        unique = []
                        for l in links:
                            if l not in unique:
                                unique.append(l)
                        suggestions = ['https://github.com/' + u for u in unique[:6]]
                        print('Suggestions:', suggestions[:4])
                        entry['suggestions'] = suggestions
                    else:
                        print('GitHub search returned', r.status_code)
                        entry['suggestions'] = []
                except Exception as e:
                    print('Search exception:', e)
                    entry['suggestions'] = []
            result['entries'].append(entry)

except Exception as e:
    print('Fatal error in script:', e)
    traceback.print_exc()

# Save results
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print('\nWrote', OUT)
