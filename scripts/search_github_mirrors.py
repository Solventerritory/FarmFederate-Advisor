"""Search GitHub for mirrors for specific dataset queries and write results to results/github_mirrors_proposed.json"""
import requests, json, os
queries = {
    'Drought_Detection': ['drought detection dataset', 'drought dataset', 'drought plant dataset'],
    'Crop_Disease': ['plant disease dataset', 'crop disease dataset', 'plant disease images']
}
headers = {'Accept':'application/vnd.github.v3+json','User-Agent':'FarmFederate-Agent'}
# Use GITHUB_TOKEN if available to avoid rate limits
token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
if token:
    headers['Authorization'] = f'token {token}'

out = {}
for key, qs in queries.items():
    out[key] = []
    for q in qs:
        try:
            r = requests.get('https://api.github.com/search/repositories', params={'q': q, 'per_page': 8}, headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
            for item in data.get('items', []):
                out[key].append({
                    'full_name': item.get('full_name'),
                    'html_url': item.get('html_url'),
                    'stars': item.get('stargazers_count', 0),
                    'desc': item.get('description') or ''
                })
        except Exception as e:
            out[key].append({'error': str(e), 'query': q})

os.makedirs('results', exist_ok=True)
with open('results/github_mirrors_proposed.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('Wrote results/github_mirrors_proposed.json')
print('Summary:')
for k, v in out.items():
    print(k, len(v))
