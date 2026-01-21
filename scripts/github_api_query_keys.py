import requests, json, time, sys
from datetime import datetime
keys = ['IP102','PlantDoc','Drought_Detection','Plant_Seedlings']
results = {'timestamp': datetime.utcnow().isoformat() + 'Z', 'queries': []}
for key in keys:
    q = key
    api_url = f'https://api.github.com/search/repositories?q={requests.utils.requote_uri(q)}&per_page=6'
    try:
        print('Querying GitHub API for', q)
        r = requests.get(api_url, timeout=8, headers={'Accept':'application/vnd.github.v3+json','User-Agent':'FarmFederateBot/1.0'})
        if r.status_code == 200:
            items = r.json().get('items', [])
            suggestions = [{'full_name': it['full_name'], 'html_url': it['html_url'], 'stars': it.get('stargazers_count',0)} for it in items]
        else:
            suggestions = []
            print('GitHub API returned', r.status_code, 'for', q)
    except Exception as e:
        print('Exception during API query for', q, e)
        suggestions = []
    results['queries'].append({'key': key, 'api_query': q, 'suggestions': suggestions})
    time.sleep(0.25)

with open('results/github_api_query_keys.json','w',encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('Wrote results/github_api_query_keys.json')