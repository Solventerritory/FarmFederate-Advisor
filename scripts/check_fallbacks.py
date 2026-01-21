import ast, requests, sys
p='c:/Users/USER_HP/Desktop/FarmFederate/FarmFederate_Kaggle_Complete.py'
s=open(p,'r',encoding='utf-8').read()
mod=ast.parse(s)
fs=None
for node in mod.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if getattr(t,'id',None)=='FALLBACK_SOURCES':
                fs=ast.literal_eval(node.value)
                break
    if fs is not None:
        break
if fs is None:
    print('FALLBACK_SOURCES not found'); sys.exit(1)

print('Checking', sum(len(v) for v in fs.values()), 'candidate URLs across', len(fs), 'keys')
for key, urls in fs.items():
    for url in urls:
        try:
            r = requests.head(url, allow_redirects=True, timeout=5)
            code = r.status_code
        except Exception as e:
            code = f'ERR:{type(e).__name__}'
        print(f'{key}\t{code}\t{url}')
