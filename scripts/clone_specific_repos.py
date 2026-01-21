import os, subprocess
ROOT = os.path.dirname(os.path.dirname(__file__))
repos = [
    'https://github.com/pratikkayal/PlantDoc-Dataset',
    'https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset',
    'https://github.com/helyne/drought_detection',
    'https://github.com/MLMasters/DroughtDetection',
    'https://github.com/WuZhuoran/Plant_Seedlings_Classification',
    'https://github.com/tectal/Plant-Seedlings-Classification',
]
for repo in repos:
    name = repo.split('github.com/')[-1].replace('/','_')
    target = os.path.join(ROOT,'external_repos',name)
    if os.path.exists(target) and os.listdir(target):
        print('Already present', target)
        continue
    print('Cloning', repo)
    try:
        res = subprocess.run(['git','clone','--depth','1',repo,target], capture_output=True, text=True, timeout=180)
        if res.returncode == 0:
            print('OK', repo)
        else:
            print('Failed', repo, res.stderr[:400])
    except Exception as e:
        print('Exception', repo, e)
