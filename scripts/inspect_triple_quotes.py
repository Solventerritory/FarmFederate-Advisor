from pathlib import Path
p = Path(r'c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate_Kaggle_Complete.py')
s = p.read_text()
count = 0
for i, line in enumerate(s.splitlines(),1):
    if '"""' in line:
        count += line.count('"""')
        print(i, line.strip(), 'cumulative', count)
print('final count', count)
