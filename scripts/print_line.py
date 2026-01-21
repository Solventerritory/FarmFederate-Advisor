from pathlib import Path
p=Path(r'c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate_Kaggle_Complete.py')
lines=p.read_text().splitlines()
for i in range(1558,1574):
    print(i+1, repr(lines[i]))
