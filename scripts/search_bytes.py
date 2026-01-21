from pathlib import Path
p=Path(r'c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate_Kaggle_Complete.py')
s=p.read_bytes()
needle=b'\n"""\n    try:'
print('pos', s.find(needle))
print(s[s.find(needle)-40:s.find(needle)+80])
