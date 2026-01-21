from pathlib import Path
p=Path(r'c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate_Kaggle_Complete.py')
s=p.read_bytes()
# Find index of snippet 'Format prediction'
idx=s.find(b'Format prediction result')
print('idx', idx)
print(s[idx-80:idx+120])
print(repr(s[idx-80:idx+120]))
