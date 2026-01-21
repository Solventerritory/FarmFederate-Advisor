import tokenize
from pathlib import Path
p=Path(r'c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate_Kaggle_Complete.py')
s=p.read_text()
from io import StringIO
f=StringIO(s)
for tok in tokenize.generate_tokens(f.readline):
    ttype, tstring, start, end, line = tok
    if tok.type == tokenize.STRING:
        print('STRING at', start, 'len', len(tstring.splitlines()), repr(tstring[:120]))
