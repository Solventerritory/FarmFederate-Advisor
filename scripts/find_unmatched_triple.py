from pathlib import Path
p = Path(r'c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate_Kaggle_Complete.py')
s = p.read_text()
state = False
open_lines = []
for i, line in enumerate(s.splitlines(),1):
    if '"""' in line:
        # count occurrences on line
        for _ in range(line.count('"""')):
            if not state:
                state = True
                open_lines.append(i)
                print(f'OPEN at {i}: {repr(line.strip())[:120]}')
            else:
                print(f'CLOSE at {i}: {repr(line.strip())[:120]}')
                state = False
                open_lines.pop()
print('state open?', state)
print('currently unmatched opens:', open_lines)
if open_lines:
    print('First unmatched open line:', open_lines[0])
    # print surrounding context
    L = open_lines[0]
    lines = s.splitlines()
    for j in range(max(1, L-5), min(len(lines), L+6)):
        print(j, repr(lines[j-1]))
