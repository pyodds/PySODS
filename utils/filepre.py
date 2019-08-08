import numpy as np
with open('inputFile.rtf', 'r') as f:
    lines = f.readlines()

lineNums = len(lines)
endLines = []

for i in range(lineNums):
    if 'val_acc=' in lines[i]:
        endLines.append(i)

ret = []

for el in endLines:
    sl = el - 1
    while sl > 0 and lines[sl][0] == '[':
        sl -= 1
    tmp = '['
    for l in range(sl+1, el):
        tmp += lines[l][1:-3] + ' '
    tmp = tmp[:-1] + ']\\'
    ret.append(tmp)

with open('outputFile.rtf', 'w') as f:
    for r in ret:
        f.write(r)
        f.write('\n')






filename='archienas0805log'
with open(filename+'.rtf') as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = np.asarray(np.asarray(lines)[i][1:-3].split(' '))

lines= np.asarray(lines)
lines=lines.astype(np.int)
np.save(filename+'data.npy', lines)


import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE

sns.set(color_codes=True)
x = np.load('archienas0805logdata.npy')
y = np.load('archiuncertaintyrl0805logdata.npy')

x_emb = TSNE(n_components=1).fit_transform(x)
y_emb = TSNE(n_components=1).fit_transform(y)
xx=x_emb.reshape(-1)
yy = y_emb.reshape(-1)


sns.kdeplot(xx, shade=True,label='enas');
sns.kdeplot(yy, shade=True,label='bayes');
