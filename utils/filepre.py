import numpy as np
filename='archiuncertaintyrl0808log10x'

with open(filename+'.rtf', 'r') as f:
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

with open('ar'+filename+'.rtf', 'w') as f:
    for r in ret:
        f.write(r)
        f.write('\n')






filename='ar'+'archiuncertaintyrl0808log10x'
with open(filename+'.rtf') as f:
    lines = f.readlines()

import numpy as np

for i in range(len(lines)):
    lines[i] = np.asarray(np.asarray(lines)[i][1:-3].split(' '))

lines= np.asarray(lines)
lines=lines.astype(np.int)
np.save(filename+'data.npy', lines)

random = np.random.randint(2, size=(3100, 78))
random[:,0]=np.random.randint(6, size=3100)
random[:,1]=np.random.randint(6, size=3100)
random[:,3]=np.random.randint(6, size=3100)
random[:,6]=np.random.randint(6, size=3100)
random[:,10]=np.random.randint(6, size=3100)
random[:,15]=np.random.randint(6, size=3100)
random[:,21]=np.random.randint(6, size=3100)
random[:,28]=np.random.randint(6, size=3100)
random[:,36]=np.random.randint(6, size=3100)
random[:,45]=np.random.randint(6, size=3100)
random[:,45]=np.random.randint(6, size=3100)
random[:,55]=np.random.randint(6, size=3100)
random[:,66]=np.random.randint(6, size=3100)

random_emb = TSNE(n_components=1).fit_transform(random)
rr = random_emb.reshape(-1)





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
z = np.load('ararchiuncertaintyrl0808log10xdata.npy')

x_emb = TSNE(n_components=1).fit_transform(x)
y_emb = TSNE(n_components=1).fit_transform(y)
z_emb = TSNE(n_components=1).fit_transform(z)
random_emb = TSNE(n_components=1).fit_transform(random)



xx = x_emb.reshape(-1)
yy = y_emb.reshape(-1)
zz = z_emb.reshape(-1)
rr = random_emb.reshape(-1)


np.save('xx.npy', xx)
np.save('yy.npy', yy)
np.save('zz.npy', zz)
np.save('rr.npy', rr)


sns.kdeplot(xx, shade=True,label='ENAS');
sns.kdeplot(yy, shade=True,label='CGNAS lambda=10');
sns.kdeplot(zz, shade=True,label='CGNAS lambda=1');
# sns.kdeplot(rr, shade=True,label='CGNAS lambda=0.1');

plt.show()