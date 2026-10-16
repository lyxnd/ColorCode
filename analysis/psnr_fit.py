import numpy as np
from PIL import Image
A='/mnt/user-data/uploads/LLIE_Color/assets/'
ids=['00739','00749','00750','00751','00752']
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
methods={'CWNet':'cwnet/low{}.png','HVI':'hvi/{}.png','Retinexformer':'retinexformer/{}.png','Diff-Retinex':'output/{}.png','ReDDiT':'reddit/{}.png'}
ps=lambda a,b:10*np.log10(1/np.mean((a-b)**2))
print('方法            原始PSNR  仅亮度标量校正  逐通道增益校正  全局3x3+偏置校正')
for m,p in methods.items():
    r=[[],[],[],[]]
    for i in ids:
        o=L(A+p.format(i)); g=L(A+f'gt/{i}.png')
        r[0].append(ps(o,g))
        r[1].append(ps(np.clip(o*g.mean()/o.mean(),0,1),g))
        r[2].append(ps(np.clip(o*g.reshape(-1,3).mean(0)/o.reshape(-1,3).mean(0),0,1),g))
        X=np.c_[o.reshape(-1,3),np.ones(o.size//3)]; M=np.linalg.lstsq(X,g.reshape(-1,3),rcond=None)[0]
        r[3].append(ps(np.clip(X@M,0,1).reshape(g.shape),g))
    print(f'{m:14s} '+'  '.join(f'{np.mean(x):12.2f}' for x in r))
