import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
A='/mnt/user-data/uploads/LLIE_Color/assets/'
ids=['00739','00749','00750','00751','00752']
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
methods={'CWNet':'cwnet/low{}.png','HVI':'hvi/{}.png','Retinexformer':'retinexformer/{}.png','Diff-Retinex':'output/{}.png','ReDDiT':'reddit/{}.png'}
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
print('== 色偏传递: 输入只做"单一标量增益"(不校白平衡)后相对GT的 a*,b* 偏差  vs  各方法输出相对GT的 a*,b* 偏差')
X=[];Y={m:[] for m in methods}
for i in ids:
    lo=L(A+f'low/{i}.png'); g=L(A+f'gt/{i}.png')
    k=lin(g).mean()/lin(lo).mean(); s=srgb(lin(lo)*k)
    lg=rgb2lab(g).reshape(-1,3).mean(0); ls=rgb2lab(s).reshape(-1,3).mean(0)
    din=ls[1:]-lg[1:]; X.append(din)
    line=f'{i} 输入色偏 Δa*={din[0]:+5.1f} Δb*={din[1]:+5.1f} |'
    for m,p in methods.items():
        lo_=rgb2lab(L(A+p.format(i))).reshape(-1,3).mean(0); d=lo_[1:]-lg[1:]; Y[m].append(d)
        line+=f' {m[:6]}({d[0]:+4.1f},{d[1]:+4.1f})'
    print(line)
X=np.array(X)
print('\n各方法输出色偏与输入色偏的相关系数 (a*, b*) 和 回归斜率(传递比例):')
for m in methods:
    y=np.array(Y[m])
    r=[np.corrcoef(X[:,j],y[:,j])[0,1] for j in range(2)]
    s=[np.polyfit(X[:,j],y[:,j],1)[0] for j in range(2)]
    print(f'{m:14s} r=({r[0]:+.2f},{r[1]:+.2f})  斜率=({s[0]:+.2f},{s[1]:+.2f})')
print('\n== Diff-Retinex vs ReDDiT: 误差中 全局部分 vs 局部部分')
for m in ['Diff-Retinex','ReDDiT','Retinexformer','HVI']:
    a=[];b=[];gl=[]
    for i in ids:
        o=L(A+methods[m].format(i)); g=L(A+f'gt/{i}.png'); Lg=rgb2lab(g)
        dE=deltaE_ciede2000(Lg,rgb2lab(o)).mean()
        Xm=np.c_[o.reshape(-1,3),np.ones(o.size//3)]; M=np.linalg.lstsq(Xm,g.reshape(-1,3),rcond=None)[0]
        dEc=deltaE_ciede2000(Lg,rgb2lab(np.clip(Xm@M,0,1).reshape(g.shape))).mean()
        a.append(dE);b.append(dEc); gl.append(o.mean()/g.mean())
    print(f'{m:14s} 原始ΔE={np.mean(a):5.2f}  全局3x3校正后={np.mean(b):4.2f}  全局部分占比={1-np.mean(b)/np.mean(a):.0%}  亮度比(输出/GT)逐图='+' '.join(f'{v:.2f}' for v in gl))
