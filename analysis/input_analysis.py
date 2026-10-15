import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from scipy.ndimage import uniform_filter
A='/mnt/user-data/uploads/LLIE_Color/assets/'
ids=['00739','00749','00750','00751','00752']
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
def C(x): l=rgb2lab(np.clip(x,0,1)); return l, np.hypot(l[...,1],l[...,2])
print('GT文件与拼图切片一致性:', np.abs(L(A+'gt/00739.png')*255-L(A+'gt/00739.png')*255).mean().round(2))
print('\n== 低光输入基本统计 (8bit 值)')
for i in ids:
    lo=L(A+f'low/{i}.png')*255
    print(i,'均值RGB=(%.1f,%.1f,%.1f)'%tuple(lo.reshape(-1,3).mean(0)),
          '像素<=2的比例 R/G/B=%.0f%%/%.0f%%/%.0f%%'%tuple((lo<=2).reshape(-1,3).mean(0)*100),
          '唯一灰阶数 B=%d'%len(np.unique(lo[...,2])))
print('\n== 信息是否还在输入里: 对输入做"理想"全局逐通道增益(线性光, 用GT求最优增益), 看色度能恢复多少')
print('  (不用任何网络, 只是 low*[gR,gG,gB]; 暗+饱和 = GT C>30 且 L<45)')
for i in ids:
    lo=L(A+f'low/{i}.png'); g=L(A+f'gt/{i}.png')
    ll,gl=lin(lo),lin(g)
    gain=(gl*ll).reshape(-1,3).sum(0)/(ll*ll).reshape(-1,3).sum(0)
    o=srgb(ll*gain)
    Lg,Cg=C(g); Lo,Co=C(o)
    mk=(Cg>30)&(Lg[...,0]<45)
    # 局部信噪比: 色度信号 (R-B) 对比 局部噪声
    lo8=lo*255
    sig=(lo8[...,0]-lo8[...,2])
    mu=uniform_filter(sig,7); noise=np.sqrt(np.maximum(uniform_filter(sig**2,7)-mu**2,0))
    print(f'{i} 增益RGB=({gain[0]:5.1f},{gain[1]:5.1f},{gain[2]:5.1f})  ΔE00={deltaE_ciede2000(Lg,Lo).mean():5.2f}  '
          f'色度比 全图={Co.mean()/Cg.mean():.2f} 暗+饱和={Co[mk].mean()/Cg[mk].mean() if mk.sum() else float("nan"):.2f}  '
          f'暗+饱和区 输入R-B={sig[mk].mean() if mk.sum() else 0:5.2f} 局部噪声σ={noise[mk].mean() if mk.sum() else 0:4.2f} (n={mk.sum()})')
print('\n== 楼梯图黄墙区域 (y270-380,x20-180) 输入原始值')
lo=L(A+'low/00739.png')*255; g=L(A+'gt/00739.png')*255
for nm,im in [('low',lo),('gt',g)]:
    p=im[270:380,20:180].reshape(-1,3); print(nm,'均值RGB=(%.2f,%.2f,%.2f) std=(%.2f,%.2f,%.2f) B通道=0比例=%.0f%%'%(*p.mean(0),*p.std(0),(p[:,2]==0).mean()*100))
