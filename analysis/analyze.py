import numpy as np, glob
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
A='/mnt/user-data/uploads/LLIE_Color/assets/'
W='/tmp/claude-0/work/'
gt={'00739':np.load(W+'gt_stair.npy'),'00749':np.load(W+'gt_pool0.npy'),'00750':np.load(W+'gt_pool1.npy'),
    '00751':np.load(W+'gt_pool2.npy'),'00752':np.load(W+'gt_pool3.npy')}
methods={'CWNet':'cwnet/low{}.png','HVI':'hvi/{}.png','Retinexformer':'retinexformer/{}.png','Diff-Retinex':'diff-retinex/{}.png','ReDDiT':'reddit/{}.png'}
def load(m,i): return np.asarray(Image.open(A+methods[m].format(i)).convert('RGB')).astype(float)/255
def lch(x):
    L=rgb2lab(x); C=np.hypot(L[...,1],L[...,2]); h=np.degrees(np.arctan2(L[...,2],L[...,1])); return L,C,h
def psnr(a,b): return 10*np.log10(1/np.mean((a-b)**2))
print('== 逐图指标: PSNR | ΔE00 | 色度比 C_out/C_gt | 亮度差 ΔL | 全局3x3颜色矩阵校正后 ΔE00 | 校正后仍剩余比例')
rows=[]
for i,g in gt.items():
    g=g/255; Lg,Cg,hg=lch(g)
    for m in methods:
        o=load(m,i); Lo,Co,ho=lch(o)
        dE=deltaE_ciede2000(Lg,Lo).mean()
        X=np.c_[o.reshape(-1,3),np.ones(o.size//3)]; M,_,_,_=np.linalg.lstsq(X,g.reshape(-1,3),rcond=None)
        oc=np.clip(X@M,0,1).reshape(g.shape); dEc=deltaE_ciede2000(Lg,rgb2lab(oc)).mean()
        cr=Co.mean()/Cg.mean(); dL=(Lo[...,0]-Lg[...,0]).mean()
        print(f'{i} {m:14s} PSNR={psnr(o,g):5.2f} ΔE={dE:5.2f} C比={cr:4.2f} ΔL={dL:+5.1f} 3x3校正后ΔE={dEc:5.2f} 剩余={dEc/dE:4.0%}')
        rows.append((i,m,dE,cr))
print()
print('== 饱和度分层: 按GT色度分组, 输出色度/GT色度 (00739楼梯图), 以及按GT亮度分组')
g=gt['00739']/255; Lg,Cg,hg=lch(g)
cb=[0,5,10,20,30,45,100]
print('GT色度区间   ', ' '.join(f'{cb[k]:>3}-{cb[k+1]:<3}' for k in range(len(cb)-1)))
for m in methods:
    o=load(m,'00739'); Lo,Co,ho=lch(o)
    s=[]
    for k in range(len(cb)-1):
        msk=(Cg>=cb[k])&(Cg<cb[k+1]); s.append(f'{Co[msk].mean()/Cg[msk].mean():7.2f}' if msk.sum()>200 else '    -  ')
    print(f'{m:14s}',' '.join(s))
print()
print('== 全部5张图合并: 高饱和(GT C>30)且暗(GT L<45)像素的色度保持率 vs 其它像素')
for m in methods:
    a=[];b=[]
    for i,gg in gt.items():
        Lg,Cg,_=lch(gg/255); Lo,Co,_=lch(load(m,i))
        mk=(Cg>30)&(Lg[...,0]<45); a.append((Co[mk].sum(),Cg[mk].sum(),mk.sum())); nm=~mk; b.append((Co[nm].sum(),Cg[nm].sum()))
    a=np.array(a);b=np.array(b)
    print(f'{m:14s} 暗+饱和: {a[:,0].sum()/a[:,1].sum():.2f} (n={int(a[:,2].sum())})   其它: {b[:,0].sum()/b[:,1].sum():.2f}')
print()
print('== 同一GT的两张不同低光输入 (00749 vs 00750; 00751 vs 00752): 两个输出之间的ΔE00 和 平均色相/色度')
for m in methods:
    out=[]
    for p,q in [('00749','00750'),('00751','00752')]:
        a=load(m,p);b=load(m,q); La=rgb2lab(a);Lb=rgb2lab(b)
        d=deltaE_ciede2000(La,Lb).mean()
        ma=La.reshape(-1,3).mean(0); mb=Lb.reshape(-1,3).mean(0)
        out.append(f'{p}/{q}: ΔE={d:5.2f}  均值a*b* ({ma[1]:+.1f},{ma[2]:+.1f}) vs ({mb[1]:+.1f},{mb[2]:+.1f})')
    print(f'{m:14s}', ' | '.join(out))
for p in ['00749','00751']:
    Lg=rgb2lab(gt[p]/255).reshape(-1,3).mean(0); print('GT',p,'均值a*b* (%+.1f,%+.1f)'%(Lg[1],Lg[2]))
