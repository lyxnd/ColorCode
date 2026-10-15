import numpy as np, glob, os
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
ids=['00739','00749','00750','00751','00752']
M={'HVI':'assets/perturb_out/hvi/{}_{}.png','Retinexformer':'assets/perturb_out/retinexformer/{}_{}.png',
   'Diff-Retinex':'assets/perturb_out/diff_retinex/{}_{}.png','ReDDiT':'assets/perturb_out/reddit/{}_{}_normal.png'}
def lab(p): return rgb2lab(np.asarray(Image.open(p).convert('RGB')).astype(float)/255)
# 参照: 扰动本身在 "输入经标量增益提亮后" 造成的 a*b* 变化 (完全不校正时的传递量)
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
ks=['warm','cool','green','dark','bright']
print('输入端参照(把扰动后的低光图直接按GT整体亮度提亮, 相对ref的变化, 即"100%传递"时的量):')
for k in ks:
    d=[]
    for i in ids:
        g=np.asarray(Image.open(f'assets/gt/{i}.png').convert('RGB')).astype(float)/255
        r=np.asarray(Image.open(f'assets/perturb/{i}_ref.png').convert('RGB')).astype(float)/255
        p=np.asarray(Image.open(f'assets/perturb/{i}_{k}.png').convert('RGB')).astype(float)/255
        s=lin(g).mean()/lin(r).mean()
        lr=rgb2lab(srgb(lin(r)*s)); lp=rgb2lab(srgb(lin(p)*s))
        d.append((lp-lr).reshape(-1,3).mean(0))
    d=np.mean(d,0); print(f'  {k:7s} ΔL={d[0]:+6.2f} Δa*={d[1]:+6.2f} Δb*={d[2]:+6.2f}')
print()
for m,pat in M.items():
    print(m)
    for k in ks:
        d=[]
        for i in ids:
            if not (os.path.exists(pat.format(i,'ref')) and os.path.exists(pat.format(i,k))): continue
            r=lab(pat.format(i,'ref')); p=lab(pat.format(i,k))
            dm=(p-r).reshape(-1,3).mean(0); d.append([*dm,deltaE_ciede2000(r,p).mean()])
        n=len(d); d=np.mean(d,0); print(f'  {k:7s} ΔL={d[0]:+6.2f} Δa*={d[1]:+6.2f} Δb*={d[2]:+6.2f} ΔE00={d[3]:5.2f} (n={n})')
print('\n各方法 ref 输入的结果 vs 原先跑出的结果 (检查推理一致性/随机性), ΔE00:')
old={'HVI':'assets/hvi/{}.png','Retinexformer':'assets/retinexformer/{}.png','Diff-Retinex':'assets/diff-retinex/{}.png','ReDDiT':'assets/reddit/{}.png'}
for m in M:
    v=[]
    for i in ids:
        if not os.path.exists(M[m].format(i,'ref')): continue
        a=np.asarray(Image.open(M[m].format(i,'ref')).convert('RGB')).astype(float)/255
        b=np.asarray(Image.open(old[m].format(i)).convert('RGB')).astype(float)/255
        h=min(a.shape[0],b.shape[0]);w=min(a.shape[1],b.shape[1])
        v.append(deltaE_ciede2000(rgb2lab(a[:h,:w]),rgb2lab(b[:h,:w])).mean())
    print(f'  {m:14s}',' '.join(f'{x:.2f}' for x in v))
print('\n== SID 00076 ReDDiT 输出 (10张同场景不同短曝光帧)')
fs=sorted(glob.glob('assets/perturb_out/reddit/00076_*'))
labs=[]
for f in fs:
    x=np.asarray(Image.open(f).convert('RGB')).astype(float)/255; l=rgb2lab(x); labs.append(l)
    print(f'  {os.path.basename(f)} meanRGB=({x[...,0].mean()*255:.0f},{x[...,1].mean()*255:.0f},{x[...,2].mean()*255:.0f}) L={l[...,0].mean():.1f} a*={l[...,1].mean():+.1f} b*={l[...,2].mean():+.1f} 饱和比例>0.98={(x>0.98).mean()*100:.1f}% <0.02={(x<0.02).mean()*100:.1f}%')
st=np.stack(labs); print('  帧间ΔE00(相邻帧平均)=%.2f'%np.mean([deltaE_ciede2000(st[j],st[j+1]).mean() for j in range(len(st)-1)]))
