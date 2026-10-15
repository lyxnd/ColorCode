# 五个方法在测试集暗部的色度处理: 是统一按亮度收缩 (回归到均值), 还是能分辨"输入颜色是真是假"?
#   块级统计与 train_chroma_prior.py 相同: 8x8 块, 输入用整图逐通道增益 (由 GT 拟合) 校正并对齐到 GT 块亮度 = "输入暗示的颜色",
#   比例 = GT 或输出色度沿暗示色相方向的分量 / 暗示色度 (1 = 输入颜色可信 / 输出保留了输入颜色).
#   1) 按输入亮度分档: 训练数据 vs 测试 GT vs 各方法 (测试集不含 00739/00740, 它们单独列出)
#   2) 暗部单元 (图 x 色相 45° 档, 输入亮度 <8DN, 暗示色度 >=20): 各方法的比例与 GT 比例的相关 = 分辨真假颜色的能力
# 输出: Retinexformer 官方 (results_color_exp/orig), HVI (output/HVIFirst), CWNet (output), ReDDiT (测试结果目录),
#       Diff-Retinex (experiments/Diff_Retinex_val_261008_104214, 已从 master 删除, 缺失时跳过; 结果 txt 用的是 25a64fa 中的文件).
# 用法 (仓库根目录): python analysis/dark_chroma_methods.py
import os, numpy as np
from PIL import Image
from skimage.color import rgb2lab
from scipy.stats import spearmanr
TR='data/LOLv2/Real_captured/Train'; RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
IDS=[f'{690+k:05d}' for k in range(100)]
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
gt=lambda i: L(f'{RD}/gt/{int(i)-689}_gt.png'); low=lambda i: L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]
OUT={'Retinexformer':'Retinexformer/results_color_exp/orig/test/{}.png','HVI':'HVI-CIDNet/output/HVIFirst/{}.png',
     'CWNet':'CWNet-Causal-Wavelet-Network/output/low{}.png','Diff-Retinex':'Diff-Retinex/experiments/Diff_Retinex_val_261008_104214/results/{}.png',
     'ReDDiT':RD+'/output/{}_normal.png'}
path=lambda m,i: OUT[m].format(int(i)-689 if m=='ReDDiT' else i)
for m in [m for m in OUT if not os.path.exists(path(m,'00690'))]: print(f'缺少 {m} 的输出, 跳过'); OUT.pop(m)
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
W=np.array([0.2126,0.7152,0.0722])
def blk(x,B=8): h,w=x.shape[0]//B*B,x.shape[1]//B*B; return x[:h,:w].reshape(h//B,B,w//B,B,-1).mean((1,3)).reshape(-1,x.shape[-1])
def lab(m): return rgb2lab(srgb(m)[None])[0]
def block_stats(lo,g,outs={}):
    ll,gl=lin(lo),lin(g); bi,bg=blk(ll),blk(gl); Yg=bg@W; ok=(Yg>1e-4)&(bg.max(1)<0.95)
    imp=bi*((gl*ll).reshape(-1,3).sum(0)/np.maximum((ll*ll).reshape(-1,3).sum(0),1e-12))
    al=lambda m: m*(Yg/np.maximum(m@W,1e-9))[:,None]
    r=dict(lvl=srgb(bi.max(1))*255,imp=lab(al(imp))[:,1:],gt=lab(bg)[:,1:])
    for t,o in outs.items(): r[t]=lab(al(blk(lin(o))))[:,1:]
    return {k:v[ok] for k,v in r.items()}
def cat(lst): return {k:np.concatenate([d[k] for d in lst]) for k in lst[0]}
def ratio(d,key,sel): return (d[key][sel]*d['imp'][sel]).sum()/max((d['imp'][sel]**2).sum(),1e-9)

D=cat([block_stats(L(f'{TR}/Low/{n}'),L(f'{TR}/Normal/{n}')) for n in sorted(os.listdir(f'{TR}/Low'))])
T={i:block_stats(low(i),gt(i),{m:L(path(m,i)) for m in OUT}) for i in IDS}
ex=cat([T[i] for i in IDS if i not in ('00739','00740')]); MS=list(OUT)
print('== 1) 按输入亮度分档的比例 (测试集不含 00739/00740)')
for cb in [(20,35),(35,60)]:
    print(f'  输入暗示色度 {cb[0]}-{cb[1]}')
    print('  输入亮度    训练GT | 测试块  测试GT '+''.join(f'{m:>14s}' for m in MS))
    for a,b in [(0,3),(3,5),(5,8),(8,12),(12,20),(20,40)]:
        f=lambda d: (d['lvl']>=a)&(d['lvl']<b)&(np.hypot(*d['imp'].T)>=cb[0])&(np.hypot(*d['imp'].T)<cb[1])
        s,t=f(D),f(ex)
        print(f'  {a:3d}-{b:<3d}DN   {ratio(D,"gt",s):.2f}  | {t.sum():6d}  {ratio(ex,"gt",t):.2f} '+''.join(f'{ratio(ex,m,t):14.2f}' for m in MS))
wall=np.load('analysis/wall_mask_00739.npy')
lo,g=low('00739'),gt('00739'); keep=blk(wall[...,None].astype(float))[:,0]>0.999
Yg=blk(lin(g))@W; ok=(Yg>1e-4)&(blk(lin(g)).max(1)<0.95)
Wb=block_stats(lo,g,{m:L(path(m,'00739')) for m in MS}); Wb={k:v[keep[ok]] for k,v in Wb.items()}; s=np.ones(len(Wb['lvl']),bool)
print(f'  00739 黄墙 ({s.sum()} 块, 输入亮度中位数 {np.median(Wb["lvl"]):.1f}DN): GT={ratio(Wb,"gt",s):.2f} '+' '.join(f'{m}={ratio(Wb,m,s):.2f}' for m in MS))

print('\n== 2) 暗部单元: 输出比例与 GT 比例的关系')
units=[]
for i in IDS:
    d=T[i]; c=np.hypot(*d['imp'].T); h=np.degrees(np.arctan2(d['imp'][:,1],d['imp'][:,0]))%360
    for hb in range(8):
        s=(d['lvl']<8)&(c>=20)&((h-hb*45)%360<45)
        if s.sum()>=15: units.append(dict(id=i,hue=hb*45,n=int(s.sum()),gt=ratio(d,'gt',s),**{m:ratio(d,m,s) for m in MS}))
g=np.array([u['gt'] for u in units]); exu=np.array([u['id'] not in ('00739','00740') for u in units]); ok=g>0.8; bad=g<0.4
print(f'  {len(units)} 个单元 / {len({u["id"] for u in units})} 张图; GT 比例中位数 {np.median(g):.2f} (四分位 {np.percentile(g,25):.2f}-{np.percentile(g,75):.2f}), 可信(>0.8) {ok.sum()} 个, 虚高(<0.4) {bad.sum()} 个')
for m in MS:
    v=np.array([u[m] for u in units])
    print(f'  {m:14s} Spearman={spearmanr(g,v)[0]:+.2f} (不含00739/40 {spearmanr(g[exu],v[exu])[0]:+.2f})  中位数 {np.median(v):.2f}  可信单元 {np.median(v[ok]):.2f}  虚高单元 {np.median(v[bad]):.2f}')
