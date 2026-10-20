# E2 全测试集归因: 色度保持率由什么决定? 00739 黄墙在这个规律里是不是离群点?
#   单元 = (图, GT 色相 45° 分档) 里 GT C*>20 的像素, 至少 400 像素; 保持率 = 输出/GT 的区域均值色度之比.
#   协变量: 输入色度信号 (局部 5x5 均值的 max-min, DN)、三种口径的噪声 -> 信噪比、区域输入亮度 (最亮通道 DN)、通道截断为 0 的比例、
#           这种颜色占整张图的比例、GT 亮度/色度/色相.
#   若"回归向均值收缩"是普遍机制, 保持率应主要由输入信噪比决定; 若 00739 黄墙远低于同信噪比的其他区域, 说明它另有原因.
# GT 与低光输入取自 ReDDiT 测试结果目录 (序号 k 对应 id 00689+k; 低光图去掉 padding).
# Diff-Retinex 的 100 张输出在提交 25a64fa 之后从 master 删除, 缺失时跳过; 结果 txt 里的 Diff-Retinex 列用的是 25a64fa 中的文件.
# 用法 (仓库根目录): python analysis/chroma_snr_attribution.py
import os, numpy as np
from PIL import Image
from skimage.color import rgb2lab
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.stats import spearmanr, rankdata
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
RX='Retinexformer/results_color_exp'
IDS=[f'{690+k:05d}' for k in range(100)]
METHODS={'Retinexformer':f'{RX}/orig/test/{{}}.png','HVI':'HVI-CIDNet/output/HVIFirst/{}.png','RF-A':f'{RX}/A/test/{{}}.png','RF-B':f'{RX}/B/test/{{}}.png',
         'Diff-Retinex':'Diff-Retinex/experiments/Diff_Retinex_val_261008_104214/results/{}.png','ReDDiT':None}
for m in [m for m,p in METHODS.items() if p is not None and not os.path.exists(p.format('00690'))]: print(f'缺少 {m} 的输出, 跳过'); METHODS.pop(m)
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def out(m,i): return L(f'{RD}/output/{int(i)-689}_normal.png' if METHODS[m] is None else METHODS[m].format(i))
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
def chroma_of_mean(img,mask): l=rgb2lab(srgb(lin(img)[mask].mean(0))[None,None])[0,0]; return np.hypot(l[1],l[2])

rows=[]
for i in IDS:
    g=L(f'{RD}/gt/{int(i)-689}_gt.png'); lo8=L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]*255
    lg=rgb2lab(g); Cg=np.hypot(lg[...,1],lg[...,2]); h=np.degrees(np.arctan2(lg[...,2],lg[...,1]))%360
    sm=uniform_filter(lo8,(5,5,0))
    n_all=np.sqrt(np.maximum(uniform_filter(lo8**2,(5,5,0))-sm**2,0)).mean(-1)         # 局部 5x5 标准差, 三通道平均
    G=lo8[...,1]; n_g=np.sqrt(np.maximum(uniform_filter(G**2,5)-uniform_filter(G,5)**2,0))  # 只用 G 通道 (不受 B 截断影响)
    hp=G-gaussian_filter(G,1.5); n_img=1.4826*np.median(np.abs(hp-np.median(hp)))/0.8     # 整图 G 通道噪声 (MAD)
    outs={m:out(m,i) for m in METHODS}
    for hb in range(8):
        hm=(h-hb*45)%360<45; mk=(Cg>20)&hm
        if mk.sum()<400: continue
        sig=(sm[mk].max(-1)-sm[mk].min(-1)).mean(); Cgt=chroma_of_mean(g,mk); m_lin=lin(lo8/255)[mk].mean(0)
        r=dict(id=i,hue=hb*45+22.5,n=int(mk.sum()),L=lg[...,0][mk].mean(),Cgt=Cgt,sig=sig,
               snr=sig/max(n_all[mk].mean(),1e-3),snr_G=sig/max(n_g[mk].mean(),1e-3),snr_img=sig/max(n_img,1e-3),
               censor=(lo8[mk]==0).any(-1).mean(),scene=((Cg>15)&hm).mean(),
               level=srgb(m_lin.max())*255)                                            # 区域均值颜色里最亮通道的输入 DN
        for m in METHODS: r[m]=chroma_of_mean(outs[m],mk)/Cgt
        rows.append(r)
MS=list(METHODS); col=lambda k,sel=None: np.array([r[k] for r in (sel or rows)])
wall=[r for r in rows if r['id']=='00739' and r['hue']==67.5][0]
print(f'单元数 {len(rows)}, 涉及 {len({r["id"] for r in rows})} 张图\n')

print('== 1) 保持率与协变量的 Spearman 相关 (单元级)')
print('协变量      '+''.join(f'{m:>14s}' for m in MS))
for c in ['snr','snr_G','snr_img','sig','level','censor','scene','L','Cgt']:
    print(f'{c:10s} '+''.join(f'{spearmanr(col(c),col(m))[0]:+14.2f}' for m in MS))

def pcorr(x,y,Z):
    rx,ry=rankdata(x),rankdata(y); Zr=np.c_[np.ones(len(x)),*[rankdata(z) for z in Z]]
    e=lambda v: v-Zr@np.linalg.lstsq(Zr,v,rcond=None)[0]
    return np.corrcoef(e(rx),e(ry))[0,1]
print('\n== 2) 偏相关 (控制 GT 亮度、GT 色度、色相 cos/sin; 截断与占比两行另外控制 snr)')
hc,hs=np.cos(np.radians(col('hue'))),np.sin(np.radians(col('hue')))
for c,extra in [('snr',[]),('censor',['snr']),('scene',['snr'])]:
    Z=[col('L'),col('Cgt'),hc,hs]+[col(e) for e in extra]
    print(f'{c:10s} '+''.join(f'{pcorr(col(c),col(m),Z):+14.2f}' for m in MS))

print('\n== 3) 按输入色度信噪比 (局部口径) 分五档的保持率中位数')
q=np.quantile(col('snr'),[0,.2,.4,.6,.8,1])
for k in range(5):
    sel=[r for r in rows if q[k]<=r['snr']<=q[k+1]]
    print(f'SNR {q[k]:5.2f}-{q[k+1]:5.2f} (n={len(sel)}) '+' '.join(f'{m}={np.median(col(m,sel)):.2f}' for m in MS))

print('\n== 3b) 按区域输入亮度 (最亮通道 DN) 分档的保持率中位数')
for a,b in zip([0,3,5,8,12,20,40],[3,5,8,12,20,40,256]):
    sel=[r for r in rows if a<=r['level']<b]
    if sel: print(f'{a:3d}-{b:<3d}DN (n={len(sel):3d}) '+' '.join(f'{m}={np.median(col(m,sel)):.2f}' for m in MS))

print('\n== 4) 00739 黄墙 (h=45-90°) 与"同信噪比的其他单元"比较 (排除 00739/00740 同场景)')
print(f'黄墙: n={wall["n"]} L={wall["L"]:.1f} C={wall["Cgt"]:.1f} 输入亮度={wall["level"]:.2f}DN (第{(col("level")<wall["level"]).mean()*100:.0f}百分位) 输入色度信号={wall["sig"]:.2f}DN 截断={wall["censor"]:.2f} 占画面={wall["scene"]:.2f}')
print('实际保持率                         '+' '.join(f'{m}={wall[m]:.2f}' for m in MS))
for key in ['snr','snr_G','snr_img']:
    v=col(key); sim=[r for r in rows if abs(np.log(r[key]/wall[key]))<0.25 and r['id'] not in ('00739','00740')]
    print(f'{key:7s}={wall[key]:.2f} 第{(v<wall[key]).mean()*100:3.0f}百分位, 同SNR {len(sim):2d} 个单元中位数 '+' '.join(f'{m}={np.median(col(m,sim)):.2f}' for m in MS))

sim=[r for r in rows if r['level']<5 and r['id'] not in ('00739','00740')]
print(f'输入亮度 <5DN 的其他单元 {len(sim)} 个, 中位数 '+' '.join(f'{m}={np.median(col(m,sim)):.2f}' for m in MS))

print('\n== 5) 保持率最低的 10 个单元 (Retinexformer 官方)')
for r in sorted(rows,key=lambda r:r['Retinexformer'])[:10]:
    print(f"{r['id']} h={r['hue']:5.1f} n={r['n']:6d} L={r['L']:4.1f} C={r['Cgt']:4.1f} 截断={r['censor']:.2f} SNR={r['snr']:4.2f} 占画面={r['scene']:.2f} | "+' '.join(f'{m}={r[m]:.2f}' for m in MS))
