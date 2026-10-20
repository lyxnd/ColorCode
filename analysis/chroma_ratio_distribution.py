# 训练集暗部 "GT 色度 / 输入暗示色度" 的分布: 是集中在一个值附近, 还是分散? 差别主要在图与图之间还是图内?
#   块统计口径同 loss_data_ft/common.py 的 block_stats (8x8 块, 输入用整图逐通道增益校正并对齐到 GT 块亮度 = 输入暗示的颜色),
#   比例 = GT 色度沿暗示色相方向的分量 / 暗示色度; 只看暗示色度 20-60 的块
#   若差别主要在图与图之间 (每张图的比例分散、图内很一致), 说明暗部颜色是否可信由每次拍摄决定, 而不是由区域内容决定
# 用法 (仓库根目录): python analysis/chroma_ratio_distribution.py   结果: analysis/chroma_ratio_distribution.txt
import sys, os, numpy as np
sys.path.insert(0,'analysis/loss_data_ft'); from common import L, TR, block_stats
names=sorted(os.listdir(f'{TR}/Low')); rows=[]
lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
for n in names:
    st=block_stats(L(f'{TR}/Low/{n}'),L(f'{TR}/Normal/{n}'))
    c=np.hypot(*st['imp'].T); r=(st['gt']*st['imp']).sum(1)/np.maximum(c**2,1e-9)
    for lo_,hi_ in [(3,5),(5,8)]:
        s=(st['lvl']>=lo_)&(st['lvl']<hi_)&(c>=20)&(c<60)
        if s.sum(): rows.append((n,lo_,r[s],c[s]))
for lv,hv in [(3,5),(5,8)]:
    R=np.concatenate([x[2] for x in rows if x[1]==lv]); C=np.concatenate([x[3] for x in rows if x[1]==lv]); w=C**2
    P(f'== 输入亮度 {lv}-{hv} DN, 暗示色度 20-60: {len(R)} 块, 加权均值 {(R*w).sum()/w.sum():.2f}, 中位数 {np.median(R):.2f}')
    h,e=np.histogram(np.clip(R,-0.5,2.0),bins=np.arange(-0.5,2.01,0.125))
    for k in range(len(h)):
        if h[k]/len(R)>=0.001: P(f'   {e[k]:+5.2f}~{e[k+1]:+5.2f} {h[k]/len(R)*100:5.1f}% '+'#'*int(h[k]/len(R)*200))
    per=np.array([(x[2]*x[3]**2).sum()/(x[3]**2).sum() for x in rows if x[1]==lv and len(x[2])>=20])
    within=[np.percentile(x[2],75)-np.percentile(x[2],25) for x in rows if x[1]==lv and len(x[2])>=20]
    P(f'   每张图的比例 ({len(per)} 张, 每张 >=20 块) 10/25/50/75/90 分位: {np.percentile(per,[10,25,50,75,90]).round(2)}')
    P(f'   图内四分位距的中位数 {np.median(within):.2f}  vs  全部块的四分位距 {np.subtract(*np.percentile(R,[75,25])):.2f}\n')
open('analysis/chroma_ratio_distribution.txt','w').write('\n'.join(lines)+'\n')
