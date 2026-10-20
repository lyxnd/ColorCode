# 两个问题的量化定义, 在 LOL-v2-real 测试集上用同一口径计算 (各方法的官方 LOL-v2-real 权重输出, 取自仓库)
# 问题一 暗部色度塌缩:
#   暗部饱和单元 = 同一张图里, 输入最亮通道 < 8 DN 且 GT 块色度 C* >= 20 的 8x8 块, 按 GT 色相每 45° 分一组, 至少 15 块
#   色度保持率 CRR = C*(输出) / C*(GT): 输出与 GT 都先在线性光下对单元求均值, 输出再缩放到 GT 的亮度 (排除亮度误差), 然后转 Lab 取 C*
#   辅助: HSV 饱和度保持率 = S(输出单元均值) / S(GT 单元均值), S = (max-min)/max, 在 sRGB 上算 (直观但随亮度变化, 不作主指标)
#   按输入亮度分档 (0-4, 4-8 DN) 给出 CRR 中位数, 即"收缩曲线"
# 问题二 跨拍摄色调不一致 (同一 GT 的多张低光图, 43 个场景 80 对):
#   色调差 ΔT = 两张输出全图平均 (a*, b*) 之差的模长; 亮度差 ΔL* = 全图平均 L* 之差的绝对值; 逐像素 ΔE00 (含噪声底, 只换噪声样本约 2.4)
#   输入有差别的对 (输入色调差 >= 1 或曝光差 >= 0.1 档, 判定与 evaluate_alignment_outputs.py 相同) 与输入几乎相同的对分开统计
# 用法 (仓库根目录, 先跑 analysis/run_alignment_cpu.py 生成配对表): python analysis/problem_metrics.py   结果: analysis/problem_metrics.txt
import sys, os, csv, hashlib, itertools, numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
sys.path.insert(0,'analysis/loss_data_ft'); from common import L, lin, srgb, W, IDS, RD, gt, low, blk
OUT={'Retinexformer':lambda i:f'Retinexformer/results_color_exp/orig/test/{i}.png','HVI-CIDNet':lambda i:f'HVI-CIDNet/output/HVIFirst/{i}.png',
     'CWNet':lambda i:f'CWNet-Causal-Wavelet-Network/output/low{i}.png','ReDDiT':lambda i:f'{RD}/output/{int(i)-689}_normal.png'}
lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
lab=lambda m: rgb2lab(srgb(m)[None,None])[0,0]
S=lambda m: (lambda s: (s.max()-s.min())/max(s.max(),1e-9))(srgb(m))

P('== 问题一: 暗部色度塌缩 (暗部饱和单元: 输入 < 8 DN 且 GT C* >= 20)')
units={m:[] for m in OUT}; ug=[]
for i in IDS:
    lo,g=low(i),gt(i); bi,bg=blk(lin(lo)),blk(lin(g)); lv=srgb(bi.max(1))*255
    lg=rgb2lab(srgb(bg)[None])[0]; C=np.hypot(lg[:,1],lg[:,2]); h=np.degrees(np.arctan2(lg[:,2],lg[:,1]))%360
    outs={m:blk(lin(L(f(i)))) for m,f in OUT.items()}
    for hb in range(8):
        s=(lv<8)&(C>=20)&((h-hb*45)%360<45)&(bg.max(1)<0.95)
        if s.sum()<15: continue
        gm=bg[s].mean(0); ug.append((i,np.median(lv[s]),int(s.sum())))
        for m in OUT:
            om=outs[m][s].mean(0); om=om*(gm@W)/max(om@W,1e-9)
            units[m].append((np.hypot(*lab(om)[1:])/np.hypot(*lab(gm)[1:]), S(om)/max(S(gm),1e-9)))
lvl=np.array([u[1] for u in ug]); ex=np.array([u[0] not in ('00739','00740') for u in ug])
P(f'  {len(ug)} 个单元, 来自 {len({u[0] for u in ug})} 张图 (其中 00739/00740 的单元 {(~ex).sum()} 个); 输入亮度中位数 {np.median(lvl):.1f} DN')
P(f'  {"方法":14s} CRR 中位数 [四分位]          不含 00739/40   0-4 DN   4-8 DN   CRR<0.7 的单元   HSV 饱和度保持率中位数')
for m in OUT:
    a=np.array(units[m]); c,s_=a[:,0],a[:,1]
    P(f'  {m:14s} {np.median(c):.2f} [{np.percentile(c,25):.2f}, {np.percentile(c,75):.2f}]        {np.median(c[ex]):.2f}         {np.median(c[lvl<4]):.2f}     {np.median(c[lvl>=4]):.2f}     {(c<0.7).mean():.0%}             {np.median(s_):.2f}')
P(f'  (0-4 DN 单元 {(lvl<4).sum()} 个, 4-8 DN 单元 {(lvl>=4).sum()} 个)')

P('\n== 问题二: 跨拍摄色调不一致 (43 个场景, 80 对)')
pc={r['pair']:r for r in csv.DictReader(open('analysis/alignment_intervention/alignment_pairs_retinexformer.csv',encoding='utf-8-sig'))}
info={(r['dark_id'],r['bright_id']) for r in pc.values() if float(r['original:input_tone_difference'])>=1 or abs(float(r['fitted_exposure_log2_ratio']))>=0.1}
grp={}
for i in IDS: grp.setdefault(hashlib.md5(open(f'{RD}/gt/{int(i)-689}_gt.png','rb').read()).hexdigest(),[]).append(i)
pairs=[p for g_ in grp.values() for p in itertools.combinations(sorted(g_),2)]
isinfo=np.array([(a,b) in info or (b,a) in info for a,b in pairs])
P(f'  输入有差别的对 {isinfo.sum()} 个, 输入几乎相同的对 {(~isinfo).sum()} 个')
P(f'  {"方法":14s} 色调差 ΔT 中位/均值 (有差别 | 几乎相同)   ΔT>3 的对   亮度差 ΔL* 中位/均值   ΔL*>3 的对   逐像素 ΔE00 均值   ΔE00>5 的对   最大 ΔT (对)')
for m,f in OUT.items():
    LB={i:rgb2lab(L(f(i))) for i in IDS}; T,Lm,E=[],[],[]
    for a,b in pairs:
        T.append(np.hypot(*(LB[a][...,1:].reshape(-1,2).mean(0)-LB[b][...,1:].reshape(-1,2).mean(0))))
        Lm.append(abs(LB[a][...,0].mean()-LB[b][...,0].mean())); E.append(deltaE_ciede2000(LB[a],LB[b]).mean())
    T,Lm,E=map(np.array,(T,Lm,E)); k=int(np.argmax(T))
    P(f'  {m:14s} {np.median(T[isinfo]):.2f}/{T[isinfo].mean():.2f} | {np.median(T[~isinfo]):.2f}/{T[~isinfo].mean():.2f}            {(T>3).mean():.0%}        {np.median(Lm):.2f}/{Lm.mean():.2f}            {(Lm>3).mean():.0%}         {E.mean():.2f}              {(E>5).mean():.0%}        {T[k]:.1f} ({pairs[k][0]}/{pairs[k][1]})')
open('analysis/problem_metrics.txt','w').write('\n'.join(lines)+'\n')
