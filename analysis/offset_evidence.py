# 不用 GT、不假设物理来源, 只用同一场景的两次拍摄检验: 两张低光图之间是否存在逐通道的加性偏差
#   若两次拍摄只差曝光和白平衡 (乘性), 每个通道的 "较亮拍摄 vs 较暗拍摄" 应是一条过原点的直线;
#   若存在加性偏差, 直线有截距, 且三个通道截距不同时会改变颜色.
#   做法: 8x8 块均值 (线性光), 用较暗拍摄该通道 1-30 DN 的块, 拟合 亮 = k * 暗 + c; 截距折算成较暗拍摄的 DN: o = -c / k
#         (o > 0 表示较暗拍摄相对于成比例的关系多出一个正偏置). 通道差 o_R - o_G、o_B - o_G 才影响颜色.
#   显著性: 对块做 200 次自助重采样, 95% 区间不含 0 且 |差| > 0.5 DN 记为显著.
#   拟合优度: 只用增益 (过原点) 与 增益+截距 在暗端的残差比较.
#   阴性对照: 输入几乎相同的对 (判定同 evaluate_alignment_outputs.py) 应得到接近 0 的截距差.
# 用法 (仓库根目录, 先跑 analysis/run_alignment_cpu.py 生成配对表): python analysis/offset_evidence.py   结果: analysis/offset_evidence.txt
import sys, os, csv, numpy as np
sys.path.insert(0,'analysis/loss_data_ft'); from common import low, lin, srgb, blk
DN=12.92*255
pc=list(csv.DictReader(open('analysis/alignment_intervention/alignment_pairs_retinexformer.csv',encoding='utf-8-sig')))
lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
rng=np.random.RandomState(0); res=[]
def fit(d,b,through_origin=False):
    if through_origin: k=(d*b).sum()/(d*d).sum(); return k,0.0
    A=np.c_[d,np.ones_like(d)]; k,c=np.linalg.lstsq(A,b,rcond=None)[0]; return k,c
for r in pc:
    bi,di=r['bright_id'],r['dark_id']; B,D=blk(lin(low(bi))),blk(lin(low(di)))
    info=float(r['original:input_tone_difference'])>=1 or abs(float(r['fitted_exposure_log2_ratio']))>=0.1
    o=np.zeros(3); boot=np.zeros((200,3)); gain_rss=0; aff_rss=0; ok=True
    for c in range(3):
        v=srgb(D[:,c])*255; s=(v>=1)&(v<30)&(B[:,c]<0.95)
        if s.sum()<50: ok=False; break
        d,b=D[s,c],B[s,c]; k,cc=fit(d,b); o[c]=-cc/k*DN
        k0,_=fit(d,b,True); gain_rss+=((b-k0*d)**2).sum(); aff_rss+=((b-k*d-cc)**2).sum()
        for t in range(200):
            j=rng.randint(len(d),size=len(d)); kk,c2=fit(d[j],b[j]); boot[t,c]=-c2/kk*DN
    if not ok: continue
    dif=np.array([o[0]-o[1],o[2]-o[1]]); bd=np.c_[boot[:,0]-boot[:,1],boot[:,2]-boot[:,1]]
    lo_,hi_=np.percentile(bd,2.5,0),np.percentile(bd,97.5,0)
    sig=((lo_>0)|(hi_<0))&(np.abs(dif)>0.5)
    res.append(dict(pair=r['pair'],info=info,o=o,dif=dif,sig=sig,red=1-aff_rss/gain_rss))
for lab,sel in [('输入有差别的对',lambda x: x['info']),('输入几乎相同的对 (阴性对照)',lambda x: not x['info'])]:
    R=[x for x in res if sel(x)]; dif=np.array([x['dif'] for x in R]); sig=np.array([x['sig'] for x in R]); red=np.array([x['red'] for x in R])
    P(f'== {lab}: {len(R)} 对')
    P(f'  截距的通道差 (较暗拍摄的 DN): |R-G| 中位数 {np.median(np.abs(dif[:,0])):.2f}, 90% 分位 {np.percentile(np.abs(dif[:,0]),90):.2f};  |B-G| 中位数 {np.median(np.abs(dif[:,1])):.2f}, 90% 分位 {np.percentile(np.abs(dif[:,1]),90):.2f}')
    P(f'  通道差显著 (95% 区间不含 0 且 >0.5 DN) 的对: R-G {sig[:,0].mean():.0%}, B-G {sig[:,1].mean():.0%}, 任一 {sig.any(1).mean():.0%}')
    P(f'  加截距后暗端残差下降: 中位数 {np.median(red):.0%}; 下降 >20% 的对 {(red>0.2).mean():.0%}')
    P(f'  三个通道的截距 (DN) 本身: 中位数 R {np.median([x["o"][0] for x in R]):+.2f}, G {np.median([x["o"][1] for x in R]):+.2f}, B {np.median([x["o"][2] for x in R]):+.2f}; 截距为负的通道占比 {np.mean([x["o"]<-0.5 for x in R]):.0%}')
P('\n== 输入有差别的对中, 截距通道差最大的 6 对 (o_R-o_G, o_B-o_G, DN)')
for x in sorted([x for x in res if x['info']],key=lambda x:-np.abs(x['dif']).max())[:6]:
    P(f'  {x["pair"]}: {x["dif"][0]:+.2f}, {x["dif"][1]:+.2f}   显著: {"R-G " if x["sig"][0] else ""}{"B-G" if x["sig"][1] else ""}   残差下降 {x["red"]:.0%}')
open('analysis/offset_evidence.txt','w').write('\n'.join(lines)+'\n')

# 替代解释: 不是加性偏差, 而是两次拍摄白平衡不同、暗部曲线非线性造成的弯曲 (过原点的幂函数 亮 = k * 暗^γ).
# 两种模型都是 2 个参数, 在同样的块上比较暗端残差; 若加性截距模型普遍更好, "加性偏差" 是更贴切的描述.
P('\n== 替代解释检验: 加性截距 (亮 = k*暗 + c) vs 过原点幂函数 (亮 = k*暗^γ), 输入有差别的对, 逐通道')
win=[]; ratio=[]
for r in pc:
    if not (float(r['original:input_tone_difference'])>=1 or abs(float(r['fitted_exposure_log2_ratio']))>=0.1): continue
    B,D=blk(lin(low(r['bright_id']))),blk(lin(low(r['dark_id'])))
    for c in range(3):
        v=srgb(D[:,c])*255; s=(v>=1)&(v<30)&(B[:,c]<0.95)&(B[:,c]>0)
        if s.sum()<50: continue
        d,b=D[s,c],B[s,c]; k,cc=fit(d,b); ra=((b-k*d-cc)**2).sum()
        g_,lk=np.polyfit(np.log(d),np.log(b),1); kp=np.exp(lk)
        from scipy.optimize import least_squares
        sol=least_squares(lambda p: p[0]*d**p[1]-b,[kp,g_]); rp=(sol.fun**2).sum()
        win.append(ra<rp); ratio.append(ra/rp)
P(f'  {len(win)} 个通道拟合: 加性截距模型残差更小的占 {np.mean(win):.0%}; 残差比 (截距/幂函数) 中位数 {np.median(ratio):.2f}')
open('analysis/offset_evidence.txt','w').write('\n'.join(lines)+'\n')
