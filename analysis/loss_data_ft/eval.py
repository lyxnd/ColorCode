# 评估三组微调 (train.py) 与官方权重: 改损失 / 改数据之后, "按输入亮度打折" 是否改变.
#   网络输入两种: 原始测试图; 去掉 GT 估出的逐通道底座的测试图 (oracle, 与 bias 组训练时的处理相同).
#   所有比例都以"原始输入"的暗示颜色为尺子 (block_stats 的 imp), 只换网络输入, GT 比例在两种输入下相同.
#   1) 00739 黄墙保持率 (亮度归一化后的色度比, 同 cross_dataset_00739.py)
#   2) 墙的输入亮度 x k (带噪, 信噪比不变, 同 intervention_00739.py I2a): 模型学到的收缩曲线
#   3) 测试集 (不含 00739/00740) 按输入亮度分档: 输出比例 vs GT 比例 (暗示色度 20-60)
#   4) 暗部单元 (图 x 色相 45° 档, <8DN, 暗示色度>=20, 同 dark_chroma_methods.py): 与 GT 比例的 Spearman, 可信/虚高单元上的输出比例
#   5) GT 中性像素 (C*<6) 的 Δa* Δb*: 改损失后是否引入整体偏色; PSNR (全部 / 9 张未见场景)
# 用法 (仓库根目录): python analysis/loss_data_ft/eval.py   结果: analysis/loss_data_ft/results.txt, 对比图 00739_compare.png
import sys, os, numpy as np, torch
sys.path.insert(0,os.path.dirname(__file__)); from common import *
from scipy.stats import spearmanr
torch.set_num_threads(int(os.environ.get('THREADS',4)))
if os.environ.get('QUICK'): IDS=sorted(set(IDS[::8]+['00739','00740','00705'])); UNSEEN=['00705']   # 只用于调试脚本
R='analysis/loss_data_ft/runs'
MODELS={'官方':'Retinexformer/pretrained_weights/LOL_v2_real.pth'}
for arm in ['l1','gan','bias']:
    for it in [1000,2000]:
        if os.path.exists(f'{R}/{arm}/G_{it}.pth'): MODELS[f'{arm}@{it}']=f'{R}/{arm}/G_{it}.pth'
TO=np.load('analysis/loss_data_ft/test_offsets.npy',allow_pickle=True).item()
wall=np.load('analysis/wall_mask_00739.npy'); G7=gt('00739'); gw=lin(G7)[wall].mean(0)
def ret(o):
    m=lin(o)[wall].mean(0); m=m*(gw@W)/(m@W); return np.hypot(*lab(m[None])[0,1:])/np.hypot(*lab(gw[None])[0,1:])
inp=lambda i,c: low(i) if c=='raw' else debias(low(i),TO[i])
GTL={i:rgb2lab(gt(i)) for i in IDS}; NEU={i:(np.hypot(GTL[i][...,1],GTL[i][...,2])<6)&(GTL[i][...,0]<45) for i in IDS}
psnr=lambda a,b: 10*np.log10(1/np.mean((a-b)**2))
EX=[i for i in IDS if i not in ('00739','00740')]
lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
res={}; crops={}
for name,ck in MODELS.items():
    net=build_rf(ck).eval()
    for c in ['raw','debias']:
        outs={i:run(net,inp(i,c)) for i in IDS}
        T={i:block_stats(low(i),gt(i),{'o':outs[i]}) for i in IDS}
        cat=lambda ids: {k:np.concatenate([T[i][k] for i in ids]) for k in T[ids[0]]}
        ex=cat(EX); r=dict(wall=ret(outs['00739']),psnr=np.mean([psnr(outs[i],gt(i)) for i in IDS]),psnr_u=np.mean([psnr(outs[i],gt(i)) for i in UNSEEN]))
        for a,b in [(3,5),(5,8),(8,12)]: s=dark_sel(ex,a,b); r[f'bin{a}']=(ratio(ex,'gt',s),ratio(ex,'o',s),int(s.sum()))
        units=[]
        for i in EX:
            d=T[i]; cc=np.hypot(*d['imp'].T); h=np.degrees(np.arctan2(d['imp'][:,1],d['imp'][:,0]))%360
            for hb in range(8):
                s=(d['lvl']<8)&(cc>=20)&((h-hb*45)%360<45)
                if s.sum()>=15: units.append((ratio(d,'gt',s),ratio(d,'o',s)))
        u=np.array(units); r['rho']=spearmanr(u[:,0],u[:,1])[0]; r['trust']=np.median(u[u[:,0]>0.8,1]); r['fake']=np.median(u[u[:,0]<0.4,1]); r['nu']=len(u)
        da=[(rgb2lab(outs[i])[NEU[i]]-GTL[i][NEU[i]]).mean(0)[1:] for i in IDS if NEU[i].sum()>=500]; r['neu']=np.mean(da,0)
        if c=='raw':
            sw=[]
            for k in [0.5,1,2,4,8]:
                x=low('00739').copy(); x[wall]=np.clip(np.round(x[wall]*255*k),0,255)/255; sw.append(ret(run(net,x)))
            r['sweep']=sw
        res[(name,c)]=r; crops[(name,c)]=outs['00739'][40:360,:]
        print(name,c,{k:(np.round(v,3) if not isinstance(v,tuple) else v) for k,v in r.items()},flush=True)

u0=res[('官方','raw')]
P(f'测试集暗部单元 {u0["nu"]} 个 (不含 00739/00740); 00739 底座 (DN) {(TO["00739"]*12.92*255).round(2)}')
P('\n== 1-2) 00739 黄墙保持率; 墙亮度 x k 的收缩曲线 (原始输入)')
P(f'{"模型":12s} 墙(原始输入) 墙(去底座输入) | k=0.5   1     2     4     8')
for name in MODELS:
    a,b=res[(name,'raw')],res[(name,'debias')]
    P(f'{name:12s}    {a["wall"]:.2f}         {b["wall"]:.2f}      | '+' '.join(f'{v:.2f} ' for v in a['sweep']))
for c,t in [('raw','原始输入'),('debias','去底座输入 (oracle)')]:
    P(f'\n== 3-5) 测试集, 网络输入 = {t}')
    g=res[('官方',c)]
    P(f'{"模型":12s} 3-5DN  5-8DN  8-12DN | 单元 Spearman  可信单元  虚高单元 | 中性区 Δa*  Δb* | PSNR 全部  未见')
    P(f'{"测试 GT":12s} {g["bin3"][0]:.2f}   {g["bin5"][0]:.2f}   {g["bin8"][0]:.2f}   |')
    for name in MODELS:
        r=res[(name,c)]
        P(f'{name:12s} {r["bin3"][1]:.2f}   {r["bin5"][1]:.2f}   {r["bin8"][1]:.2f}   |    {r["rho"]:+.2f}        {r["trust"]:.2f}      {r["fake"]:.2f}  |  {r["neu"][0]:+5.1f} {r["neu"][1]:+5.1f}  |  {r["psnr"]:.2f}  {r["psnr_u"]:.2f}')
open('analysis/loss_data_ft/results.txt','w').write('\n'.join(lines)+'\n')
from PIL import Image
names=[n for n in MODELS if '@1000' not in n]
row=lambda c: np.concatenate([G7[40:360]]+[crops[(n,c)] for n in names],1)
M=np.concatenate([row('raw'),row('debias')],0); Image.fromarray(np.round(M*255).astype(np.uint8)).resize((M.shape[1]//2,M.shape[0]//2)).save('analysis/loss_data_ft/00739_compare.png')
print('列: GT, '+', '.join(names)+'; 上行原始输入, 下行去底座输入')
