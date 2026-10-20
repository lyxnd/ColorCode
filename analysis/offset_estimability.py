# 逐通道暗端底座能否只从低光图本身估计出来 (不用 GT、不用语义)?
#   目标: check_debias.py 用 GT 拟合出的每张图逐通道底座, 取通道差 R-G、B-G (影响颜色的是通道之间的差), 单位为输入 DN
#   估计量 (只用低光输入, sRGB DN):
#     分位数   : 各通道第 p 百分位之差 (真正的黑处信号约为 0, 读数之差约等于底座之差)
#     零像素比例: 各通道被压到 0 的像素比例之差 (底座为负时更多像素被截到 0)
#     最暗块   : 最亮通道 < 6 DN 的 8x8 块里, 通道差的中位数
#     线性回归 : 各通道 0-15 DN 的直方图 + 以上统计量, 岭回归; 训练集 5 折交叉验证, 再用整个训练集拟合、在测试集上评估
#   另外比较 LOL-v2-real 训练集里 "来自 LOL-v1" 与 "新拍" 两批图的底座离散程度
# 用法 (仓库根目录, 先跑 analysis/loss_data_ft/check_debias.py): python analysis/offset_estimability.py   结果: analysis/offset_estimability.txt
import sys, os, hashlib, numpy as np
from PIL import Image
from scipy.stats import spearmanr
sys.path.insert(0,'analysis/loss_data_ft'); from common import L, TR, IDS, low, blk
names=sorted(os.listdir(f'{TR}/Low')); DN=12.92*255
To=np.load('analysis/loss_data_ft/train_offsets.npy')*DN; Te=np.load('analysis/loss_data_ft/test_offsets.npy',allow_pickle=True).item()
Te=np.array([Te[i] for i in IDS])*DN
lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
def feats(x):
    d=x*255; f={}
    for p in [0.5,1,2,5]: q=np.percentile(d.reshape(-1,3),p,axis=0); f[f'p{p}']=q
    f['zero']=(d.reshape(-1,3)<0.5).mean(0)
    b=blk(d); s=b.max(1)<6; f['dark']=np.median(b[s],0) if s.sum()>=20 else np.median(b[np.argsort(b.max(1))[:20]],0)
    h=np.concatenate([np.histogram(d[...,c],bins=16,range=(0,16))[0]/d[...,c].size for c in range(3)])
    return f,h
diff=lambda v: np.array([v[0]-v[1],v[2]-v[1]])   # (R-G, B-G)
Ftr=[feats(L(f'{TR}/Low/{n}')) for n in names]; Fte=[feats(low(i)) for i in IDS]
ok_tr=np.abs(To).sum(1)>0; ok_te=np.abs(Te).sum(1)>0
ytr=np.array([diff(v) for v in To]); yte=np.array([diff(v) for v in Te])
P(f'GT 拟合出底座的图: 训练集 {ok_tr.sum()}/{len(names)}, 测试集 {ok_te.sum()}/100; 目标为通道差 R-G、B-G (DN)')
P(f'  目标的离散程度 (标准差): 训练集 R-G {ytr[ok_tr,0].std():.2f}, B-G {ytr[ok_tr,1].std():.2f}; 测试集 R-G {yte[ok_te,0].std():.2f}, B-G {yte[ok_te,1].std():.2f}')
P('\n== 1) 单个统计量与 GT 底座差的 Spearman 相关 (训练集 / 测试集)')
for k in ['p0.5','p1','p2','p5','zero','dark']:
    a=np.array([diff(f[k]) for f,_ in Ftr]); b=np.array([diff(f[k]) for f,_ in Fte])
    P(f'  {k:6s}  R-G: {spearmanr(a[ok_tr,0],ytr[ok_tr,0])[0]:+.2f} / {spearmanr(b[ok_te,0],yte[ok_te,0])[0]:+.2f}    B-G: {spearmanr(a[ok_tr,1],ytr[ok_tr,1])[0]:+.2f} / {spearmanr(b[ok_te,1],yte[ok_te,1])[0]:+.2f}')
def X(F): return np.array([np.concatenate([h]+[diff(f[k]) for k in ['p0.5','p1','p2','p5','zero','dark']]+[f['zero']]) for f,h in F])
Xtr,Xte=X(Ftr)[ok_tr],X(Fte)[ok_te]; Ytr,Yte=ytr[ok_tr],yte[ok_te]
mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9; Z=lambda x: np.c_[(x-mu)/sd,np.ones(len(x))]
def ridge(A,y,lam=10.0): return np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@y)
r2=lambda y,p: 1-((y-p)**2).sum(0)/((y-y.mean(0))**2).sum(0)
rng=np.random.RandomState(0); fold=rng.permutation(len(Xtr))%5; cv=np.zeros_like(Ytr)
for k in range(5):
    w=ridge(Z(Xtr[fold!=k]),Ytr[fold!=k]); cv[fold==k]=Z(Xtr[fold==k])@w
w=ridge(Z(Xtr),Ytr); pte=Z(Xte)@w
P('\n== 2) 岭回归 (直方图 + 统计量, 共 %d 维)' % Xtr.shape[1])
P(f'  训练集 5 折交叉验证 R²: R-G {r2(Ytr,cv)[0]:.2f}, B-G {r2(Ytr,cv)[1]:.2f};  Spearman R-G {spearmanr(cv[:,0],Ytr[:,0])[0]:+.2f}, B-G {spearmanr(cv[:,1],Ytr[:,1])[0]:+.2f}')
P(f'  测试集 R²: R-G {r2(Yte,pte)[0]:.2f}, B-G {r2(Yte,pte)[1]:.2f};  Spearman R-G {spearmanr(pte[:,0],Yte[:,0])[0]:+.2f}, B-G {spearmanr(pte[:,1],Yte[:,1])[0]:+.2f}')
P(f'  测试集平均绝对误差 (DN): R-G {np.abs(pte-Yte).mean(0)[0]:.2f}, B-G {np.abs(pte-Yte).mean(0)[1]:.2f}  (若一律猜训练集均值: {np.abs(Ytr.mean(0)-Yte).mean(0)[0]:.2f}, {np.abs(Ytr.mean(0)-Yte).mean(0)[1]:.2f})')

H=lambda p: hashlib.md5(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()
v1={H(f'data/LOLv1/{s}/input/{f}') for s in ['Train','Test'] for f in os.listdir(f'data/LOLv1/{s}/input')}
src=np.array(['v1' if H(f'{TR}/Low/{n}') in v1 else 'new' for n in names])
P('\n== 3) LOL-v2-real 训练集两批图的底座离散程度 (GT 拟合, DN)')
for s,lab in [('v1','来自 LOL-v1'),('new','新拍')]:
    m=ok_tr&(src==s); q=lambda a: np.percentile(a,[25,75])
    P(f'  {lab:10s} {m.sum():3d} 张: R-G 四分位 {q(ytr[m,0]).round(2)}, B-G 四分位 {q(ytr[m,1]).round(2)}; |R-G|>1DN 的比例 {(np.abs(ytr[m,0])>1).mean():.0%}, |B-G|>1DN {(np.abs(ytr[m,1])>1).mean():.0%}')
open('analysis/offset_estimability.txt','w').write('\n'.join(lines)+'\n')
