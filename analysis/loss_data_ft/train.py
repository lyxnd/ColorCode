# 同一 Retinexformer、同一起点 (官方 LOL_v2_real.pth)、同一批次序列, 只改一处的三组短微调 (CPU):
#   l1   : 官方配方 (L1 + mixup + 8 种翻转旋转 + 梯度裁剪 0.01) 继续训练, 作为对照 (patch 128 本身也会改变行为)
#   gan  : l1 + 条件 PatchGAN 对抗损失 (LSGAN, 权重 0.01, 同 pix2pix 的 L1:GAN=100:1) -> 只改损失: 回归换成"从条件分布里挑一个"
#   bias : l1, 但训练输入先去掉 GT 估出的逐通道暗端底座 (check_debias.py) -> 只改数据
# 学习率 1e-4 余弦退火到 1e-6; batch 8, patch 128 (CPU 内存放不下官方的 256x8). 每 250 步存档, 可断点续训.
# 用法 (仓库根目录): python analysis/loss_data_ft/train.py --arm l1 [--iters 2000]
import sys, os, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,os.path.dirname(__file__)); from common import *
p=argparse.ArgumentParser(); p.add_argument('--arm',choices=['l1','gan','bias'],required=True)
p.add_argument('--iters',type=int,default=2000); p.add_argument('--threads',type=int,default=4); p.add_argument('--out'); a=p.parse_args()
torch.set_num_threads(a.threads); OUT=a.out or f'analysis/loss_data_ft/runs/{a.arm}'; os.makedirs(OUT,exist_ok=True)
PS,BS,LR0,LR1,LAM=128,8,1e-4,1e-6,0.01

names=sorted(os.listdir(f'{TR}/Low')); offs=np.load('analysis/loss_data_ft/train_offsets.npy')
u8=lambda x: np.round(x*255).astype(np.uint8)
LQ=np.stack([u8(debias(L(f'{TR}/Low/{n}'),o)) if a.arm=='bias' else u8(L(f'{TR}/Low/{n}')) for n,o in zip(names,offs)])
GT=np.stack([u8(L(f'{TR}/Normal/{n}')) for n in names])
print(f'{a.arm}: {len(names)} 对训练图已载入',flush=True)

def batch(rng):
    """与组别无关的独立随机流: 三组看到完全相同的裁剪、增广与 mixup"""
    idx=rng.randint(len(names),size=BS); xs,ys=[],[]
    for i in idx:
        h,w=LQ.shape[1:3]; t,l=rng.randint(h-PS+1),rng.randint(w-PS+1); f=rng.randint(8)
        x,y=LQ[i,t:t+PS,l:l+PS],GT[i,t:t+PS,l:l+PS]
        if f&1: x,y=x[:,::-1],y[:,::-1]
        if f&2: x,y=x[::-1],y[::-1]
        if f&4: x,y=x.transpose(1,0,2),y.transpose(1,0,2)
        xs.append(x); ys.append(y)
    x=torch.from_numpy(np.stack(xs).transpose(0,3,1,2).copy()).float()/255; y=torch.from_numpy(np.stack(ys).transpose(0,3,1,2).copy()).float()/255
    if rng.rand()<0.5:   # 官方 Mixing_Augment(use_identity=True): 一半概率 mixup, Beta(1.2,1.2)
        lam=rng.beta(1.2,1.2); perm=torch.from_numpy(rng.permutation(BS))
        x,y=lam*x+(1-lam)*x[perm],lam*y+(1-lam)*y[perm]
    return x,y

class PatchD(nn.Module):
    """pix2pix 的 70x70 PatchGAN (n_layers=3), 输入为 (低光图, 增强图) 拼接; 为 CPU 预算 ndf 取 32"""
    def __init__(s,ndf=32):
        super().__init__(); c=[6,ndf,ndf*2,ndf*4,ndf*8]; L_=[nn.Conv2d(6,ndf,4,2,1),nn.LeakyReLU(0.2,True)]
        for k in range(1,4): L_+=[nn.Conv2d(c[k],c[k+1],4,2 if k<3 else 1,1,bias=False),nn.BatchNorm2d(c[k+1]),nn.LeakyReLU(0.2,True)]
        L_+=[nn.Conv2d(c[4],1,4,1,1)]; s.m=nn.Sequential(*L_)
    def forward(s,x,y): return s.m(torch.cat([x,y],1))

torch.manual_seed(0); G=build_rf(); G.train(); optG=torch.optim.Adam(G.parameters(),LR0,betas=(0.9,0.999))
D=PatchD() if a.arm=='gan' else None; optD=torch.optim.Adam(D.parameters(),2e-4,betas=(0.5,0.999)) if D else None
lr_at=lambda it: LR1+(LR0-LR1)*0.5*(1+np.cos(np.pi*it/a.iters))
rng=np.random.RandomState(0); start=0; ck=f'{OUT}/last.pth'
if os.path.exists(ck):
    s=torch.load(ck,map_location='cpu',weights_only=False); G.load_state_dict(s['G']); optG.load_state_dict(s['optG']); rng.set_state(s['rng']); start=s['it']
    if D: D.load_state_dict(s['D']); optD.load_state_dict(s['optD'])
    torch.set_rng_state(s['trng']); print(f'从第 {start} 步续训',flush=True)
def save(it):
    s=dict(G=G.state_dict(),optG=optG.state_dict(),rng=rng.get_state(),trng=torch.get_rng_state(),it=it)
    if D: s.update(D=D.state_dict(),optD=optD.state_dict())
    torch.save(s,ck+'.tmp'); os.replace(ck+'.tmp',ck)
    if it%500==0: torch.save({'params':G.state_dict()},f'{OUT}/G_{it}.pth')

t0=time.time(); log=[]
for it in range(start+1,a.iters+1):
    for o in [optG]+([optD] if D else []): o.param_groups[0]['lr']=lr_at(it-1)*(2 if o is optD else 1)
    x,y=batch(rng); out=G(x); l1=F.l1_loss(out,y); loss=l1; ladv=ld=torch.tensor(0.)
    if D:
        ld=0.5*(((D(x,y)-1)**2).mean()+(D(x,out.detach())**2).mean()); optD.zero_grad(); ld.backward(); optD.step()
        ladv=((D(x,out)-1)**2).mean(); loss=l1+LAM*ladv
    optG.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(G.parameters(),0.01); optG.step()
    log.append([l1.item(),ladv.item(),ld.item()])
    if it%50==0 or it==a.iters:
        m=np.mean(log,0); log=[]
        print(f'{a.arm} it {it:5d}  L1 {m[0]:.4f}  adv {m[1]:.3f}  D {m[2]:.3f}  lr {lr_at(it-1):.2e}  {(time.time()-t0)/(it-start):.2f}s/it',flush=True)
    if it%250==0 or it==a.iters: save(it)
print(f'{a.arm} 完成',flush=True)
