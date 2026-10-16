# eval.py 之外的两项核对
#   1) 各组相对 l1@2000 改变了多少 (测试集每 10 张取 1 张, 平均逐像素差): 若 gan 组的变化不超过 l1 自身两个存档之间的差, 说明对抗项几乎没起作用
#   2) 00739 黄墙的 Lab 与色相 (保持率只看色度大小, 不看色相)
# 用法 (仓库根目录): python analysis/loss_data_ft/extra_checks.py   结果: analysis/loss_data_ft/extra_checks.txt
import sys, os, numpy as np, torch
sys.path.insert(0,os.path.dirname(__file__)); from common import *
torch.set_num_threads(4); R='analysis/loss_data_ft/runs'
ck=lambda k: 'Retinexformer/pretrained_weights/LOL_v2_real.pth' if k=='官方' else f'{R}/{k.replace("@","/G_")}.pth'
nets={k:build_rf(ck(k)).eval() for k in ['官方','l1@1000','l1@2000','gan@2000','bias@2000']}
lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
P('== 1) 输出变化量 (测试集 10 张, 原始输入)')
d={}
for i in IDS[::10]:
    o={k:run(n,low(i)) for k,n in nets.items()}
    for a,b in [('l1@1000','l1@2000'),('gan@2000','l1@2000'),('bias@2000','l1@2000')]: d.setdefault((a,b),[]).append(np.abs(o[a]-o[b]).mean()*255)
for (a,b),v in d.items(): P(f'  {a:>9s} vs {b}: 平均逐像素差 {np.mean(v):.2f} DN')
P('\n== 2) 00739 黄墙 (原始输入)')
wall=np.load('analysis/wall_mask_00739.npy')
def wl(o): a=lab(lin(o)[wall].mean(0)[None])[0]; return a,np.degrees(np.arctan2(a[2],a[1]))
a0,h0=wl(gt('00739')); P(f'  {"GT":9s} Lab=({a0[0]:.0f},{a0[1]:+.1f},{a0[2]:+.1f}) 色相 {h0:.1f}°')
for k,n in nets.items():
    a,h=wl(run(n,low('00739'))); P(f'  {k:9s} Lab=({a[0]:.0f},{a[1]:+.1f},{a[2]:+.1f}) 色相 {h:.1f}° (差 {h-h0:+.1f}°)')
open('analysis/loss_data_ft/extra_checks.txt','w').write('\n'.join(lines)+'\n')
