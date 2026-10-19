# 同一张输入只改曝光, 输出色调会不会变? (对齐干预里 "只对齐曝光" 反而让同场景输出更不一致, 这里排除两次采集的差别, 单独检验)
#   每张测试图在线性光下乘 k = 0.5 / 1 / 2 (浮点, 不再量化; 色度比例不变), 推理后看:
#     输出全图平均 a*b* 相对 k=1 的变化 (色调差)
#     输入色偏的传递比例: 输入色偏 = 输入提亮到 GT 亮度后的平均 a*b* - GT 平均 a*b*; 输出色偏 = 输出平均 a*b* - GT 平均 a*b*;
#       传递比例 = 输出色偏在输入色偏方向上的投影 / 输入色偏大小 (1 = 照搬输入色偏, 0 = 完全去掉)
# 用法 (仓库根目录): python analysis/exposure_tone_check.py   结果: analysis/exposure_tone_check.txt
import sys, os, numpy as np, torch
from skimage.color import rgb2lab
sys.path.insert(0,'analysis/loss_data_ft'); from common import gt, low, IDS, lin, srgb, W, build_rf, run, q8
torch.set_num_threads(int(os.environ.get('THREADS',4)))
rf=build_rf('Retinexformer/pretrained_weights/LOL_v2_real.pth').eval()
sys.path.insert(0,'HVI-CIDNet'); from net.CIDNet import CIDNet
hvi=CIDNet(); hvi.load_state_dict(torch.load('HVI-CIDNet/weights/best_PSNR.pth',map_location='cpu',weights_only=False)); hvi.eval(); hvi.trans.gated2=True; hvi.trans.alpha=0.8
@torch.no_grad()
def hvi_run(x): return q8(hvi(torch.from_numpy(x.transpose(2,0,1).copy()).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0))
ab=lambda x: rgb2lab(np.clip(x,0,1))[...,1:].reshape(-1,2).mean(0)
K=[0.5,1,2]; lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
P('同一张输入只改曝光 (线性光 x k), 测试集 100 张')
P(f'{"模型":14s} | 输出色调差 (相对 k=1) 中位 / 均值: k=0.5      k=2      | 输入色偏传递比例中位数: k=0.5   k=1   k=2   (只统计输入色偏 >= 3 的图)')
for name,fn in [('Retinexformer',lambda x: run(rf,x)),('HVI-CIDNet',hvi_run)]:
    d={k:[] for k in K}; tr={k:[] for k in K}
    for i in IDS:
        lo,g=low(i),gt(i); gab=ab(g); ll=lin(lo)
        cin=ab(srgb(ll*(lin(g)@W).mean()/max((ll@W).mean(),1e-6)))-gab
        outs={k:ab(fn(srgb(ll*k).astype(np.float32))) for k in K}
        for k in K:
            d[k].append(np.hypot(*(outs[k]-outs[1])))
            if np.hypot(*cin)>=3: tr[k].append((outs[k]-gab)@cin/(cin@cin))
    P(f'{name:14s} |   {np.median(d[0.5]):.2f} / {np.mean(d[0.5]):.2f}     {np.median(d[2]):.2f} / {np.mean(d[2]):.2f}   |   '
      +'   '.join(f'{np.median(tr[k]):.2f}' for k in K)+f'   ({len(tr[1])} 张)')
open('analysis/exposure_tone_check.txt','w').write('\n'.join(lines)+'\n')
