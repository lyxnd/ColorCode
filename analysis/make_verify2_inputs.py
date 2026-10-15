# 第二轮验证输入 -> assets/verify2/
# 实验5 (模型本身是否压低饱和度): 由GT合成, 与真实low同亮度, 但不带色偏 (单一标量缩放)
#   {id}_cleanwb.png  无噪声 + 无色偏
#   {id}_noisywb.png  有噪声 + 无色偏   (与第一轮 clean/noisy 合起来构成 色偏x噪声 2x2)
# 实验6 (曝光是否导致色调不一致): 把同一GT的第二张输入缩放到第一张的整体亮度
#   00750_exp.png / 00752_exp.png      原输入, 亮度对齐到 00749 / 00751
#   00750_wbexp.png / 00752_wbexp.png  白平衡后的输入, 亮度对齐到 00749_wb / 00751_wb
import numpy as np, os
from PIL import Image
from scipy.ndimage import gaussian_filter
A='assets/'; O='assets/verify2/'; os.makedirs(O,exist_ok=True)
ids=['00739','00749','00750','00751','00752']
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
def q(x): return np.clip(np.round(x*255),0,255).astype(np.uint8)
hp=lambda x: (x-gaussian_filter(x,(1.5,1.5,0))).reshape(-1,3).var(0)
rng=np.random.default_rng(1)
for i in ids:
    lo=L(A+f'low/{i}.png'); g=L(A+f'gt/{i}.png'); ll,gl=lin(lo),lin(g)
    dark=gl*(ll.mean()/gl.mean())                     # 单一标量: 同亮度, 不改通道比例 -> 无色偏
    d8=srgb(dark)*255; c8=np.round(d8)
    Image.fromarray(q(srgb(dark))).save(O+f'{i}_cleanwb.png')
    sig=np.sqrt(np.maximum(hp(lo*255)-hp(c8),0))/0.95  # 与第一轮相同的噪声强度估计
    Image.fromarray(np.clip(np.round(d8+rng.normal(0,1,d8.shape)*sig),0,255).astype(np.uint8)).save(O+f'{i}_noisywb.png')
    print(i,'cleanwb 均值RGB',np.round(srgb(dark).reshape(-1,3).mean(0)*255,1),' 噪声σ',np.round(sig,2))
for a,b in [('00749','00750'),('00751','00752')]:
    for src,tag in [('low/{}.png','exp'),('verify/{}_wb.png','wbexp')]:
        la=lin(L(A+src.format(a))); lb=lin(L(A+src.format(b)))
        k=la.mean()/lb.mean(); Image.fromarray(q(srgb(lb*k))).save(O+f'{b}_{tag}.png')
        print(f'{b}_{tag}: 亮度缩放系数 {k:.3f} (对齐到 {a})')
