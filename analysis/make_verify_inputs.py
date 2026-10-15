# 生成验证实验的输入图 -> assets/verify/
#   {id}_wb.png    实验1/4: 用GT求的逐通道增益做白平衡(线性光), 保持整体亮度不变, 只去掉全局色偏
#   00739_ctx.png  实验2: 只对黄墙以外的区域做白平衡, 黄墙像素保持原样 (换"背景色调")
#   {id}_clean.png 实验3: 由GT合成, 与真实low同亮度同色偏, 量化到8bit, 无噪声
#   {id}_noisy.png 实验3: 同上, 但加入与真实low相当强度的噪声
import numpy as np, os
from PIL import Image
from skimage.color import rgb2lab
from scipy.ndimage import median_filter, gaussian_filter
A='assets/'; O='assets/verify/'; os.makedirs(O,exist_ok=True)
ids=['00739','00749','00750','00751','00752']
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
def q(x): return np.clip(np.round(x*255),0,255).astype(np.uint8)
rng=np.random.default_rng(0)
for i in ids:
    lo=L(A+f'low/{i}.png'); g=L(A+f'gt/{i}.png'); ll,gl=lin(lo),lin(g)
    gain=(gl*ll).reshape(-1,3).sum(0)/(ll*ll).reshape(-1,3).sum(0)     # 低光->GT 的逐通道增益
    wb=gain/gain.mean()                                                  # 只保留通道间比例 = 白平衡
    wb=wb*(ll.mean()/ (ll*wb).mean())                                    # 保持整体亮度不变
    Image.fromarray(q(srgb(ll*wb))).save(O+f'{i}_wb.png')
    # 实验3: GT 线性 -> 按通道缩放到与真实low相同的通道均值 (同亮度、同色偏)
    scale=ll.reshape(-1,3).mean(0)/gl.reshape(-1,3).mean(0)
    dark=gl*scale
    Image.fromarray(q(srgb(dark))).save(O+f'{i}_clean.png')
    # 噪声强度: 用真实low的高频残差(MAD)估计, 在sRGB 8bit域逐通道
    # 噪声强度: 真实low 与 无噪声合成图 的高频能量之差 (高斯高通, 逐通道, 8bit 域)
    lo8=lo*255; d8=srgb(dark)*255; c8=np.round(d8)
    hp=lambda x: (x-gaussian_filter(x,(1.5,1.5,0))).reshape(-1,3).var(0)
    sig=np.sqrt(np.maximum(hp(lo8)-hp(c8),0))/0.95
    noisy=np.clip(np.round(d8+rng.normal(0,1,d8.shape)*sig),0,255).astype(np.uint8)
    Image.fromarray(noisy).save(O+f'{i}_noisy.png')
    print(i,'wb增益(R,G,B)=(%.2f,%.2f,%.2f)'%tuple(wb),' 噪声σ(8bit)=(%.2f,%.2f,%.2f)'%tuple(sig),
          ' low均值=%s clean均值=%s'%(np.round(lo8.reshape(-1,3).mean(0),1),np.round(srgb(dark).reshape(-1,3).mean(0)*255,1)))
    if i=='00739':
        lab=rgb2lab(g); C=np.hypot(lab[...,1],lab[...,2])
        wall=((C>30)&(lab[...,0]<50)&(lab[...,2]>20)).astype(float)
        m=np.clip(gaussian_filter(wall,2)*1.5,0,1)[...,None]
        ctx=srgb(ll*wb)*(1-m)+lo*m
        Image.fromarray(q(ctx)).save(O+'00739_ctx.png'); np.save('analysis/wall_mask_00739.npy',wall>0)
        print('黄墙掩码像素数',int(wall.sum()))
