# 免训练检验: "全局颜色听网络的, 局部色度关系听输入的"
# 输出 = 网络输出的亮度 × (低光输入模糊后的色度坐标 × 由网络输出估计的逐通道增益)
# 增益只用网络输出和输入计算, 不用 GT
import os, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab, deltaE_ciede2000
G='assets/gt'; LOW='assets/low_full'
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
W=np.array([0.2126,0.7152,0.0722])
def fuse(out, low, sigma=2, alpha=1.0):
    o=lin(out); b=gaussian_filter(lin(low),(sigma,sigma,0))
    gain=o.reshape(-1,3).mean(0)/np.maximum(b.reshape(-1,3).mean(0),1e-6)
    c=b*gain; y_o=(o*W).sum(2,keepdims=True); y_c=np.maximum((c*W).sum(2,keepdims=True),1e-6)
    new=c*(y_o/y_c)
    return srgb(alpha*new+(1-alpha)*o)
C=lambda l: np.hypot(l[...,1],l[...,2])

if __name__=='__main__':
    ids=['00739','00749','00750','00751','00752']; wall=np.load('analysis/wall_mask_00739.npy')
    R='Retinexformer/results_color_exp'
    print('组/设置                  5张ΔE00  5张暗饱和保持  黄墙保持  5张PSNR')
    for g in ['orig','A','B']:
        for sigma,alpha in [(None,None),(2,1.0),(2,0.5),(4,1.0)]:
            e=[];num=den=0;p=[]
            for i in ids:
                out=L(f'{R}/{g}/test/{i}.png'); gt=L(f'{G}/{i}.png'); low=L(f'assets/low/{i}.png')
                res=out if sigma is None else fuse(out,low,sigma,alpha)
                lg=rgb2lab(gt); lr=rgb2lab(res)
                e.append(deltaE_ciede2000(lg,lr).mean()); p.append(10*np.log10(1/np.mean((res-gt)**2)))
                mk=(C(lg)>30)&(lg[...,0]<45); num+=C(lr)[mk].sum(); den+=C(lg)[mk].sum()
                if i=='00739': w=C(lr)[wall].mean()/C(lg)[wall].mean()
            tag='原输出' if sigma is None else f'色度引导 σ={sigma} α={alpha}'
            print(f'{g:5s} {tag:20s} {np.mean(e):7.2f} {num/den:12.2f} {w:9.2f} {np.mean(p):8.2f}')
