# E1 留出法全局拟合: 黄墙变灰是"整张图的白平衡/色调偏了"还是"只有墙这块变灰"?
#   只用墙以外的像素拟合一个全局颜色变换, 再作用到黄墙上:
#   - 若是全局去黄(白平衡误判), 墙外像素也被去黄 -> 拟合出"加黄"的变换 -> 墙恢复, 且墙外中性像素偏蓝
#   - 若是局部变灰, 墙外像素基本是对的 -> 变换接近恒等 -> 墙仍灰, 中性像素不偏
#   色度用"区域在线性光下取平均后的 C*", 不受逐像素噪声抬高色度的影响.
#   3x3+偏置 的留出拟合外推时本身会去饱和 (ReDDiT 也会掉 0.1-0.2), 逐通道增益的留出拟合对 ReDDiT 无偏, 以它为准.
# GT 与低光输入取自 ReDDiT 测试结果目录 (按文件名排序, 序号 k 对应 id 00689+k; 低光图去掉上下 8、左右 4 像素的 padding).
# 用法 (仓库根目录): python analysis/holdout_fit_00739.py
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from scipy.ndimage import binary_dilation, distance_transform_edt
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
RX='Retinexformer/results_color_exp'
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
METHODS={'Retinexformer':'orig','RF-A':'A','RF-A200':'A_seed200','RF-B':'B','RF-B200':'B_seed200','RF-C':'C','RF-D':'D','Diff-Retinex':'Diff-Retinex','ReDDiT':'ReDDiT'}
def path(m,i):
    d=METHODS[m]
    if d=='ReDDiT': return f'{RD}/output/{int(i)-689}_normal.png'
    if d=='Diff-Retinex': return f'Diff-Retinex/experiments/Diff_Retinex_val_261008_104214/results/{i}.png'
    return f'{RX}/{d}/test/{i}.png'
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
def chroma_of_mean(img,mask): l=rgb2lab(srgb(lin(img)[mask].mean(0))[None,None])[0,0]; return np.hypot(l[1],l[2])
B=8  # 8x8 块平均后再拟合, 减小输出噪声导致的斜率衰减
def blk(x): h,w=x.shape[0]//B*B,x.shape[1]//B*B; return x[:h,:w].reshape(h//B,B,w//B,B,-1).mean((1,3))

i='00739'; g=L(f'{RD}/gt/{int(i)-689}_gt.png'); lg=rgb2lab(g)
wall=np.load('analysis/wall_mask_00739.npy'); near=binary_dilation(wall,iterations=15)
neutral=(np.hypot(lg[...,1],lg[...,2])<6)&~near
fm=blk((~near)[...,None].astype(float))[...,0]>0.999
Cg=chroma_of_mean(g,wall)
print(f'黄墙像素 {wall.sum()}, 墙外(扩 15px 后)用于拟合的 8x8 块 {fm.sum()}, 墙外 GT 中性像素 {neutral.sum()}; GT 黄墙区域均值色度 C*={Cg:.1f}\n')
print('方法            原始 | 墙外拟合: 逐通道增益 | 墙外拟合: 3x3+偏置(参考) || 墙外中性像素 Δa*   Δb*')
for m in METHODS:
    o=L(path(m,i)); ob,gb=blk(lin(o)),blk(lin(g))
    k=(ob[fm]*gb[fm]).sum(0)/(ob[fm]**2).sum(0)
    X=np.c_[ob[fm],np.ones(fm.sum())]; M=np.linalg.lstsq(X,gb[fm],rcond=None)[0]
    o_aff=srgb(np.clip(np.c_[lin(o).reshape(-1,3),np.ones(o.size//3)]@M,0,1).reshape(o.shape))
    dn=(rgb2lab(o)-lg)[neutral].mean(0)
    print(f'{m:14s}  {chroma_of_mean(o,wall)/Cg:.2f} |        {chroma_of_mean(srgb(lin(o)*k),wall)/Cg:.2f}          |          {chroma_of_mean(o_aff,wall)/Cg:.2f}            ||        {dn[1]:+5.2f} {dn[2]:+5.2f}')

print('\n== 墙内按"到墙边界的距离"分层的色度保持率')
d=distance_transform_edt(wall); bins=[(1,5),(5,15),(15,30),(30,60)]; mks=[(d>=a)&(d<b) for a,b in bins]
print('方法           '+'  '.join(f'{a:>3}-{b:<3}px' for a,b in bins))
print('像素数         '+'  '.join(f'{x.sum():>9d}' for x in mks))
for m in METHODS:
    o=L(path(m,i))
    print(f'{m:14s} '+'  '.join(f'{chroma_of_mean(o,x)/chroma_of_mean(g,x):9.2f}' for x in mks))
