# 对照论文 (paper/) 与代码的几处核对
#   1) ReDDiT 的 RATR "反射率" x~0 = (y - z')/h' (论文 3.3 节, h' = 最大通道): 代码 diffusion.loss 中 y 是 [-1,1] 的低光图, T = max(y)+0.1
#      -> 在 00739 黄墙上实际算出的值 (代码口径 vs 论文的 [0,1] 口径)
#   2) ReDDiT 论文表 1 的 LOLv2-real PSNR (Retinexformer 27.71, ReDDiT-4 31.25) 是否为 GT 均值亮度校正后的口径
#   3) 00739 墙外 GT 中性像素的色偏 (CWNet 训练用了带逐通道 RGB 偏移 +-50 的"颜色异常"负样本)
# 用法 (仓库根目录): python analysis/paper_checks.py
import subprocess, io, numpy as np, torch, torchvision.transforms as Tv
from PIL import Image
from skimage.color import rgb2lab
from scipy.ndimage import binary_dilation
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
IDS=[f'{690+k:05d}' for k in range(100)]
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
gt=lambda i: L(f'{RD}/gt/{int(i)-689}_gt.png'); low=lambda i: L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]
wall=np.load('analysis/wall_mask_00739.npy')

print('== 1) ReDDiT RATR 在 00739 黄墙上的"反射率"')
def snr_map(x):   # 同 ReDDiT/model/ddpm_modules/diffusion.py 的 SNR_map
    b=Tv.GaussianBlur(kernel_size=15,sigma=3)(x); gray=lambda t: t[:,0:1]*0.299+t[:,1:2]*0.587+t[:,2:3]*0.114
    return (gray(b)-gray(x)).abs()
lo=low('00739')
for tag,y0 in [('代码口径 (y 在 [-1,1])',lo*2-1),('论文口径 (y 在 [0,1])',lo)]:
    y=torch.from_numpy(y0.transpose(2,0,1)).float()[None]; T=y.max(1,keepdim=True)[0]+0.1; y=y/T; y=(y-snr_map(y))[0].numpy().transpose(1,2,0)
    print(f'  {tag}: 墙上均值 RGB=({",".join(f"{v:+.3f}" for v in y[wall].mean(0))}), 全图范围 [{y.min():.2f}, {y.max():.2f}]')
print(f'  墙的低光输入 [0,1] 均值 RGB=({",".join(f"{v:.4f}" for v in lo[wall].mean(0))}); 训练脚本 train_lol2_real.sh: --w_snr 0.4 --w_gt 0.0 --w_lpips 0.6')

print('\n== 2) LOLv2-real 全测试集 PSNR: 原始 / GT 均值亮度校正 (输出乘以 GT 与输出灰度均值之比)')
def psnr(a,b): return 10*np.log10(1/np.mean((a-b)**2))
gray=lambda x: (x*[0.299,0.587,0.114]).sum(-1).mean()
for m,f in {'Retinexformer':lambda i: L(f'Retinexformer/results_color_exp/orig/test/{i}.png'),'ReDDiT':lambda i: L(f'{RD}/output/{int(i)-689}_normal.png')}.items():
    r=[(psnr(f(i),gt(i)),psnr(np.clip(f(i)*gray(gt(i))/gray(f(i)),0,1),gt(i))) for i in IDS]
    print(f'  {m:14s} {np.mean([a for a,_ in r]):.2f} / {np.mean([b for _,b in r]):.2f}')
print('  ReDDiT 论文表 1: Retinexformer 27.71, ReDDiT-4 31.25 (Retinexformer 原论文 22.80)')

print('\n== 3) 00739 墙外 GT 中性像素 (C*<6) 的色偏 (输出 - GT)')
g=gt('00739'); lg=rgb2lab(g); neu=(np.hypot(lg[...,1],lg[...,2])<6)&~binary_dilation(wall,iterations=15)
O={'Retinexformer':'Retinexformer/results_color_exp/orig/test/00739.png','HVI':'HVI-CIDNet/output/HVIFirst/00739.png',
   'CWNet':'CWNet-Causal-Wavelet-Network/output/low00739.png','ReDDiT':f'{RD}/output/50_normal.png'}
for m,p in O.items():
    d=(rgb2lab(L(p))-lg)[neu].mean(0); print(f'  {m:14s} Δa*={d[1]:+5.2f} Δb*={d[2]:+5.2f}')
try:
    o=np.asarray(Image.open(io.BytesIO(subprocess.run(['git','show','25a64fa:Diff-Retinex/experiments/Diff_Retinex_val_261008_104214/results/00739.png'],capture_output=True,check=True).stdout)).convert('RGB')).astype(float)/255
    d=(rgb2lab(o)-lg)[neu].mean(0); print(f'  {"Diff-Retinex":14s} Δa*={d[1]:+5.2f} Δb*={d[2]:+5.2f}')
except Exception as e: print('  (取不到 Diff-Retinex 输出:',e,')')
