# 色度对比实验: 在之前分析的 5 张图(有GT)和第一轮合成验证图上逐组比较
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
R='Retinexformer/results_color_exp'; ids=['00739','00749','00750','00751','00752']
groups=['orig','A','B','C','D']
load=lambda p: rgb2lab(np.asarray(Image.open(p).convert('RGB')).astype(float)/255)
C=lambda l: np.hypot(l[...,1],l[...,2])
GT={i:load(f'assets/gt/{i}.png') for i in ids}
wall=np.load('analysis/wall_mask_00739.npy')
def ret(g,kind):
    num=den=0; e=[]
    for i in ids:
        o=load(f'{R}/{g}/test/{i}.png' if kind=='test' else f'{R}/{g}/verify2/{i}_{kind}.png')
        mk=(C(GT[i])>30)&(GT[i][...,0]<45); num+=C(o)[mk].sum(); den+=C(GT[i])[mk].sum()
        e.append(deltaE_ciede2000(GT[i],o).mean())
    return num/den, np.mean(e)
print('== 之前分析的 5 张图 (真实低光输入)')
print(f'{"组":5s} 暗饱和保持  ΔE00   楼梯黄墙保持')
for g in groups:
    r,e=ret(g,'test'); o=load(f'{R}/{g}/test/00739.png')
    print(f'{g:5s} {r:8.2f} {e:7.2f} {C(o)[wall].mean()/C(GT["00739"])[wall].mean():10.2f}')
print('\n== 第一轮合成验证图: 暗+饱和色度保持率 (5张合计)')
print(f'{"组":5s} {"无噪声(带色偏)":>12s} {"加噪声(带色偏)":>12s} {"去色偏":>8s}')
for g in groups:
    print(f'{g:5s} {ret(g,"clean")[0]:12.2f} {ret(g,"noisy")[0]:12.2f} {ret(g,"wb")[0]:8.2f}')
