# 色度对比实验: 全测试集(100张)评估 + 难例子集分析 + 训练日志
import os, re, numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity
R='Retinexformer/results_color_exp'; G='assets/gt'
groups=['orig','A','A_seed200','B','B_seed200','C','D']
names=sorted(os.listdir(G))
rgb=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
C=lambda l: np.hypot(l[...,1],l[...,2])
per={g:{} for g in groups}
for n in names:
    g_rgb=rgb(f'{G}/{n}'); g_lab=rgb2lab(g_rgb); mk=(C(g_lab)>30)&(g_lab[...,0]<45)
    for g in groups:
        o=rgb(f'{R}/{g}/test/{n}'); l=rgb2lab(o)
        per[g][n]=dict(psnr=10*np.log10(1/np.mean((o-g_rgb)**2)),
                       ssim=structural_similarity(o,g_rgb,channel_axis=2,data_range=1.0),
                       de=deltaE_ciede2000(g_lab,l).mean(),
                       num=C(l)[mk].sum(), den=C(g_lab)[mk].sum(), npix=int(mk.sum()))
def agg(g,sub):
    d=[per[g][n] for n in sub]
    return (np.mean([x['psnr'] for x in d]),np.mean([x['ssim'] for x in d]),np.mean([x['de'] for x in d]),
            sum(x['num'] for x in d)/max(sum(x['den'] for x in d),1e-9))
print('== 全测试集 100 张')
print(f'{"组":10s} {"PSNR":>6s} {"SSIM":>6s} {"ΔE00":>6s} {"暗饱和保持":>8s}')
for g in groups: print(f'{g:10s} %6.2f %6.3f %6.2f %8.3f'%agg(g,names))
# 难例: 官方权重在该图暗+饱和像素上保持率<0.8, 且这类像素>=2000
hard=[n for n in names if per['orig'][n]['npix']>=2000 and per['orig'][n]['num']/per['orig'][n]['den']<0.8]
print(f'\n== 难例子集: 官方权重暗饱和保持率<0.8 的图 ({len(hard)} 张): {" ".join(x[:-4] for x in hard)}')
print(f'{"组":10s} {"PSNR":>6s} {"SSIM":>6s} {"ΔE00":>6s} {"暗饱和保持":>8s}')
for g in groups: print(f'{g:10s} %6.2f %6.3f %6.2f %8.3f'%agg(g,hard))
print('\n== 难例逐图暗饱和保持率')
print('图     '+' '.join(f'{g:>9s}' for g in groups))
for n in hard: print(f'{n[:-4]} '+' '.join(f'{per[g][n]["num"]/per[g][n]["den"]:9.2f}' for g in groups))
print('\n== 训练日志: 验证 PSNR')
for t in ['A','A_seed200','B','B_seed200','C','D']:
    s=open(f'Retinexformer/logs_color_exp/{t}.log',encoding='utf-8',errors='ignore').read()
    v=[float(x) for x in re.findall(r'Validation ValSet,\s+# psnr: ([\d.]+)',s)]
    print(f'{t:10s} 最后={v[-1]:.2f} 后10次均值={np.mean(v[-11:-1]):.2f}±{np.std(v[-11:-1]):.2f} 最佳={max(v):.2f}')
