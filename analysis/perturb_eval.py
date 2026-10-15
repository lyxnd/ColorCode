# 用法: python perturb_eval.py <方法输出目录> <GT目录>
# 输出目录里应有 {id}_{ref,warm,cool,green,dark,bright}.png (对 assets/perturb 中的输入推理得到)
# 指标: 每种扰动相对 ref 输出的 Δa*, Δb*, ΔL, ΔE00。理想的色彩恒常方法: 白平衡扰动下 Δa*/Δb* ≈ 0, 曝光扰动下 ΔL ≈ 0
import sys, numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
out, gtd = sys.argv[1], sys.argv[2]
ids=['00739','00749','00750','00751','00752']
L=lambda p: rgb2lab(np.asarray(Image.open(p).convert('RGB')).astype(float)/255)
# 输入端施加的色偏大小(作为参照): 对 ref 输入做同样增益后的 a*b* 变化
for k in ['warm','cool','green','dark','bright']:
    d=[]
    for i in ids:
        r=L(f'{out}/{i}_ref.png'); p=L(f'{out}/{i}_{k}.png')
        dm=(p-r).reshape(-1,3).mean(0); d.append([dm[0],dm[1],dm[2],deltaE_ciede2000(r,p).mean()])
    d=np.mean(d,0)
    print(f'{k:7s} ΔL={d[0]:+5.2f} Δa*={d[1]:+5.2f} Δb*={d[2]:+5.2f} ΔE00(vs ref)={d[3]:5.2f}')
