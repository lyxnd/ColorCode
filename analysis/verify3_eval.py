# 第三轮验证评估. 把 assets/verify3/ 下 15 张图用各方法推理, 输出放 assets/verify3_out/<方法>/ (文件名同输入)
# 用法: python analysis/verify3_eval.py
import numpy as np, os
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
ids=['00739','00749','00750','00751','00752']
D={'HVI':'hvi','Retinexformer':'retinexformer','Diff-Retinex':'diff_retinex','ReDDiT':'reddit'}
SUF={'ReDDiT':'_normal'}
def lab(root,m,name):
    p=f'{root}/{D[m]}/{name}{SUF.get(m,"")}.png'
    if not os.path.exists(p): return None
    im=Image.open(p).convert('RGB')
    if im.size!=(600,400): im=im.resize((600,400),Image.BICUBIC)
    return rgb2lab(np.asarray(im).astype(float)/255)
dE=lambda a,b: deltaE_ciede2000(a,b).mean()
print('== 实验7: 只换噪声样本, 输出变化多少?')
print('   预测: 若色调不一致来自噪声敏感, "换噪声种子"的输出差应接近真实成对图的互差; 且明显大于"同输入重跑"')
for m in D:
    sAB=[];sAA=[];mab=[]
    for i in ids:
        a=lab('assets/verify3_out',m,f'{i}_seedA'); b=lab('assets/verify3_out',m,f'{i}_seedB'); a2=lab('assets/verify3_out',m,f'{i}_seedA2')
        if a is None or b is None: continue
        sAB.append(dE(a,b))
        if a2 is not None: sAA.append(dE(a,a2))
        mab.append(np.abs(a.reshape(-1,3).mean(0)-b.reshape(-1,3).mean(0)))
    real=[]
    for p,q in [('00749','00750'),('00751','00752')]:
        x=lab('assets/perturb_out',m,f'{p}_ref'); y=lab('assets/perturb_out',m,f'{q}_ref')
        if x is not None and y is not None: real.append(dE(x,y))
    f=lambda v: ' - ' if not v else f'{np.mean(v):.2f} (逐图 '+' '.join(f'{x:.1f}' for x in v)+')'
    mm=np.mean(mab,0) if mab else [np.nan]*3
    print(f'  {m:14s} 换噪声种子={f(sAB)}\n  {"":14s} 同输入重跑={f(sAA)}  真实成对图互差={f(real)}  换种子后全图平均L*a*b*变化=({mm[0]:.2f},{mm[1]:.2f},{mm[2]:.2f})')
