# 第二轮验证评估. 把 assets/verify2/ 下 14 张图用各方法推理, 输出放到 assets/verify2_out/<方法>/ (文件名同输入)
# 用法: python analysis/verify2_eval.py
import numpy as np, os
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
ids=['00739','00749','00750','00751','00752']
D={'HVI':'hvi','Retinexformer':'retinexformer','Diff-Retinex':'diff_retinex','ReDDiT':'reddit'}
SUF={'ReDDiT':'_normal'}
def path(m,name):
    d=D[m]; s=SUF.get(m,'')
    for root in ['assets/verify2_out','assets/verify_out']:
        p=f'{root}/{d}/{name}{s}.png'
        if os.path.exists(p): return p
    if name.endswith('_ref'):
        p=f'assets/perturb_out/{d}/{name}{s}.png'
        if os.path.exists(p): return p
    return None
def lab(m,name):
    p=path(m,name)
    if p is None: return None
    im=Image.open(p).convert('RGB')
    if im.size!=(600,400): im=im.resize((600,400),Image.BICUBIC)
    return rgb2lab(np.asarray(im).astype(float)/255)
LG={i:rgb2lab(np.asarray(Image.open(f'assets/gt/{i}.png').convert('RGB')).astype(float)/255) for i in ids}
C=lambda l: np.hypot(l[...,1],l[...,2])
def ret(m,kind,mask='darksat'):
    num=den=0
    for i in ids:
        l=lab(m,f'{i}_{kind}')
        if l is None: return None
        mk=(C(LG[i])>30)&(LG[i][...,0]<45) if mask=='darksat' else np.ones(l.shape[:2],bool)
        num+=C(l)[mk].sum(); den+=C(LG[i])[mk].sum()
    return num/den
print('== 实验5: 色偏 x 噪声 2x2, 色度保持率 (5张合计)')
print('   预测: 若模型本身压低饱和度, "无噪声+无色偏"仍明显 < 1')
for mask,title in [('darksat','暗+饱和像素'),('all','全部像素')]:
    print(f'  [{title}]')
    for m in D:
        r={k:ret(m,k,mask) for k in ['clean','noisy','cleanwb','noisywb','ref']}
        f=lambda v: ' -  ' if v is None else f'{v:.2f}'
        print(f'  {m:14s} 无噪声+无色偏={f(r["cleanwb"])}  无噪声+有色偏={f(r["clean"])}  有噪声+无色偏={f(r["noisywb"])}  有噪声+有色偏={f(r["noisy"])}  真实低光={f(r["ref"])}')
print('\n== 实验6: 同一GT两张输入的输出互差 ΔE00')
print('   预测: 若曝光是原因, 亮度对齐(exp/wbexp)后互差应下降')
for m in D:
    s=[]
    for a,b in [('00749','00750'),('00751','00752')]:
        for ka,kb,t in [('ref','ref','原始'),('ref','exp','亮度对齐'),('wb','wb','白平衡'),('wb','wbexp','白平衡+亮度对齐')]:
            x,y=lab(m,f'{a}_{ka}'),lab(m,f'{b}_{kb}')
            s.append(f'{t}={deltaE_ciede2000(x,y).mean():.2f}' if x is not None and y is not None else f'{t}=-')
        s.append('|')
    print(f'  {m:14s} '+' '.join(s))
