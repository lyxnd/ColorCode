# 用"全局色调"而非逐像素ΔE, 重新评估同一GT两张输入的输出差异
# 指标: 两张输出的全图平均 a*, b* 之差 |Δa*|+|Δb*| 记作 色调差; 以及 平均L*之差
import numpy as np, os
from PIL import Image
from skimage.color import rgb2lab
D={'HVI':'hvi','Retinexformer':'retinexformer','Diff-Retinex':'diff_retinex','ReDDiT':'reddit'}
SUF={'ReDDiT':'_normal'}
def lab(m,name):
    for root in ['assets/perturb_out','assets/verify_out','assets/verify2_out','assets/verify3_out']:
        p=f'{root}/{D[m]}/{name}{SUF.get(m,"")}.png'
        if os.path.exists(p):
            im=Image.open(p).convert('RGB')
            if im.size!=(600,400): im=im.resize((600,400),Image.BICUBIC)
            return rgb2lab(np.asarray(im).astype(float)/255).reshape(-1,3).mean(0)
    return None
def diff(x,y): d=x-y; return f'色调差{abs(d[1])+abs(d[2]):4.2f}(Δa*{d[1]:+.1f},Δb*{d[2]:+.1f}) ΔL{d[0]:+.1f}'
for m in D:
    print(m)
    for a,b in [('00749','00750'),('00751','00752')]:
        for ka,kb,t in [('ref','ref','原始     '),('wb','wb','白平衡   '),('ref','exp','亮度对齐 '),('wb','wbexp','白平衡+亮度')]:
            x,y=lab(m,f'{a}_{ka}'),lab(m,f'{b}_{kb}')
            print(f'  {a}/{b} {t} '+(diff(x,y) if x is not None and y is not None else '-'))
    s=[]
    for i in ['00739','00749','00750','00751','00752']:
        x,y=lab(m,f'{i}_seedA'),lab(m,f'{i}_seedB')
        if x is not None: d=x-y; s.append(abs(d[1])+abs(d[2]))
    print(f'  换噪声种子 色调差 均值 {np.mean(s):.2f}')
