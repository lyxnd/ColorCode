# 验证实验评估. 先把 assets/verify/ 下 16 张图用各方法推理, 输出放到 assets/verify_out/<方法>/ (文件名同输入)
# 原图(未处理)的结果复用 assets/perturb_out/<方法>/{id}_ref
# 用法: python analysis/verify_eval.py
import numpy as np, os
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
ids=['00739','00749','00750','00751','00752']
M={  # 方法: (verify输出模板, ref输出模板)
 'HVI':          ('assets/verify_out/hvi/{}.png',          'assets/perturb_out/hvi/{}_ref.png'),
 'Retinexformer':('assets/verify_out/retinexformer/{}.png','assets/perturb_out/retinexformer/{}_ref.png'),
 'Diff-Retinex': ('assets/verify_out/diff_retinex/{}.png', 'assets/perturb_out/diff_retinex/{}_ref.png'),
 'ReDDiT':       ('assets/verify_out/reddit/{}_normal.png','assets/perturb_out/reddit/{}_ref_normal.png'),
}
GT={i:np.asarray(Image.open(f'assets/gt/{i}.png').convert('RGB')).astype(float)/255 for i in ids}
LG={i:rgb2lab(GT[i]) for i in ids}
wall=np.load('analysis/wall_mask_00739.npy')
def load(p,i):
    if not os.path.exists(p): return None
    im=Image.open(p).convert('RGB')
    if im.size!=(600,400): im=im.resize((600,400),Image.BICUBIC)   # ReDDiT 非配对脚本会 resize
    return rgb2lab(np.asarray(im).astype(float)/255)
C=lambda l: np.hypot(l[...,1],l[...,2])
def get(m,i,kind):
    v,r=M[m]; return load(r.format(i),i) if kind=='ref' else load(v.format(f'{i}_{kind}'),i)

print('== 实验1/2: 楼梯黄墙 (GT 色度均值 %.1f, b* %.1f)'%(C(LG['00739'])[wall].mean(),LG['00739'][...,2][wall].mean()))
print('   色度保持率 = 输出黄墙C*/GT黄墙C*;  预测: 若歧义成立, wb 与 ctx 应明显高于 ref')
for m in M:
    s=[]
    for k in ['ref','wb','ctx']:
        l=get(m,'00739',k); s.append('   -   ' if l is None else f'{C(l)[wall].mean()/C(LG["00739"])[wall].mean():.2f}(b*{l[...,2][wall].mean():+.0f})')
    print(f'  {m:14s} 原图={s[0]}  全图去色偏={s[1]}  仅背景去色偏={s[2]}')

print('\n== 实验3: 暗+饱和像素(GT C*>30 且 L<45) 色度保持率, 5 张合计')
print('   预测: 若噪声是重要原因, clean 应明显高于 noisy; noisy 应接近 ref(真实低光)')
for m in M:
    out=[]
    for k in ['clean','noisy','ref']:
        num=den=0; ok=True
        for i in ids:
            l=get(m,i,k)
            if l is None: ok=False; break
            mk=(C(LG[i])>30)&(LG[i][...,0]<45); num+=C(l)[mk].sum(); den+=C(LG[i])[mk].sum()
        out.append(f'{num/den:.2f}' if ok else ' -  ')
    print(f'  {m:14s} 无噪声={out[0]}  加噪声={out[1]}  真实低光={out[2]}')

print('\n== 实验4: 同一GT的两张输入, 输出之间的 ΔE00 (以及各自与GT的 ΔE00)')
print('   预测: 若差异来自输入白平衡, wb 后两张输出之间的 ΔE 应明显下降')
for m in M:
    s=[]
    for p,q in [('00749','00750'),('00751','00752')]:
        for k in ['ref','wb']:
            a,b=get(m,p,k),get(m,q,k)
            if a is None or b is None: s.append(f'{p}/{q} {k}: -'); continue
            s.append(f'{p}/{q} {k}: 互差{deltaE_ciede2000(a,b).mean():4.2f} 对GT {deltaE_ciede2000(LG[p],a).mean():4.2f}/{deltaE_ciede2000(LG[q],b).mean():4.2f}')
    print(f'  {m:14s} '+' | '.join(s))
