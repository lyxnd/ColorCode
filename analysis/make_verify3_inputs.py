# 第三轮验证 (实验7: 输出是否对噪声细节敏感) -> assets/verify3/
#   {id}_seedA.png  由GT合成的暗图(与真实low同亮度同色偏) + 噪声(随机种子A)
#   {id}_seedB.png  同一张暗图 + 同强度噪声(随机种子B)   -> 只有噪声的具体样本不同
#   {id}_seedA2.png 与 seedA 完全相同的副本              -> 测模型自身推理随机性 (对照)
import numpy as np, os, shutil
from PIL import Image
from scipy.ndimage import gaussian_filter
A='assets/'; O='assets/verify3/'; os.makedirs(O,exist_ok=True)
ids=['00739','00749','00750','00751','00752']
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
hp=lambda x: (x-gaussian_filter(x,(1.5,1.5,0))).reshape(-1,3).var(0)
for i in ids:
    lo=L(A+f'low/{i}.png'); g=L(A+f'gt/{i}.png'); ll,gl=lin(lo),lin(g)
    dark=gl*(ll.reshape(-1,3).mean(0)/gl.reshape(-1,3).mean(0))
    d8=srgb(dark)*255
    sig=np.sqrt(np.maximum(hp(lo*255)-hp(np.round(d8)),0))/0.95
    for tag,seed in [('seedA',100),('seedB',200)]:
        n=np.random.default_rng(seed).normal(0,1,d8.shape)*sig
        Image.fromarray(np.clip(np.round(d8+n),0,255).astype(np.uint8)).save(O+f'{i}_{tag}.png')
    shutil.copyfile(O+f'{i}_seedA.png',O+f'{i}_seedA2.png')
    a=np.asarray(Image.open(O+f'{i}_seedA.png')).astype(float); b=np.asarray(Image.open(O+f'{i}_seedB.png')).astype(float)
    print(i,'噪声σ',np.round(sig,2),' 两个种子输入之间平均绝对差(8bit)=%.2f'%np.abs(a-b).mean())
