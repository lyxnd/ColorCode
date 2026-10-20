# LOL-v1 训练集与 LOL-v2-real 测试集的重叠, 以及它对各方法 (尤其 ReDDiT) 结果的影响
#   1) 权重命名核对: Retinexformer LOL_v1.pth / LOL_v2_real.pth 在 LOL-v1 测试集与 LOL-v2-real 测试集上的 PSNR (论文: 25.16 / 22.80)
#   2) 重叠: LOL-v2-real 测试图按与 LOL-v1 (训练+测试) 的关系分三组:
#      输入相同 = 同编号低光图逐像素相同; 场景见过 = GT 缩略图相关 >0.95 但输入不同; 场景未见 = 其余
#   3) 各方法在三组上的 PSNR (原始 / GT 均值校正); LOL_v1 权重是已知泄漏的对照
#   4) 各方法在暗部单元 (<8DN, 暗示色度>=20) 上分辨真假颜色的能力, 按组分开 (统计口径同 dark_chroma_methods.py)
# 数据: data/LOLv1/{Train,Test}/{input,target}; LOL-v2-real 测试 GT 与低光图取自 ReDDiT 测试结果目录.
# 用法 (仓库根目录): python analysis/leakage_check.py   (CPU 约 10 分钟)
import os, io, subprocess, importlib.util, numpy as np, torch
from PIL import Image
from skimage.color import rgb2lab
from scipy.stats import spearmanr
from collections import Counter
torch.set_num_threads(8)
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'; V1='data/LOLv1'
IDS=[f'{690+k:05d}' for k in range(100)]
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
gt=lambda i: L(f'{RD}/gt/{int(i)-689}_gt.png'); low=lambda i: L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
def psnr(a,b): return 10*np.log10(1/np.mean((a-b)**2))
gray=lambda x: (x*[0.299,0.587,0.114]).sum(-1).mean()
spec=importlib.util.spec_from_file_location('rf','Retinexformer/basicsr/models/archs/RetinexFormer_arch.py'); rfm=importlib.util.module_from_spec(spec); spec.loader.exec_module(rfm)
def load_rf(ck):
    n=rfm.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
    n.load_state_dict({k.replace('module.',''):v for k,v in torch.load(ck,map_location='cpu',weights_only=False)['params'].items()}); return n.eval()
@torch.no_grad()
def run(net,img): return np.round(net(torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0)*255)/255

print('== 1) 权重命名核对 (PSNR, 输出量化到 8bit)')
rf1=load_rf('Retinexformer/pretrained_weights/LOL_v1.pth'); rf2=load_rf('Retinexformer/pretrained_weights/LOL_v2_real.pth')
v1t=sorted(os.listdir(f'{V1}/Test/input')); rf1_out={i:run(rf1,low(i)) for i in IDS}
for n,net,o2 in [('LOL_v1.pth',rf1,lambda i: rf1_out[i]),('LOL_v2_real.pth',rf2,lambda i: L(f'Retinexformer/results_color_exp/orig/test/{i}.png'))]:
    p1=np.mean([psnr(run(net,L(f'{V1}/Test/input/{f}')),L(f'{V1}/Test/target/{f}')) for f in v1t]); p2=np.mean([psnr(o2(i),gt(i)) for i in IDS])
    print(f'  {n:16s} LOL-v1 测试 {p1:.2f} dB | LOL-v2-real 测试 {p2:.2f} dB')
print('  论文: LOL-v1 25.16, LOL-v2-real 22.80  (results_color_exp/orig 即 LOL_v2_real.pth 的输出)')

print('\n== 2) LOL-v2-real 测试集与 LOL-v1 的重叠')
def thumb(x): v=np.asarray(Image.fromarray((x*255).astype(np.uint8)).convert('L').resize((48,32),Image.BILINEAR)).astype(float); v=(v-v.mean())/(v.std()+1e-6); return v.ravel()
T1=np.stack([thumb(L(f'{V1}/{s}/target/{f}')) for s in ['Train','Test'] for f in sorted(os.listdir(f'{V1}/{s}/target'))])
grp={}
for i in IDS:
    f=f'{V1}/Train/input/{int(i)}.png'
    if os.path.exists(f) and np.abs(L(f)-low(i)).mean()*255<0.01: grp[i]='输入相同'
    elif (T1@thumb(gt(i))).max()/T1.shape[1]>0.95: grp[i]='场景见过'
    else: grp[i]='场景未见'
G=['输入相同','场景见过','场景未见']; print('  '+', '.join(f'{g} {c} 张' for g,c in Counter(grp.values()).items()))
for gname in G[1:]: print(f'  {gname}: '+' '.join(i for i in IDS if grp[i]==gname))
print(f'  00739: {grp["00739"]} (LOL-v1 Train/739.png 的输入与 GT 均与之逐像素相同)')

def git_png(p): return np.asarray(Image.open(io.BytesIO(subprocess.run(['git','show',f'25a64fa:{p}'],capture_output=True,check=True).stdout)).convert('RGB')).astype(float)/255
OUT={'RF(LOL_v1,泄漏对照)':lambda i: rf1_out[i],'Retinexformer':lambda i: L(f'Retinexformer/results_color_exp/orig/test/{i}.png'),
     'HVI':lambda i: L(f'HVI-CIDNet/output/HVIFirst/{i}.png'),'CWNet':lambda i: L(f'CWNet-Causal-Wavelet-Network/output/low{i}.png'),
     'Diff-Retinex':lambda i: git_png(f'Diff-Retinex/experiments/Diff_Retinex_val_261008_104214/results/{i}.png'),'ReDDiT':lambda i: L(f'{RD}/output/{int(i)-689}_normal.png')}
cache={m:{i:f(i) for i in IDS} for m,f in OUT.items()}
print('\n== 3) 各组 PSNR: 原始 / GT 均值校正')
print(f'  {"方法":22s}'+''.join(f'{g:>20s}' for g in G))
for m in OUT:
    s=''
    for gname in G:
        ids=[i for i in IDS if grp[i]==gname]
        s+=f'{np.mean([psnr(cache[m][i],gt(i)) for i in ids]):12.2f} / {np.mean([psnr(np.clip(cache[m][i]*gray(gt(i))/gray(cache[m][i]),0,1),gt(i)) for i in ids]):5.2f}'
    print(f'  {m:22s}{s}')

print('\n== 4) 暗部单元分辨真假颜色的能力 (输出比例与 GT 比例的 Spearman / 平均 |输出-GT|)')
W=np.array([0.2126,0.7152,0.0722])
def blk(x,B=8): h,w=x.shape[0]//B*B,x.shape[1]//B*B; return x[:h,:w].reshape(h//B,B,w//B,B,-1).mean((1,3)).reshape(-1,x.shape[-1])
def lab(m): return rgb2lab(srgb(m)[None])[0]
units=[]
for i in IDS:
    lo,g=low(i),gt(i); ll,gl=lin(lo),lin(g); bi,bg=blk(ll),blk(gl); Yg=bg@W; ok=(Yg>1e-4)&(bg.max(1)<0.95)
    al=lambda m: m*(Yg/np.maximum(m@W,1e-9))[:,None]
    imp=lab(al(bi*((gl*ll).reshape(-1,3).sum(0)/np.maximum((ll*ll).reshape(-1,3).sum(0),1e-12))))[:,1:]
    ref={'gt':lab(bg)[:,1:],**{m:lab(al(blk(lin(cache[m][i]))))[:,1:] for m in OUT}}
    lvl=srgb(bi.max(1))*255; c=np.hypot(*imp.T); h=np.degrees(np.arctan2(imp[:,1],imp[:,0]))%360
    for hb in range(8):
        s=ok&(lvl<8)&(c>=20)&((h-hb*45)%360<45)
        if s.sum()>=15: units.append(dict(g=grp[i],**{k:(v[s]*imp[s]).sum()/(imp[s]**2).sum() for k,v in ref.items()}))
for gname in ['输入相同','场景未见']:
    U=[u for u in units if u['g']==gname]; gv=np.array([u['gt'] for u in U])
    print(f'  {gname} ({len(U)} 个单元): '+'  '.join(f"{m}={spearmanr(gv,[u[m] for u in U])[0]:+.2f}/{np.mean(np.abs(np.array([u[m] for u in U])-gv)):.2f}" for m in OUT))
