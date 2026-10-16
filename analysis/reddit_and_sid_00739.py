# ReDDiT 为什么能保住 00739 黄墙? 以及换成 SID 训练的权重会怎样?
#   1) ReDDiT 在 CPU 上复现 (4 步, lolv2_real_4step_gen.pth; 与 test.py 相同的 reflect padding 和 [-1,1] 归一化)
#   2) ReDDiT 的输入干预 (与 intervention_00739.py 同一套): 去噪/换噪声/打乱, 墙亮度 x k, 换背景, 只改 B 通道
#   3) 测试集上各方法输出隐含的"整图逐通道 增益+偏置"映射: 偏置是否跟踪 GT 求出的每次采集偏置
#   4) 同一 Retinexformer 结构与训练配置 (L1, mixup, 256 patch), 换成 SID 训练的权重推理 00739
# 保持率 = 亮度归一化后的色度比 (同 intervention_00739.py). 对比图: analysis/sid_vs_lol_00739.png (GT | LOL 权重 | SID 权重 | ReDDiT)
# 用法 (仓库根目录): python analysis/reddit_and_sid_00739.py   (CPU 约 15 分钟; 需要 opencv-python, scikit-learn, lpips, torchvision)
import sys, os, json, importlib.util, numpy as np, torch, cv2
from PIL import Image
from skimage.color import rgb2lab
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
torch.set_num_threads(8)
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
IDS=[f'{690+k:05d}' for k in range(100)]
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
gt=lambda i: L(f'{RD}/gt/{int(i)-689}_gt.png'); low=lambda i: L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
W=np.array([0.2126,0.7152,0.0722]); q8=lambda x: np.clip(np.round(x*255),0,255)/255

sys.path.insert(0,'ReDDiT'); from model.ddpm_modules import diffusion, unet
m=json.load(open('ReDDiT/config/lolv2_real_test.json'))['model']; u=m['unet']; b=m['beta_schedule']['train']
rd=diffusion.GaussianDiffusion(unet.UNet(in_channel=u['in_channel'],out_channel=u['out_channel'],norm_groups=32,inner_channel=u['inner_channel'],
        channel_mults=u['channel_multiplier'],attn_res=u['attn_res'],res_blocks=u['res_blocks'],dropout=u['dropout'],image_size=m['diffusion']['image_size']),
    image_size=m['diffusion']['image_size'],num_timesteps=b['n_timestep'],time_scale=b['time_scale'],channels=m['diffusion']['channels'],
    w_gt=0,w_snr=0,w_str=0,w_lpips=0,loss_type='l1',conditional=True,schedule_opt=None)
rd.set_new_noise_schedule(b,torch.device('cpu'))   # 原代码把调度放在 CUDA 上
rd.load_state_dict(torch.load('ReDDiT/checkpoint/lolv2_real_4step_gen.pth',map_location='cpu',weights_only=False),strict=False); rd.eval()
@torch.no_grad()
def reddit(img,seed=0):
    x=torch.from_numpy(cv2.copyMakeBorder(np.ascontiguousarray(img,dtype=np.float32),8,8,4,4,cv2.BORDER_REFLECT).transpose(2,0,1))[None]*2-1
    torch.manual_seed(seed); return ((rd.super_resolution(x)[0].numpy().transpose(1,2,0)+1)/2).clip(0,1)[8:408,4:604]
spec=importlib.util.spec_from_file_location('rf','Retinexformer/basicsr/models/archs/RetinexFormer_arch.py'); rfm=importlib.util.module_from_spec(spec); spec.loader.exec_module(rfm)
def load_rf(ck):
    n=rfm.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
    n.load_state_dict({k.replace('module.',''):v for k,v in torch.load(ck,map_location='cpu',weights_only=False)['params'].items()}); return n.eval()
@torch.no_grad()
def run(net,img): return net(torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0)

i='00739'; g=gt(i); lo=low(i); wall=np.load('analysis/wall_mask_00739.npy'); gw=lin(g)[wall].mean(0); lw=lin(lo)[wall].mean(0)
def ret(o):
    mm=lin(o)[wall].mean(0); mm=mm*(gw@W)/(mm@W); a=rgb2lab(srgb(mm)[None,None])[0,0]; bb=rgb2lab(srgb(gw)[None,None])[0,0]
    return np.hypot(a[1],a[2])/np.hypot(bb[1],bb[2])
rgbw=lambda o: '('+','.join(f'{v:.0f}' for v in o[wall].mean(0)*255)+')'
def rep(tag,x): o=reddit(x); print(f'  {tag:44s} 墙保持率={ret(o):.2f}  墙均值RGB={rgbw(o)}',flush=True)

print('== 1) ReDDiT CPU 复现')
ref=L(f'{RD}/output/{int(i)-689}_normal.png')
for s in [0,1]:
    o=reddit(lo,s); from skimage.color import deltaE_ciede2000
    print(f'  seed={s}: 与已有输出 ΔE00={deltaE_ciede2000(rgb2lab(o),rgb2lab(ref)).mean():.2f}, PSNR={10*np.log10(1/np.mean((o-g)**2)):.2f}, 墙保持率={ret(o):.2f} (已有输出 {ret(ref):.2f})')

print('\n== 2) ReDDiT 输入干预 (GT 黄墙均值 RGB=('+','.join(f'{v:.0f}' for v in srgb(gw)*255)+'))')
rng=np.random.default_rng(0)
rep('原始输入',lo)
x=lo.copy(); x[wall]=srgb(lw); rep('墙 = 自身均值色, 平坦无噪',x)
x=lo.copy(); x[wall]=gaussian_filter(lo,(3,3,0))[wall]; rep('墙 = 高斯模糊 σ=3',x)
rep('整图高斯模糊 σ=2',gaussian_filter(lo,(2,2,0)))
sd=(lo[wall]*255).std(0); x=lo.copy(); x[wall]=q8(srgb(lw)+rng.normal(0,1,(wall.sum(),3))*sd/255); rep('墙 = 均值色 + 同强度独立高斯噪声',x)
x=lo.copy(); x[wall]=lo[wall][rng.permutation(wall.sum())]; rep('墙内像素随机打乱',x)
for k in [0.5,2,4]:
    x=lo.copy(); x[wall]=np.clip(np.round(lo[wall]*255*k),0,255)/255; rep(f'墙亮度 x{k} (带噪, 信噪比不变)',x)
Ybg=(lin(lo)[~wall]@W).mean()
for s in [0,2]:
    x=np.empty_like(lo); x[:]=(srgb(np.ones(3)*Ybg)*255+rng.normal(0,s,lo.shape))/255; x[wall]=lo[wall]; rep(f'背景 = 平坦中性灰 + 噪声 σ={s}DN',q8(x))
for j in ['00751','00700','00763']:
    bj=low(j); x=srgb(lin(bj)*lin(lo)[~wall].mean()/lin(bj)[~wall].mean()); x[wall]=lo[wall]; rep(f'背景 = {j} 的低光图',q8(x))
mb=lo[wall][:,2].mean()
x=lo.copy(); w=x[wall]; w[:,2]=mb; x[wall]=w; rep(f'B = 均值 {mb*255:.2f}DN, 无噪声',x)
x=lo.copy(); w=x[wall]; w[:,2]=0; x[wall]=w; rep('B = 0',x)
for p in [0.3,0.5]:
    x=lo.copy(); w=x[wall]; w[:,2]=(rng.random(len(w))<p)/255; x[wall]=w; rep(f'B = 0/1 随机, 均值 {p:.2f}DN',x)
x=lo.copy(); w=x[wall]; w[:,2]=0; x[wall]=w
print(f'  (对照) Retinexformer LOL 权重在 B=0 时墙均值RGB={rgbw(run(load_rf("Retinexformer/pretrained_weights/LOL_v2.pth"),x))}')

print('\n== 3) 测试集: 各方法输出隐含的整图逐通道偏置 与 GT 求出的偏置 (通道差) 的相关')
OUT={'Retinexformer':'Retinexformer/results_color_exp/orig/test/{}.png','HVI':'HVI-CIDNet/output/HVIFirst/{}.png','CWNet':'CWNet-Causal-Wavelet-Network/output/low{}.png',
     'Diff-Retinex':'Diff-Retinex/experiments/Diff_Retinex_val_261008_104214/results/{}.png','ReDDiT':RD+'/output/{}_normal.png'}
path=lambda mm,ii: OUT[mm].format(int(ii)-689 if mm=='ReDDiT' else ii)
for mm in [mm for mm in OUT if not os.path.exists(path(mm,'00690'))]: print(f'  缺少 {mm} 的输出, 跳过'); OUT.pop(mm)
def blk(x,B=8): h,w=x.shape[0]//B*B,x.shape[1]//B*B; return x[:h,:w].reshape(h//B,B,w//B,B,-1).mean((1,3)).reshape(-1,3)
def offsets(xi,yo,sel): return np.array([-np.polyfit(xi[sel,c],yo[sel,c],1)[1]/np.polyfit(xi[sel,c],yo[sel,c],1)[0]*12.92*255 for c in range(3)])
og,om=[],{mm:[] for mm in OUT}
for ii in IDS:
    bi,bg=blk(lin(low(ii))),blk(lin(gt(ii))); ok=(bg.max(1)<0.95)&(srgb(bi.max(1))*255>=1)
    og.append(offsets(bi,bg,ok))
    for mm in OUT: om[mm].append(offsets(bi,blk(lin(L(path(mm,ii)))),ok))
og=np.array(og)
for mm in OUT:
    o=np.array(om[mm]); print(f'  {mm:14s} R-G r={spearmanr(og[:,0]-og[:,1],o[:,0]-o[:,1])[0]:+.2f}   B-G r={spearmanr(og[:,2]-og[:,1],o[:,2]-o[:,1])[0]:+.2f}')

print('\n== 4) Retinexformer: LOL 权重 vs SID 权重 (同结构, 同为 L1 + mixup + 256 patch 训练)')
lg=rgb2lab(g); neu=(np.hypot(lg[...,1],lg[...,2])<6)&~wall; nd=neu&(lg[...,0]<45); nb=neu&(lg[...,0]>=60)
labm=lambda o,mk: rgb2lab(srgb(lin(o)[mk].mean(0))[None,None])[0,0]
nets={'LOL权重':load_rf('Retinexformer/pretrained_weights/LOL_v2.pth'),'SID权重':load_rf('Retinexformer/pretrained_weights/SID.pth')}
lw_=labm(g,wall); print(f'  GT       黄墙 a*={lw_[1]:+5.1f} b*={lw_[2]:+5.1f} 色相={np.degrees(np.arctan2(lw_[2],lw_[1])):5.1f}°')
outs={}
for n,net in nets.items():
    o=outs[n]=run(net,lo); w=labm(o,wall); a=labm(o,nd); bb=labm(o,nb)
    print(f'  {n:8s} 黄墙保持率={ret(o):.2f} 色相={np.degrees(np.arctan2(w[2],w[1])):5.1f}° | GT中性区 暗(L<45) a*={a[1]:+5.1f} b*={a[2]:+5.1f}  亮(L>=60) a*={bb[1]:+5.1f} b*={bb[2]:+5.1f} | 全图均值RGB=('+','.join(f'{v:.0f}' for v in o.reshape(-1,3).mean(0)*255)+')')
print('  墙亮度 x k (带噪, 信噪比不变):')
for k in [0.5,1,2,4]:
    x=lo.copy(); x[wall]=np.clip(np.round(lo[wall]*255*k),0,255)/255
    print(f'    k={k:<3} '+'  '.join(f'{n}={ret(run(net,x)):.2f}' for n,net in nets.items()))
M=np.concatenate([np.asarray(Image.fromarray(np.round(a*255).astype(np.uint8)).resize((300,200))) for a in [g,outs['LOL权重'],outs['SID权重'],ref]],1)
Image.fromarray(M).save('analysis/sid_vs_lol_00739.png')
