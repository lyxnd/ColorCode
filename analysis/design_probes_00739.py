# 从网络设计找 00739 黄墙变灰的原因: 在各方法内部取中间结果
#   HVI-CIDNet   : HVI 变换按 sin(pi*I/2)^k 压暗部色度, 网络在 HVI 空间做残差 -> HV 残差置零时 ("什么都不做") 墙会是多少?
#   Retinexformer: 输出 = 光照提亮图 img*(1+illu_map) + denoiser 残差 -> 残差占多少亮度, 残差本身的色度
#   Diff-Retinex : 光照 L 单通道, 颜色全在 TDN 回归出的反射率 R 里 -> R 在墙上是否已经变灰 (RDA 权重超过 100MB 未上传, 只能看 TDN)
#   ReDDiT       : 4 步 DDIM, 不做残差 -> 每一步网络预测的 x0 (第 1 步从纯噪声出发, 最接近"条件均值"的估计)
#   CWNet 依赖 mamba 的 CUDA 算子, CPU 上无法推理, 只从代码看: 输出 = 输入 + 残差 (同为加法式提亮)
# 保持率 = 亮度归一化后的色度比 (同 intervention_00739.py)
# 用法 (仓库根目录): python analysis/design_probes_00739.py
import sys, json, importlib.util, numpy as np, torch, cv2
from PIL import Image
from skimage.color import rgb2lab
torch.set_num_threads(8)
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
W=np.array([0.2126,0.7152,0.0722])
i='00739'; g=L(f'{RD}/gt/{int(i)-689}_gt.png'); lo=L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]; wall=np.load('analysis/wall_mask_00739.npy'); gw=lin(g)[wall].mean(0)
def ret(o,target=gw):
    m=lin(np.clip(o,1e-4,1))[wall].mean(0); m=m*(target@W)/(m@W); a=rgb2lab(srgb(m)[None,None])[0,0]; b=rgb2lab(srgb(target)[None,None])[0,0]
    return np.hypot(a[1],a[2])/np.hypot(b[1],b[2])
T=lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()[None]
N=lambda t: t[0].detach().numpy().transpose(1,2,0)
rgbw=lambda o: '('+','.join(f'{v:.0f}' for v in o[wall].mean(0)*255)+')'
print(f'GT 黄墙 RGB={rgbw(g)}')

print('\n== HVI-CIDNet')
sys.path.insert(0,'HVI-CIDNet'); from net.CIDNet import CIDNet
hvi=CIDNet(); hvi.load_state_dict(torch.load('HVI-CIDNet/weights/best_PSNR.pth',map_location='cpu',weights_only=False)); hvi.eval(); hvi.trans.gated2=True; hvi.trans.alpha=0.8
with torch.no_grad():
    h_in=hvi.trans.HVIT(T(lo)); cap={}; orig=hvi.trans.PHVIT
    hvi.trans.PHVIT=lambda img: (cap.update(v=img.clone()), orig(img))[1]; o=N(hvi(T(lo)).clamp(0,1)); hvi.trans.PHVIT=orig
    ident=cap['v'].clone(); ident[:,:2]=h_in[:,:2]; o_id=N(hvi.trans.PHVIT(ident).clamp(0,1)); hv_gt=N(hvi.trans.HVIT(T(g)))
k=hvi.trans.density_k.item(); cs=lambda I: (np.sin(I*np.pi/2)+1e-8)**k; I_out=N(cap['v'])[...,2][wall].mean()
print(f'  学到的 k={k:.3f}; 恒等路径的饱和度比 cs(I_in)/cs(I_out) 随输入亮度 (输出亮度取墙的输出 {I_out*255:.0f}DN): '+'  '.join(f'{d}DN={cs(d/255)/cs(I_out):.2f}' for d in [2,4,8,16,32,64]))
print(f'  墙: 实际输出保持率={ret(o):.2f}, HV 残差置零后={ret(o_id):.2f};  |HV| 输入 {np.hypot(*N(h_in)[...,:2][wall].mean(0)):.3f} -> 网络输出 {np.hypot(*N(cap["v"])[...,:2][wall].mean(0)):.3f} (GT 为 {np.hypot(*hv_gt[...,:2][wall].mean(0)):.3f})')

print('\n== Retinexformer')
spec=importlib.util.spec_from_file_location('rf','Retinexformer/basicsr/models/archs/RetinexFormer_arch.py'); rfm=importlib.util.module_from_spec(spec); spec.loader.exec_module(rfm)
rf=rfm.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
rf.load_state_dict({k_.replace('module.',''):v for k_,v in torch.load('Retinexformer/pretrained_weights/LOL_v2.pth',map_location='cpu',weights_only=False)['params'].items()}); rf.eval()
f={}; hk=rf.body[0].estimator.register_forward_hook(lambda m,inp,out: f.update(map=out[1].detach()))
with torch.no_grad(): o=N(rf(T(lo)).clamp(0,1))
hk.remove(); lit=N(T(lo)*f['map']+T(lo)); res=o-lit
print(f'  墙: 提亮图={rgbw(np.clip(lit,0,1))}  denoiser 残差={rgbw(res)}  输出={rgbw(o)}')
print(f'  残差占输出亮度 {1-(lin(np.clip(lit,0,1))[wall].mean(0)@W)/(lin(o)[wall].mean(0)@W):.0%}; 提亮图保持率={ret(lit):.2f}, 残差本身保持率={ret(res):.2f}, 输出={ret(o):.2f}')

print('\n== Diff-Retinex (TDN)')
sys.path.insert(0,'Diff-Retinex'); from model.Diff_TDN.TDN_network import DecomNet
tdn=DecomNet(); tdn.load_state_dict(torch.load('Diff-Retinex/model/Diff_TDN/weights/checkpoint_LOL_Diff_TDN.pth',map_location='cpu',weights_only=False)['model']); tdn.eval()
with torch.no_grad(): R_lo=N(tdn(T(lo))[0]); R_hi=N(tdn(T(g))[0])
la=rgb2lab(srgb(lin(R_lo)[wall].mean(0))[None,None])[0,0]; lb=rgb2lab(srgb(lin(R_hi)[wall].mean(0))[None,None])[0,0]
print(f'  墙的反射率 R: 低光图分解 Lab=({la[0]:.0f},{la[1]:+.1f},{la[2]:+.1f}), GT 分解 Lab=({lb[0]:.0f},{lb[1]:+.1f},{lb[2]:+.1f}); 低光/GT 色度 (亮度归一化)={ret(R_lo,lin(R_hi)[wall].mean(0)):.2f}')
try:
    import subprocess; dr=np.asarray(Image.open(__import__('io').BytesIO(subprocess.run(['git','show',f'25a64fa:Diff-Retinex/experiments/Diff_Retinex_val_261008_104214/results/{i}.png'],capture_output=True,check=True).stdout)).convert('RGB')).astype(float)/255
    print(f'  Diff-Retinex 最终输出 (TDN -> RDA/IDA 扩散 -> R*L) 墙保持率={ret(dr):.2f}')
except Exception as e: print('  (取不到 Diff-Retinex 最终输出:',e,')')

print('\n== ReDDiT: 每一步网络预测的 x0')
for name in [n for n in sys.modules if n=='model' or n.startswith('model.')]: del sys.modules[name]   # Diff-Retinex 与 ReDDiT 都有名为 model 的包
sys.path.remove('Diff-Retinex'); sys.path.insert(0,'ReDDiT'); from model.ddpm_modules import diffusion, unet
m=json.load(open('ReDDiT/config/lolv2_real_test.json'))['model']; u=m['unet']; b=m['beta_schedule']['train']
rd=diffusion.GaussianDiffusion(unet.UNet(in_channel=u['in_channel'],out_channel=u['out_channel'],norm_groups=32,inner_channel=u['inner_channel'],
        channel_mults=u['channel_multiplier'],attn_res=u['attn_res'],res_blocks=u['res_blocks'],dropout=u['dropout'],image_size=m['diffusion']['image_size']),
    image_size=m['diffusion']['image_size'],num_timesteps=b['n_timestep'],time_scale=b['time_scale'],channels=m['diffusion']['channels'],
    w_gt=0,w_snr=0,w_str=0,w_lpips=0,loss_type='l1',conditional=True,schedule_opt=None)
rd.set_new_noise_schedule(b,torch.device('cpu')); rd.load_state_dict(torch.load('ReDDiT/checkpoint/lolv2_real_4step_gen.pth',map_location='cpu',weights_only=False),strict=False); rd.eval()
@torch.no_grad()
def steps(img,seed=0,zero=False):
    x=torch.from_numpy(cv2.copyMakeBorder(np.ascontiguousarray(img,dtype=np.float32),8,8,4,4,cv2.BORDER_REFLECT).transpose(2,0,1))[None]*2-1
    torch.manual_seed(seed); x_t=torch.zeros_like(x) if zero else torch.randn_like(x); out=[]
    for ts in reversed(range(1,rd.num_timesteps+1)):     # 与 diffusion.ddim 一致 (stride=1)
        t=x_t.new_ones([1],dtype=torch.long)*ts; s=t-1
        eps=rd.denoise_fn(torch.cat([x,x_t],1),torch.FloatTensor([rd.sqrt_alphas_cumprod_prev[t*rd.time_scale]]).repeat(1,1))[0]
        x0=rd.predict_start_from_noise(x_t,t*rd.time_scale,eps).clip(-1,1); eps=rd.predict_eps_from_x(x_t,x0,t*rd.time_scale)
        out.append(((x0[0].numpy().transpose(1,2,0)+1)/2)[8:408,4:604]); x_t=rd.sqrt_alphas_cumprod[s*rd.time_scale]*x0+rd.sqrt_one_minus_alphas_cumprod[s*rd.time_scale]*eps
    return out
for seed in [0,1]: print(f'  seed={seed}: '+'  '.join(f'第{k+1}步={ret(o):.2f}' for k,o in enumerate(steps(lo,seed))))
print('  初始噪声置零: '+'  '.join(f'第{k+1}步={ret(o):.2f}' for k,o in enumerate(steps(lo,zero=True))))
x=lo.copy(); x[wall]=srgb(lin(lo)[wall].mean(0)); print('  墙平坦无噪的输入: '+'  '.join(f'第{k+1}步={ret(o):.2f}' for k,o in enumerate(steps(x))))
