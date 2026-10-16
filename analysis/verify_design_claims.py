# 复核 "LOL-v2 暗部发灰原因总结.md" 里关于各方法设计的结论: 重新推理, 在网络内部取中间结果
#   1) Retinexformer: 输出 = 提亮图 img*(1+illu_map) + denoiser 的 mapping 残差 (RetinexFormer_arch.py: input_img=img*illu_map+img; out=mapping(fea)+x)
#      -> 提亮倍数 1+illu_map、残差占输出亮度的比例; 不只看 00739 的墙, 也看整图与全部 100 张测试图
#   2) HVI-CIDNet: 学到的 k; HV 残差置零 (只保留 HVI 变换的恒等路径) 时墙的保持率 vs 实际输出
#   3) Diff-Retinex TDN: 光照 L 的通道数; 墙上低光图分解的反射率 R 与 GT 分解的 R 的色度比
#   4) ReDDiT RATR: 代码中 y/(max(y)+0.1) - SNR_map 在墙上的值 (数据 min_max=(-1,1), 即代码口径) 与论文 [0,1] 口径
#   5) ReDDiT: 4 步 DDIM 每一步网络预测的 x0 在墙上的保持率
# 保持率 = 线性光下区域均值、亮度归一化后的 Lab 色度比 (同 cross_dataset_00739.py)
# 用法 (仓库根目录): python analysis/verify_design_claims.py   结果: analysis/verify_design_claims.txt
import sys, os, json, importlib.util, numpy as np, torch, cv2, torchvision.transforms as Tv
from PIL import Image
from skimage.color import rgb2lab
torch.set_num_threads(int(os.environ.get('THREADS',4)))
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
IDS=[f'{690+k:05d}' for k in range(100)]
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
gt=lambda i: L(f'{RD}/gt/{int(i)-689}_gt.png'); low=lambda i: L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
W=np.array([0.2126,0.7152,0.0722])
lo,g=low('00739'),gt('00739'); wall=np.load('analysis/wall_mask_00739.npy'); gw=lin(g)[wall].mean(0)
def ret(o,target=gw):
    m=lin(np.clip(o,1e-4,1))[wall].mean(0); m=m*(target@W)/(m@W); a=rgb2lab(srgb(m)[None,None])[0,0]; b=rgb2lab(srgb(target)[None,None])[0,0]
    return np.hypot(a[1],a[2])/np.hypot(b[1],b[2])
T=lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()[None]
N=lambda t: t[0].detach().numpy().transpose(1,2,0)
dn=lambda v: '('+','.join(f'{x*255:.1f}' for x in v)+')'
lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
Y=lambda x: lin(np.clip(x,0,1))@W   # 线性亮度

P('== 1) Retinexformer (LOL_v2_real.pth): 提亮图 vs denoiser 残差')
spec=importlib.util.spec_from_file_location('rf','Retinexformer/basicsr/models/archs/RetinexFormer_arch.py'); rfm=importlib.util.module_from_spec(spec); spec.loader.exec_module(rfm)
rf=rfm.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
rf.load_state_dict({k.replace('module.',''):v for k,v in torch.load('Retinexformer/pretrained_weights/LOL_v2_real.pth',map_location='cpu',weights_only=False)['params'].items()}); rf.eval()
cap={}; rf.body[0].estimator.register_forward_hook(lambda m,i,o: cap.update(map=o[1].detach()))
cap2={}; rf.body[0].denoiser.mapping.register_forward_hook(lambda m,i,o: cap2.update(res=o.detach()))
@torch.no_grad()
def rf_parts(x):
    o=N(rf(T(x))); gain=1+N(cap['map']); lit=x*gain; res=N(cap2['res'])
    assert np.abs(lit+res-o).max()<1e-4   # 输出 = 提亮图 + mapping 残差 (未截断), 与代码结构一致
    return gain,lit,res,o
gain,lit,res,o=rf_parts(lo)
P(f'  00739 墙: 输入 {dn(lo[wall].mean(0))} DN -> 提亮图 {dn(lit[wall].mean(0))} -> 输出 {dn(np.clip(o,0,1)[wall].mean(0))}; GT {dn(g[wall].mean(0))}')
P(f'  墙上提亮倍数 1+illu_map (R,G,B 均值) = ({",".join(f"{v:.2f}" for v in gain[wall].mean(0))}); 达到 GT 需要的 sRGB 倍数 ≈ {g[wall].mean()/lo[wall].mean():.0f}')
P(f'  墙: 提亮图占输出线性亮度 {Y(lit)[wall].mean()/Y(o)[wall].mean():.1%}, 残差占 {1-Y(lit)[wall].mean()/Y(o)[wall].mean():.1%}; 保持率 提亮图={ret(lit):.2f} 残差={ret(np.clip(res,1e-4,1)):.2f} 输出={ret(o):.2f}')
P(f'  00739 整图: 提亮倍数中位数 {np.median(gain):.2f}, 提亮图占输出线性亮度 {Y(lit).mean()/Y(o).mean():.1%}')
G_,S_=[],[]
for i in IDS:
    x=low(i); gn,lt,_,oo=rf_parts(x); G_.append(np.median(gn)); S_.append(Y(lt).mean()/Y(oo).mean())
P(f'  测试集 100 张: 每张图提亮倍数中位数的 [最小, 中位, 最大] = [{min(G_):.2f}, {np.median(G_):.2f}, {max(G_):.2f}]; '
  f'提亮图占输出线性亮度 [最小, 中位, 最大] = [{min(S_):.1%}, {np.median(S_):.1%}, {max(S_):.1%}]')

P('\n== 2) HVI-CIDNet (best_PSNR.pth, alpha=0.8): 恒等路径 vs 实际输出')
sys.path.insert(0,'HVI-CIDNet'); from net.CIDNet import CIDNet
hvi=CIDNet(); hvi.load_state_dict(torch.load('HVI-CIDNet/weights/best_PSNR.pth',map_location='cpu',weights_only=False)); hvi.eval(); hvi.trans.gated2=True; hvi.trans.alpha=0.8
with torch.no_grad():
    h_in=hvi.trans.HVIT(T(lo)); c={}; orig=hvi.trans.PHVIT
    hvi.trans.PHVIT=lambda img: (c.update(v=img.clone()), orig(img))[1]; oh=N(hvi(T(lo)).clamp(0,1)); hvi.trans.PHVIT=orig
    ident=c['v'].clone(); ident[:,:2]=h_in[:,:2]; oid=N(hvi.trans.PHVIT(ident).clamp(0,1))
P(f'  学到的 k = {hvi.trans.density_k.item():.3f}; 墙: 实际输出保持率 {ret(oh):.2f}, HV 置为输入值 (恒等路径) {ret(oid):.2f}')

P('\n== 3) Diff-Retinex TDN (checkpoint_LOL_Diff_TDN.pth)')
sys.path.insert(0,'Diff-Retinex'); from model.Diff_TDN.TDN_network import DecomNet
tdn=DecomNet(); tdn.load_state_dict(torch.load('Diff-Retinex/model/Diff_TDN/weights/checkpoint_LOL_Diff_TDN.pth',map_location='cpu',weights_only=False)['model']); tdn.eval()
with torch.no_grad(): Rl,Ll=tdn(T(lo)); Rh,_=tdn(T(g))
P(f'  光照 L 的形状 {tuple(Ll.shape)} (单通道); 墙: 低光图分解的 R 相对 GT 分解的 R 的色度比 = {ret(N(Rl),lin(N(Rh))[wall].mean(0)):.2f}')

P('\n== 4) ReDDiT RATR (diffusion.loss: T=max(y)+0.1, y=y/T, y=y-SNR_map(y))')
def snr_map(x): b=Tv.GaussianBlur(kernel_size=15,sigma=3)(x); gr=lambda t: t[:,0:1]*0.299+t[:,1:2]*0.587+t[:,2:3]*0.114; return (gr(b)-gr(x)).abs()
for tag,y0 in [('代码口径 [-1,1]',lo*2-1),('论文口径 [0,1]',lo)]:
    y=T(y0); y=y/(y.max(1,keepdim=True)[0]+0.1); y=N(y-snr_map(y))
    P(f'  {tag}: 墙上均值 ({",".join(f"{v:+.3f}" for v in y[wall].mean(0))})')

P('\n== 5) ReDDiT 学生模型 (lolv2_real_4step_gen.pth): 每步预测的 x0')
for n in [n for n in sys.modules if n=='model' or n.startswith('model.')]: del sys.modules[n]
sys.path.remove('Diff-Retinex'); sys.path.insert(0,'ReDDiT'); from model.ddpm_modules import diffusion, unet
m=json.load(open('ReDDiT/config/lolv2_real_test.json'))['model']; u=m['unet']; b=m['beta_schedule']['train']
rd=diffusion.GaussianDiffusion(unet.UNet(in_channel=u['in_channel'],out_channel=u['out_channel'],norm_groups=32,inner_channel=u['inner_channel'],
        channel_mults=u['channel_multiplier'],attn_res=u['attn_res'],res_blocks=u['res_blocks'],dropout=u['dropout'],image_size=m['diffusion']['image_size']),
    image_size=m['diffusion']['image_size'],num_timesteps=b['n_timestep'],time_scale=b['time_scale'],channels=m['diffusion']['channels'],
    w_gt=0,w_snr=0,w_str=0,w_lpips=0,loss_type='l1',conditional=True,schedule_opt=None)
rd.set_new_noise_schedule(b,torch.device('cpu')); miss=rd.load_state_dict(torch.load('ReDDiT/checkpoint/lolv2_real_4step_gen.pth',map_location='cpu',weights_only=False),strict=False); rd.eval()
P(f'  载入权重: 缺失 {len(miss.missing_keys)} 个键, 多余 {len(miss.unexpected_keys)} 个键')
with torch.no_grad():
    x=torch.from_numpy(cv2.copyMakeBorder(np.ascontiguousarray(lo,dtype=np.float32),8,8,4,4,cv2.BORDER_REFLECT).transpose(2,0,1))[None]*2-1
    torch.manual_seed(0); xt=torch.randn_like(x); out=[]
    for ts in reversed(range(1,rd.num_timesteps+1)):
        t=xt.new_ones([1],dtype=torch.long)*ts; s=t-1
        eps=rd.denoise_fn(torch.cat([x,xt],1),torch.FloatTensor([rd.sqrt_alphas_cumprod_prev[t*rd.time_scale]]).repeat(1,1))[0]
        x0=rd.predict_start_from_noise(xt,t*rd.time_scale,eps).clip(-1,1); eps=rd.predict_eps_from_x(xt,x0,t*rd.time_scale)
        out.append(((N(x0)+1)/2)[8:408,4:604]); xt=rd.sqrt_alphas_cumprod[s*rd.time_scale]*x0+rd.sqrt_one_minus_alphas_cumprod[s*rd.time_scale]*eps
P('  seed=0: '+'  '.join(f'第{k+1}步={ret(o_):.2f}' for k,o_ in enumerate(out))+f';  仓库里的输出 {ret(L(f"{RD}/output/50_normal.png")[8:408,4:604] if L(f"{RD}/output/50_normal.png").shape[0]==416 else L(f"{RD}/output/50_normal.png")):.2f}')
open('analysis/verify_design_claims.txt','w').write('\n'.join(lines)+'\n')
