# 用 LOL-v2-real 权重测试时有没有泄漏?
#   1) 数据: LOL-v2-real 训练集与测试集是否重叠 (GT 文件 md5、低光图逐像素相同、GT 缩略图相关 >0.95 视为同场景);
#      LOL-v1 训练集与 LOL-v2-real 测试集的重叠 (只对用过 LOL-v1 训练的模型构成泄漏)
#   2) 行为: 见过的图, 模型能复现那张 GT 特有的整体亮度 (输入里推不出来, 所以论文才用 GT 均值校正). 比较三组图上的整体亮度误差 |log2(输出亮度/GT 亮度)|:
#        S = LOL-v2-real 训练图 (所有 LOL-v2 权重都见过) 抽 20 张
#        O = LOL-v2-real 测试图中, 低光图与 LOL-v1 训练图逐像素相同的 86 张
#        U = 与 LOL-v1、LOL-v2 训练集都不同场景的 9 张
#      只用 LOL-v2-real 训练的模型: O 与 U 都没见过, 应接近 (差别只来自场景难度); 见过 LOL-v1 的模型: O 应接近 S.
#        V = LOL-v1 测试集 15 张: 对 LOL_v1 权重是"同一套采集惯例、但没见过" -> 区分"记住了这张图"与"熟悉 LOL-v1 的亮度惯例"
#            (其中 12 张在 LOL-v2-real 训练集里, 所以对 LOL-v2 权重大多是见过的; ReDDiT 不在 V 上推理)
#      对照: Retinexformer LOL_v1.pth (已知用 LOL-v1 训练) 与 LOL_v2_real.pth (同结构, 只用 LOL-v2-real).
# 用法 (仓库根目录): python analysis/leakage_v2_check.py   (CPU 约 30 分钟, ReDDiT 在训练图上的推理缓存到 analysis/leakage_v2_cache/)
import sys, os, json, hashlib, numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0,'analysis/loss_data_ft'); from common import L, lin, srgb, q8, gt, low, IDS, UNSEEN, RD, TR, build_rf, run, W
torch.set_num_threads(int(os.environ.get('THREADS',4)))
lines=[]; P=lambda s='': (print(s,flush=True),lines.append(s))
V1='data/LOLv1/Train'; md5=lambda p: hashlib.md5(open(p,'rb').read()).hexdigest()
names=sorted(os.listdir(f'{TR}/Low'))

P('== 1) 数据重叠')
tr_gt={md5(f'{TR}/Normal/{n}') for n in names}; te_gt={i:md5(f'{RD}/gt/{int(i)-689}_gt.png') for i in IDS}
pix=lambda x: hashlib.md5(np.round(x*255).astype(np.uint8).tobytes()).hexdigest()
tr_lo={pix(L(f'{TR}/Low/{n}')) for n in names}
def thumb(x): t=np.asarray(Image.fromarray(np.round(x*255).astype(np.uint8)).convert('L').resize((48,32),Image.BILINEAR)).astype(float); t=(t-t.mean())/(t.std()+1e-6); return t.ravel()
Ttr=np.stack([thumb(L(f'{TR}/Normal/{n}')) for n in names]); best={i:(Ttr@thumb(gt(i))/Ttr.shape[1]).max() for i in IDS}
P(f'  LOL-v2-real 训练集 {len(names)} 张 vs 测试集 100 张: GT 文件相同 {sum(te_gt[i] in tr_gt for i in IDS)} 张, 低光图逐像素相同 {sum(pix(low(i)) in tr_lo for i in IDS)} 张, '
  f'GT 缩略图相关 >0.95 {sum(v>0.95 for v in best.values())} 张 (每张测试图与训练集的最高相关: 中位数 {np.median(list(best.values())):.2f}, 最大 {max(best.values()):.2f})')
O=[i for i in IDS if os.path.exists(f'{V1}/input/{int(i)}.png') and np.abs(L(f'{V1}/input/{int(i)}.png')-low(i)).mean()*255<0.01]
U=sorted(UNSEEN); S=names[::35][:20]
P(f'  LOL-v1 训练集 vs LOL-v2-real 测试集: 低光图逐像素相同 {len(O)} 张 (O 组); 两个训练集都没有的场景 {len(U)} 张 (U 组)')

P('\n== 2) 整体亮度误差 |log2(输出平均亮度 / GT 平均亮度)| (档), 中位数; 括号内为原始 PSNR 均值')
Y=lambda x: (lin(np.clip(x,0,1))@W).mean()
err=lambda o,g: abs(np.log2(Y(o)/Y(g)))
psnr=lambda o,g: 10*np.log10(1/np.mean((o-g)**2))
src={'S':[(L(f'{TR}/Low/{n}'),L(f'{TR}/Normal/{n}'),n) for n in S],'O':[(low(i),gt(i),i) for i in O],'U':[(low(i),gt(i),i) for i in U],
     'V':[(L(f'data/LOLv1/Test/input/{f}'),L(f'data/LOLv1/Test/target/{f}'),'v1_'+f[:-4]) for f in sorted(os.listdir('data/LOLv1/Test/input'))]}
def report(name,fn,skip=()):
    r={}
    for k,lst in src.items():
        if k in skip: continue
        e,p=[],[]
        for lo,g,i in lst:
            o=fn['S'](lo,i) if k in 'SV' else fn['T'](lo,i)
            e.append(err(o,g)); p.append(psnr(o,g))
        r[k]=(np.median(e),np.mean(p))
    P(f'  {name:26s} '+'   '.join(f'{k}: {r[k][0]:.2f} 档 ({r[k][1]:.2f} dB)' if k in r else f'{k}:      —          ' for k in 'SOUV'))
for nm,ck in [('Retinexformer LOL_v1 (已知泄漏)','LOL_v1'),('Retinexformer LOL_v2_real','LOL_v2_real')]:
    net=build_rf(f'Retinexformer/pretrained_weights/{ck}.pth').eval(); f=lambda lo,i: run(net,lo); report(nm,{'S':f,'T':f})
sys.path.insert(0,'HVI-CIDNet'); from net.CIDNet import CIDNet
hvi=CIDNet(); hvi.load_state_dict(torch.load('HVI-CIDNet/weights/best_PSNR.pth',map_location='cpu',weights_only=False)); hvi.eval(); hvi.trans.gated2=True; hvi.trans.alpha=0.8
@torch.no_grad()
def hvi_run(lo,i=None): return q8(hvi(torch.from_numpy(lo.transpose(2,0,1).copy()).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0))
report('HVI-CIDNet LOL-v2-real',{'S':hvi_run,'T':lambda lo,i: L(f'HVI-CIDNet/output/HVIFirst/{i}.png')})
report('CWNet LOL-v2-real',{'S':None,'T':lambda lo,i: L(f'CWNet-Causal-Wavelet-Network/output/low{i}.png')},skip='SV')   # CWNet 依赖 CUDA, 只有测试集的现成输出
sys.path.remove('HVI-CIDNet')
for n in [n for n in sys.modules if n=='model' or n.startswith('model.')]: del sys.modules[n]
sys.path.insert(0,'ReDDiT'); from model.ddpm_modules import diffusion, unet
m=json.load(open('ReDDiT/config/lolv2_real_test.json'))['model']; u=m['unet']; b=m['beta_schedule']['train']
rd=diffusion.GaussianDiffusion(unet.UNet(in_channel=u['in_channel'],out_channel=u['out_channel'],norm_groups=32,inner_channel=u['inner_channel'],
        channel_mults=u['channel_multiplier'],attn_res=u['attn_res'],res_blocks=u['res_blocks'],dropout=u['dropout'],image_size=m['diffusion']['image_size']),
    image_size=m['diffusion']['image_size'],num_timesteps=b['n_timestep'],time_scale=b['time_scale'],channels=m['diffusion']['channels'],
    w_gt=0,w_snr=0,w_str=0,w_lpips=0,loss_type='l1',conditional=True,schedule_opt=None)
rd.set_new_noise_schedule(b,torch.device('cpu')); rd.load_state_dict(torch.load('ReDDiT/checkpoint/lolv2_real_4step_gen.pth',map_location='cpu',weights_only=False),strict=False); rd.eval()
CA='analysis/leakage_v2_cache'; os.makedirs(CA,exist_ok=True)
@torch.no_grad()
def rd_run(lo,n):   # 与 verify_design_claims.py 相同的 4 步 DDIM (seed 0), 输入 reflect padding 到 416x608
    cf=f'{CA}/{n}.png'
    if os.path.exists(cf): return L(cf)
    x=torch.from_numpy(cv2.copyMakeBorder(np.ascontiguousarray(lo,dtype=np.float32),8,8,4,4,cv2.BORDER_REFLECT).transpose(2,0,1))[None]*2-1
    torch.manual_seed(0); xt=torch.randn_like(x)
    for ts in reversed(range(1,rd.num_timesteps+1)):
        t=xt.new_ones([1],dtype=torch.long)*ts; s=t-1
        eps=rd.denoise_fn(torch.cat([x,xt],1),torch.FloatTensor([rd.sqrt_alphas_cumprod_prev[t*rd.time_scale]]).repeat(1,1))[0]
        x0=rd.predict_start_from_noise(xt,t*rd.time_scale,eps).clip(-1,1); eps=rd.predict_eps_from_x(xt,x0,t*rd.time_scale)
        xt=rd.sqrt_alphas_cumprod[s*rd.time_scale]*x0+rd.sqrt_one_minus_alphas_cumprod[s*rd.time_scale]*eps
    o=q8(((x0[0].numpy().transpose(1,2,0)+1)/2)[8:408,4:604]); Image.fromarray(np.round(o*255).astype(np.uint8)).save(cf); return o
report('ReDDiT lolv2_real_4step',{'S':rd_run,'T':lambda lo,i: L(f'{RD}/output/{int(i)-689}_normal.png')},skip='V')
open('analysis/leakage_v2_check.txt','w').write('\n'.join(lines)+'\n')

# 3) 拆开 "拍摄惯例" 与 "背过这张图": LOL-v2-real 训练集里有 399 张来自 LOL-v1 训练集、12 张来自 LOL-v1 测试集, 所以只用 LOL-v2 训练的模型也学过 LOL-v1 的惯例
import hashlib as _h
H=lambda p: _h.md5(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()
v1tr={H(f'data/LOLv1/Train/input/{f}') for f in os.listdir('data/LOLv1/Train/input')}; v1te={H(f'data/LOLv1/Test/input/{f}') for f in os.listdir('data/LOLv1/Test/input')}
tr_h={n:H(f'{TR}/Low/{n}') for n in names}
P(f'\n== 3) LOL-v2-real 训练集 {len(names)} 张: 与 LOL-v1 训练集相同 {sum(v in v1tr for v in tr_h.values())} 张, 与 LOL-v1 测试集相同 {sum(v in v1te for v in tr_h.values())} 张')
P('   S 组按来源拆开 (都是 LOL-v2 权重见过的训练图), 整体亮度误差中位数:')
rf1=build_rf('Retinexformer/pretrained_weights/LOL_v1.pth').eval(); rf2=build_rf('Retinexformer/pretrained_weights/LOL_v2_real.pth').eval()
for nm,f in [('Retinexformer LOL_v1',lambda lo,n: run(rf1,lo)),('Retinexformer LOL_v2_real',lambda lo,n: run(rf2,lo)),('ReDDiT',rd_run)]:
    e={'v1':[],'new':[]}
    for lo,g,n in src['S']: e['v1' if tr_h[n] in v1tr|v1te else 'new'].append(err(f(lo,n),g))
    P(f'     {nm:26s} LOL-v1 来源 {len(e["v1"])} 张: {np.median(e["v1"]):.2f} 档   新拍的 {len(e["new"])} 张: {np.median(e["new"]):.2f} 档')
v2te={pix(low(i)):i for i in IDS}; P3=sorted(v2te[H(f'data/LOLv1/Test/input/{f}')] for f in os.listdir('data/LOLv1/Test/input') if H(f'data/LOLv1/Test/input/{f}') not in set(tr_h.values()))
P(f'\n== 4) LOL-v1 惯例、但两个训练集都没有的测试图 {P3}: 无论 ReDDiT 的教师是否用过 LOL-v1 训练集都没见过.')
P('   若某模型背过 O 组, 这几张的误差应明显高于它在 O 组上的典型值. 整体亮度误差 (档), 括号内为在该模型 O 组分布中的百分位:')
for nm,f in [('Retinexformer LOL_v1 (已知泄漏)',lambda i: run(rf1,low(i))),('Retinexformer LOL_v2_real',lambda i: run(rf2,low(i))),('HVI-CIDNet',lambda i: L(f'HVI-CIDNet/output/HVIFirst/{i}.png')),
             ('CWNet',lambda i: L(f'CWNet-Causal-Wavelet-Network/output/low{i}.png')),('ReDDiT',lambda i: L(f'{RD}/output/{int(i)-689}_normal.png'))]:
    eo=np.array([err(f(i),gt(i)) for i in O]); ep=[err(f(i),gt(i)) for i in P3]
    P(f'     {nm:26s} O 组 [25%, 中位, 75%] = [{np.percentile(eo,25):.2f}, {np.median(eo):.2f}, {np.percentile(eo,75):.2f}];  这几张: '+'  '.join(f'{v:.2f} ({(eo<v).mean()*100:.0f}%)' for v in ep))
open('analysis/leakage_v2_check.txt','w').write('\n'.join(lines)+'\n')
