# 00739 黄墙变灰的干预实验: 只改输入, 用官方权重 (CPU) 重新推理 Retinexformer 与 HVI-CIDNet, 看墙的色度怎么变.
#   保持率 = 亮度归一化后的色度比: 把输出墙的区域均值 (线性光) 缩放到 GT 墙的亮度后再比 C*, 排除输出亮度变化的影响.
#   I1 去掉墙上的噪声/量化        -> 若是噪声导致的收缩, 应明显回升
#   I2 墙的输入亮度 x k            -> (a) 真实带噪像素乘 k (信噪比不变) (b) 平坦无噪均值色乘 k
#   I3 换背景的色偏 / 亮度          -> 全局或上下文颜色恒常性的作用
#   I4 缩小墙的面积                -> "颜色占满画面"的作用
#   I5 色相扫描 (墙平坦, 同 L*C*)   -> 是否只针对黄色
#   I6 Retinexformer 内部          -> 光照提亮阶段 vs denoiser 之后
# 权重: Retinexformer/pretrained_weights/LOL_v2.pth (与 results_color_exp/orig 输出逐像素一致),
#       HVI-CIDNet/weights/best_PSNR.pth (官方 LOL-v2-Real 评测设置 alpha=0.8, 与 output/HVIFirst 一致).
# 用法 (仓库根目录): python analysis/intervention_00739.py
import sys, importlib.util, numpy as np, torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import gaussian_filter
torch.set_num_threads(8)
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
gt=lambda i: L(f'{RD}/gt/{int(i)-689}_gt.png')
low=lambda i: L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
W=np.array([0.2126,0.7152,0.0722])

spec=importlib.util.spec_from_file_location('rf','Retinexformer/basicsr/models/archs/RetinexFormer_arch.py')
rfm=importlib.util.module_from_spec(spec); spec.loader.exec_module(rfm)
rf=rfm.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
rf.load_state_dict({k.replace('module.',''):v for k,v in torch.load('Retinexformer/pretrained_weights/LOL_v2.pth',map_location='cpu',weights_only=False)['params'].items()}); rf.eval()
sys.path.insert(0,'HVI-CIDNet'); from net.CIDNet import CIDNet
hvi=CIDNet(); hvi.load_state_dict(torch.load('HVI-CIDNet/weights/best_PSNR.pth',map_location='cpu',weights_only=False)); hvi.eval()
hvi.trans.gated2=True; hvi.trans.alpha=0.8
@torch.no_grad()
def run(net,img): return net(torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0)
NETS={'Retinexformer':rf,'HVI':hvi}

i='00739'; g=gt(i); lo=low(i); wall=np.load('analysis/wall_mask_00739.npy')
gw=lin(g)[wall].mean(0); lw=lin(lo)[wall].mean(0)
def ret(o,target=gw):
    m=lin(o)[wall].mean(0); m=m*(target@W)/(m@W); a=rgb2lab(srgb(m)[None,None])[0,0]; b=rgb2lab(srgb(target)[None,None])[0,0]
    return np.hypot(a[1],a[2])/np.hypot(b[1],b[2])
def rep(tag,x,target=gw): print(f'  {tag:40s} '+'  '.join(f'{n}={ret(run(net,x),target):.2f}' for n,net in NETS.items()))
q8=lambda x: np.round(x*255)/255
print(f'GT 黄墙均值 RGB=({",".join(f"{v:.0f}" for v in srgb(gw)*255)}), 输入墙均值 DN=({",".join(f"{v:.2f}" for v in srgb(lw)*255)})')
rep('原始输入',lo)

print('\n== I1 去掉墙上的噪声/量化 (float 输入)')
x=lo.copy(); x[wall]=srgb(lw); rep('墙 = 自身均值色, 平坦无噪',x)
x=lo.copy(); x[wall]=gaussian_filter(lo,(3,3,0))[wall]; rep('墙 = 高斯模糊 σ=3',x)
rep('整图高斯模糊 σ=2',gaussian_filter(lo,(2,2,0)))

print('\n== I2 只改墙的输入亮度 (色度不变)')
for k in [0.5,1,2,4,8]:
    a=lo.copy(); a[wall]=np.clip(np.round(lo[wall]*255*k),0,255)/255   # sRGB 10DN 以下是线性段, 直接乘 DN
    b=lo.copy(); b[wall]=srgb(lw*k)
    print(f'  k={k:<3} (a)带噪同SNR: '+' '.join(f'{n}={ret(run(net,a)):.2f}' for n,net in NETS.items())
          +'   (b)平坦无噪: '+' '.join(f'{n}={ret(run(net,b)):.2f}' for n,net in NETS.items()))

print('\n== I3 换背景 (墙保留真实低光像素)')
Ybg=(lin(lo)[~wall]@W).mean()
for rg,bg in [(1,1),(1,0.5),(1,2),(1.3,0.6),(0.8,1.3)]:
    c=np.array([rg,1,bg]); c=c*Ybg/(c@W); x=np.empty_like(lo); x[:]=srgb(c); x[wall]=lo[wall]; rep(f'平坦背景 R/G={rg} B/G={bg} (同亮度)',q8(x))
for k in [0.5,2,4]:
    x=lo.copy(); x[~wall]=srgb(lin(lo)[~wall]*k); rep(f'真实背景亮度 x{k}',q8(x))
for j in ['00751','00700','00763']:
    b=low(j); x=srgb(lin(b)*lin(lo)[~wall].mean()/lin(b)[~wall].mean()); x[wall]=lo[wall]; rep(f'背景 = {j} 的低光图 (同亮度)',q8(x))

print('\n== I4 缩小墙的面积 (去掉的部分换成同亮度中性灰, 只统计剩下的墙)')
cols=np.where(wall.any(0))[0]
for frac in [0.5,0.8]:
    rm=wall.copy(); rm[:,:cols[int(len(cols)*(1-frac))]]=False; keep=wall&~rm
    x=lo.copy(); x[rm]=srgb(np.full(3,(lin(lo)[rm]@W).mean())); x=q8(x)
    s=[]
    for n,net in NETS.items():
        o=run(net,x); m=lin(o)[keep].mean(0); t=lin(g)[keep].mean(0); m=m*(t@W)/(m@W)
        a=rgb2lab(srgb(m)[None,None])[0,0]; bb=rgb2lab(srgb(t)[None,None])[0,0]; s.append(f'{n}={np.hypot(a[1],a[2])/np.hypot(bb[1],bb[2]):.2f}')
    print(f'  去掉 {frac:.0%}, 剩余墙占画面 {keep.mean():.0%}                 '+'  '.join(s))

print('\n== I5 色相扫描: 目标色保持 GT 墙的 L*、C*, 只转色相; 用墙专属逐通道增益反推输入 (平坦, float)')
lab=rgb2lab(srgb(gw)[None,None])[0,0]; L0,C0,h0=lab[0],np.hypot(lab[1],lab[2]),np.degrees(np.arctan2(lab[2],lab[1]))
for dh in range(0,360,45):
    hh=np.radians(h0+dh); t=lin(lab2rgb(np.array([[[L0,C0*np.cos(hh),C0*np.sin(hh)]]]))[0,0])
    x=lo.copy(); x[wall]=srgb(t*lw/gw); rep(f'色相 {(h0+dh)%360:5.1f}°, 输入 DN=({",".join(f"{v:.1f}" for v in srgb(t*lw/gw)*255)})',x,t)

print('\n== I6 Retinexformer 内部: 光照提亮后 (img*illu_map+img) vs 最终输出')
st=rf.body[0]; feats={}
hk=st.estimator.register_forward_hook(lambda m,inp,out: feats.update(map=out[1].detach()))
o=run(rf,lo); hk.remove()
x=torch.from_numpy(lo.transpose(2,0,1))[None].float(); lit=np.clip((x*feats['map']+x)[0].numpy().transpose(1,2,0),0,1)
lg=rgb2lab(g); neu=(np.hypot(lg[...,1],lg[...,2])<6)&~wall
for tag,im in [('提亮后',lit),('最终输出',o)]:
    l=rgb2lab(srgb(lin(im)[neu].mean(0))[None,None])[0,0]
    print(f'  {tag:6s} 墙保持率={ret(im):.2f}  墙均值 RGB=({",".join(f"{v:.0f}" for v in im[wall].mean(0)*255)})  GT中性区域 a*={l[1]:+.1f} b*={l[2]:+.1f}')
