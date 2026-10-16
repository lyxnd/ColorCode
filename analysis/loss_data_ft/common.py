# 损失/数据对照微调实验的公共函数: 读图、色彩空间、8x8 块统计、逐通道暗端偏置估计与校正、Retinexformer 构建
import os, importlib.util, numpy as np, torch
from PIL import Image
from skimage.color import rgb2lab
TR='data/LOLv2/Real_captured/Train'
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
IDS=[f'{690+k:05d}' for k in range(100)]   # LOL-v2-real 测试集; ReDDiT 结果目录里按 1..100 编号
UNSEEN=['00705','00706','00707','00753','00754','00755','00756','00767','00768']   # 与 LOL-v1 无重叠的场景 (leakage_check.py)
W=np.array([0.2126,0.7152,0.0722])
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
gt=lambda i: L(f'{RD}/gt/{int(i)-689}_gt.png'); low=lambda i: L(f'{RD}/input/{int(i)-689}_lq.png')[8:408,4:604]
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
q8=lambda x: np.round(np.clip(x,0,1)*255)/255
def blk(x,B=8): h,w=x.shape[0]//B*B,x.shape[1]//B*B; return x[:h,:w].reshape(h//B,B,w//B,B,-1).mean((1,3)).reshape(-1,x.shape[-1])
def lab(m): return rgb2lab(srgb(m)[None])[0]

def dark_offset(lo,g):
    """逐通道暗端底座 (线性光): 在输入 2-20DN 的块上拟合 gl≈a*ll+b, 底座 o=-b/a (输入比"与 GT 成正比"多出的量). 块不足 100 个时返回 0"""
    bi,bg=blk(lin(lo)),blk(lin(g)); v=srgb(bi.max(1))*255; s=(v>=2)&(v<20)&(bg.max(1)<0.95)
    if s.sum()<100: return np.zeros(3)
    return np.array([-np.polyfit(bi[s,c],bg[s,c],1)[1]/np.polyfit(bi[s,c],bg[s,c],1)[0] for c in range(3)])
def debias(lo,o): return q8(srgb(np.clip(lin(lo)-o,0,1)))   # 去掉底座后仍量化到 8bit, 与原始输入格式一致

def block_stats(lo,g,outs={}):
    """每个 8x8 块: 输入亮度 DN, 输入暗示 a*b* (整图逐通道增益校正, 对齐到 GT 块亮度), GT a*b*, 各输出 a*b* (同样对齐亮度)"""
    ll,gl=lin(lo),lin(g); bi,bg=blk(ll),blk(gl); Yg=bg@W; ok=(Yg>1e-4)&(bg.max(1)<0.95)
    imp=bi*((gl*ll).reshape(-1,3).sum(0)/np.maximum((ll*ll).reshape(-1,3).sum(0),1e-12))
    al=lambda m: m*(Yg/np.maximum(m@W,1e-9))[:,None]
    r=dict(lvl=srgb(bi.max(1))*255,imp=lab(al(imp))[:,1:],gt=lab(bg)[:,1:])
    for t,o in outs.items(): r[t]=lab(al(blk(lin(o))))[:,1:]
    return {k:v[ok] for k,v in r.items()}
def ratio(d,key,sel): return (d[key][sel]*d['imp'][sel]).sum()/max((d['imp'][sel]**2).sum(),1e-9)
def dark_sel(d,lo,hi): c=np.hypot(*d['imp'].T); return (d['lvl']>=lo)&(d['lvl']<hi)&(c>=20)&(c<60)

_spec=importlib.util.spec_from_file_location('rf','Retinexformer/basicsr/models/archs/RetinexFormer_arch.py')
rfm=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rfm)
def build_rf(ck='Retinexformer/pretrained_weights/LOL_v2_real.pth'):
    n=rfm.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
    sd=torch.load(ck,map_location='cpu',weights_only=False); sd=sd.get('params',sd.get('G',sd))
    n.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}); return n
@torch.no_grad()
def run(net,img): return q8(net(torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0))
