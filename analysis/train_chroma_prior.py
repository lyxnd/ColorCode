# 训练集分析: 模型为什么会把暗部饱和色压灰?
#   intervention_00739.py 显示: Retinexformer/HVI 按"输入有多暗"决定压多少色度, 与实际噪声无关.
#   这里检查这种收缩是不是训练数据的条件均值 (回归到均值), 以及训练数据里暗部颜色为什么不可信.
#   1) 训练集里有没有 00739 同一场景 (灰度缩略图相关)
#   2) 训练数据的"应收缩量": 8x8 块, 输入用整图逐通道增益 (由 GT 拟合) 校正并对齐到 GT 块亮度, 得到"输入暗示的颜色";
#      按输入亮度 (块均值最亮通道 DN) 与输入暗示色度分档, 统计 GT 色度 / 输入暗示色度 (沿暗示色相方向投影)
#      = 只看局部、但知道正确白平衡的 L2 回归器应学到的收缩量
#   3) 块大小 4~32 是否改变比例 (噪声导致的衰减会随块变大而消失)
#   4) 改用 增益+逐通道偏置 校正后比例是否回到 1; 并用暗端 (2-20DN) 块估计每张图的逐通道底座 (黑电平/暗端响应差)
#   5) 官方权重在训练图 (每 3 张取 1 张) 上的同一统计, 与数据的条件均值对比
# 用法 (仓库根目录): python analysis/train_chroma_prior.py   第 5 步在 CPU 上约 20 分钟, 推理结果缓存到 analysis/train_model_cache/
import os, sys, importlib.util, numpy as np, torch
from PIL import Image
from skimage.color import rgb2lab
TR='data/LOLv2/Real_captured/Train'; RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
CACHE='analysis/train_model_cache'
names=sorted(os.listdir(f'{TR}/Low'))
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
W=np.array([0.2126,0.7152,0.0722])
LV=[(0,3),(3,5),(5,8),(8,12),(12,20),(20,40),(40,256)]
def blk(x,B=8): h,w=x.shape[0]//B*B,x.shape[1]//B*B; return x[:h,:w].reshape(h//B,B,w//B,B,-1).mean((1,3)).reshape(-1,x.shape[-1])
def lab(m): return rgb2lab(srgb(m)[None])[0]

def block_stats(lo,g,outs={},B=8,offset=False):
    """返回每个块: 输入亮度 DN, 输入暗示 a*b*, GT a*b*, 以及各输出 (对齐到 GT 块亮度) 的 a*b*"""
    ll,gl=lin(lo),lin(g); bi,bg=blk(ll,B),blk(gl,B); Yg=bg@W; ok=(Yg>1e-4)&(bg.max(1)<0.95)
    if offset:  # 每通道 gl ≈ a*ll + b
        X=np.c_[bi,np.ones(len(bi))]; co=np.stack([np.linalg.lstsq(X[ok][:,[c,3]],bg[ok][:,c],rcond=None)[0] for c in range(3)])
        imp=np.clip(bi*co[:,0]+co[:,1],1e-6,None)
    else:
        imp=bi*((gl*ll).reshape(-1,3).sum(0)/np.maximum((ll*ll).reshape(-1,3).sum(0),1e-12))
    al=lambda m: m*(Yg/np.maximum(m@W,1e-9))[:,None]
    r=dict(lvl=srgb(bi.max(1))*255,imp=lab(al(imp))[:,1:],gt=lab(bg)[:,1:])
    for t,o in outs.items(): r[t]=lab(al(blk(lin(o),B)))[:,1:]
    return {k:v[ok] for k,v in r.items()}
class Acc:
    """按 (输入亮度档, 暗示色度档) 累加: 某目标沿暗示色相方向的分量 / 暗示色度"""
    def __init__(s,keys,cb): s.cb=cb; s.d={(k,lv):[0.,0.,0] for k in keys for lv in LV}
    def add(s,st):
        c=np.hypot(*st['imp'].T); sel0=(c>=s.cb[0])&(c<s.cb[1])
        for lv in LV:
            sel=sel0&(st['lvl']>=lv[0])&(st['lvl']<lv[1])
            for k in [k for k,l in s.d if l==lv]:
                s.d[(k,lv)][0]+=(st[k][sel]*st['imp'][sel]).sum(); s.d[(k,lv)][1]+=(c[sel]**2).sum(); s.d[(k,lv)][2]+=int(sel.sum())
    def get(s,k,lv): a=s.d[(k,lv)]; return (a[0]/a[1] if a[2]>=30 else float('nan')), a[2]

print('== 1) 训练集里与 00739 GT 最相似的场景 (48x32 灰度缩略图归一化相关; 同场景 00739/00740 为 1.00)')
def thumb(p): x=np.asarray(Image.open(p).convert('L').resize((48,32),Image.BILINEAR)).astype(float); x=(x-x.mean())/(x.std()+1e-6); return x.ravel()
T=np.stack([thumb(f'{TR}/Normal/{n}') for n in names]); v=thumb(f'{RD}/gt/50_gt.png'); c=T@v/len(v); o=np.argsort(-c)[:5]
print('   '+'  '.join(f'{names[k]}({c[k]:.2f})' for k in o))

print('\n== 2) 训练数据的应收缩量: GT 色度 / 输入暗示色度 (只用逐通道增益校正), 689 张')
accs={cb:Acc(['gt'],cb) for cb in [(20,35),(35,60)]}; acc_off=Acc(['gt'],(35,60)); offs=[]
for n in names:
    lo,g=L(f'{TR}/Low/{n}'),L(f'{TR}/Normal/{n}'); st=block_stats(lo,g)
    for a in accs.values(): a.add(st)
    acc_off.add(block_stats(lo,g,offset=True))
    bi,bg=blk(lin(lo)),blk(lin(g)); s=(srgb(bi.max(1))*255>=2)&(srgb(bi.max(1))*255<20)&(bg.max(1)<0.95)
    if s.sum()>=100:   # 只用暗端 (2-20DN) 的块逐通道拟合 gl≈a*ll+b, 底座 = -b/a 折算成输入 DN (sRGB 10DN 以下为线性段)
        offs.append([-np.polyfit(bi[s,c],bg[s,c],1)[1]/np.polyfit(bi[s,c],bg[s,c],1)[0]*12.92*255 for c in range(3)])
for cb,a in accs.items():
    print(f'   暗示色度 {cb[0]}-{cb[1]}: '+'  '.join(f'{lv[0]}-{lv[1]}DN={a.get("gt",lv)[0]:.2f}' for lv in LV))

print('\n== 3) 块大小对 3-5DN / 暗示色度 35-60 一档的影响 (每 2 张取 1 张)')
for B in [4,8,16,32]:
    a=Acc(['gt'],(35,60))
    for n in names[::2]: a.add(block_stats(L(f'{TR}/Low/{n}'),L(f'{TR}/Normal/{n}'),B=B))
    r,k=a.get('gt',(3,5)); print(f'   块 {B:2d}x{B:<2d}: {r:.2f} (块数 {k})')

print('\n== 4) 增益+逐通道偏置校正后 (暗示色度 35-60)')
for lv in LV:
    r0,k0=accs[(35,60)].get('gt',lv); r1,k1=acc_off.get('gt',lv)
    print(f'   {lv[0]:3d}-{lv[1]:<3d}DN  只用增益: {r0:.2f} (块 {k0:6d})   增益+偏置: {r1:.2f} (块 {k1:6d})')
o=np.array(offs); d=o-o[:,[1]]
print(f'   暗端逐通道底座 (DN, 正=输入多一个正偏置; {len(o)} 张图): 四分位 R {np.round(np.percentile(o[:,0],[25,75]),1)} G {np.round(np.percentile(o[:,1],[25,75]),1)} B {np.round(np.percentile(o[:,2],[25,75]),1)}')
print(f'   通道间底座差: R-G 四分位 {np.round(np.percentile(d[:,0],[25,75]),1)}, B-G 四分位 {np.round(np.percentile(d[:,2],[25,75]),1)}; |差|>1DN 的图占比 R-G {(np.abs(d[:,0])>1).mean():.0%}, B-G {(np.abs(d[:,2])>1).mean():.0%}')

print('\n== 5) 官方权重在训练图上 (每 3 张取 1 张) 的收缩 vs 数据条件均值 (暗示色度 35-60 / 20-35)')
spec=importlib.util.spec_from_file_location('rf','Retinexformer/basicsr/models/archs/RetinexFormer_arch.py'); rfm=importlib.util.module_from_spec(spec); spec.loader.exec_module(rfm)
rf=rfm.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
rf.load_state_dict({k.replace('module.',''):v for k,v in torch.load('Retinexformer/pretrained_weights/LOL_v2.pth',map_location='cpu',weights_only=False)['params'].items()}); rf.eval()
sys.path.insert(0,'HVI-CIDNet'); from net.CIDNet import CIDNet
hvi=CIDNet(); hvi.load_state_dict(torch.load('HVI-CIDNet/weights/best_PSNR.pth',map_location='cpu',weights_only=False)); hvi.eval(); hvi.trans.gated2=True; hvi.trans.alpha=0.8
@torch.no_grad()
def run(net,img): return net(torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0)
def cached(tag,net,n,lo):
    p=f'{CACHE}/{tag}/{n}'
    if not os.path.exists(p): os.makedirs(os.path.dirname(p),exist_ok=True); Image.fromarray(np.round(run(net,lo)*255).astype(np.uint8)).save(p)
    return L(p)
sub={cb:Acc(['gt','RF','HVI'],cb) for cb in [(35,60),(20,35)]}
for n in names[::3]:
    lo,g=L(f'{TR}/Low/{n}'),L(f'{TR}/Normal/{n}')
    st=block_stats(lo,g,outs={'RF':cached('Retinexformer',rf,n,lo),'HVI':cached('HVI',hvi,n,lo)})
    for a in sub.values(): a.add(st)
for cb,a in sub.items():
    print(f'   暗示色度 {cb[0]}-{cb[1]}:')
    for lv in LV:
        g_,k=a.get('gt',lv); print(f'     {lv[0]:3d}-{lv[1]:<3d}DN (块 {k:6d})  数据(GT)={g_:.2f}  Retinexformer={a.get("RF",lv)[0]:.2f}  HVI={a.get("HVI",lv)[0]:.2f}')
