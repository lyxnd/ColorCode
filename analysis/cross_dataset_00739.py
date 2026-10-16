# 同一 Retinexformer 结构与训练配置 (n_feat=40, blocks [1,2,2], L1, mixup), 只换训练数据集, 推理 00739: 墙的颜色是否各不相同?
#   "数据 + 回归" 的预测: 墙变灰 / 保住 / 整体偏向某个颜色, 取决于该数据集里暗部输入颜色与 GT 颜色的关系;
#   若各数据集的模型都把墙压到差不多的灰度, 中性区也都不偏, 则说明原因在结构/回归本身而不是数据.
#   指标: 墙的色度保持率 (亮度归一化) 与色相差; GT 中性区 (暗/亮) 的 Δa* Δb* (整体偏色方向); 输出亮度.
#   不同数据集的输入亮度差别很大 (SID 约 48/255, LOL 约 10/255), 所以对整图输入做线性曝光 x1/x2/x4/x8 后各推理一次.
# 权重: Retinexformer/pretrained_weights/*.pth (官方发布: LOL_v1, LOL_v2_real, LOL_v2_synthetic, SID, SMID, SDSD_indoor, SDSD_outdoor, FiveK, NTIRE),
#       有几个跑几个. 对比图: analysis/cross_dataset_00739.png (每行一个权重, 列为 GT 与各曝光).
# 用法 (仓库根目录): python analysis/cross_dataset_00739.py
import glob, os, importlib.util, numpy as np, torch
from PIL import Image
from skimage.color import rgb2lab
from scipy.ndimage import binary_dilation
torch.set_num_threads(8)
RD='ReDDiT/experiments/lolv2_test_real/lolv2_test_real_numstep_4_w_snr_0.2_w_gt_0.2_w_lpips_0.1_260921_193705/results'
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(float)/255
def lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
W=np.array([0.2126,0.7152,0.0722])
g=L(f'{RD}/gt/50_gt.png'); lo=L(f'{RD}/input/50_lq.png')[8:408,4:604]; wall=np.load('analysis/wall_mask_00739.npy'); gw=lin(g)[wall].mean(0)
lg=rgb2lab(g); neu=(np.hypot(lg[...,1],lg[...,2])<6)&~binary_dilation(wall,iterations=15); nd=neu&(lg[...,0]<45); nb=neu&(lg[...,0]>=60)
labm=lambda o,mk: rgb2lab(srgb(lin(o)[mk].mean(0))[None,None])[0,0]
def wall_stats(o):
    m=lin(o)[wall].mean(0); m=m*(gw@W)/(m@W); a=rgb2lab(srgb(m)[None,None])[0,0]; b=rgb2lab(srgb(gw)[None,None])[0,0]
    dh=(np.degrees(np.arctan2(a[2],a[1]))-np.degrees(np.arctan2(b[2],b[1]))+180)%360-180
    return np.hypot(a[1],a[2])/np.hypot(b[1],b[2]), dh
spec=importlib.util.spec_from_file_location('rf','Retinexformer/basicsr/models/archs/RetinexFormer_arch.py'); rfm=importlib.util.module_from_spec(spec); spec.loader.exec_module(rfm)
@torch.no_grad()
def run(net,img): return net(torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0)
K=[1,2,4,8]; rows=[]
print(f'输入整图平均 {lo.mean()*255:.1f}/255; GT 黄墙色相 {np.degrees(np.arctan2(*labm(g,wall)[[2,1]])):.1f}°, GT 全图平均 L*={lg[...,0].mean():.1f}')
print('权重              曝光 | 墙保持率 墙色相差 | 暗中性区 Δa* Δb* | 亮中性区 Δa* Δb* | 输出平均 L*')
for ck in sorted(glob.glob('Retinexformer/pretrained_weights/*.pth')):
    name=os.path.basename(ck)[:-4]
    net=rfm.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(ck,map_location='cpu',weights_only=False)['params'].items()}); net.eval()
    outs=[]
    for k in K:
        o=run(net,srgb(lin(lo)*k)); outs.append(o); r,dh=wall_stats(o); a=labm(o,nd)-labm(g,nd); b=labm(o,nb)-labm(g,nb)
        print(f'{name:16s} x{k:<3} |  {r:5.2f}   {dh:+6.1f}° |   {a[1]:+5.1f} {a[2]:+5.1f}    |   {b[1]:+5.1f} {b[2]:+5.1f}    |  {rgb2lab(o)[...,0].mean():5.1f}')
    rows.append(np.concatenate([g]+outs,1))
M=np.concatenate(rows,0); Image.fromarray(np.round(M*255).astype(np.uint8)).resize((M.shape[1]//3,M.shape[0]//3)).save('analysis/cross_dataset_00739.png')
