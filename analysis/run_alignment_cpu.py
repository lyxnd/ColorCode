# 在 CPU 上跑原文档的干预实验 (run_alignment_intervention.sh 的替代: 原脚本依赖的 infer_folder.py 不存在, 且需要 GPU)
#   1) 从 ReDDiT 测试结果目录取 LOL-v2-real 测试集的低光图 (去掉 padding) 与 GT, 存成 Test/Low、Test/Normal
#   2) make_alignment_inputs.py: 同场景两张输入里, 把较暗的一张分别按 只对齐曝光 / 逐通道增益 / 增益+偏置 / 3x3+偏置 对齐到较亮的一张
#   3) 用 Retinexformer 与 HVI-CIDNet 的 LOL-v2-real 官方权重推理全部输入
#   4) evaluate_alignment_outputs.py: 对齐后两张输出的色调差、亮度差还剩多少
# 用法 (仓库根目录): python analysis/run_alignment_cpu.py   结果: analysis/alignment_intervention/alignment_summary_<模型>.txt
import os, sys, subprocess, numpy as np, torch
from PIL import Image
sys.path.insert(0,'analysis/loss_data_ft'); from common import gt, low, IDS, build_rf, run, q8
torch.set_num_threads(int(os.environ.get('THREADS',4)))
R='analysis/alignment_intervention'; D=f'{R}/data'
for sub in ['Low','Normal']: os.makedirs(f'{D}/{sub}',exist_ok=True)
save=lambda x,p: Image.fromarray(np.round(np.clip(x,0,1)*255).astype(np.uint8)).save(p)
for i in IDS:
    if not os.path.exists(f'{D}/Normal/{i}.png'): save(low(i),f'{D}/Low/{i}.png'); save(gt(i),f'{D}/Normal/{i}.png')
if not os.path.exists(f'{R}/inputs/alignment_manifest.csv'):
    subprocess.run([sys.executable,'analysis/make_alignment_inputs.py','--test_low',f'{D}/Low','--test_gt',f'{D}/Normal','--save_dir',f'{R}/inputs'],check=True)
L=lambda p: np.asarray(Image.open(p).convert('RGB')).astype(np.float32)/255
rf=build_rf('Retinexformer/pretrained_weights/LOL_v2_real.pth').eval()
sys.path.insert(0,'HVI-CIDNet'); from net.CIDNet import CIDNet
hvi=CIDNet(); hvi.load_state_dict(torch.load('HVI-CIDNet/weights/best_PSNR.pth',map_location='cpu',weights_only=False)); hvi.eval(); hvi.trans.gated2=True; hvi.trans.alpha=0.8
@torch.no_grad()
def hvi_run(x): return q8(hvi(torch.from_numpy(x.transpose(2,0,1).copy()).float()[None]).clamp(0,1)[0].numpy().transpose(1,2,0))
files=sorted(f for f in os.listdir(f'{R}/inputs') if f.endswith('.png'))
for name,fn in [('retinexformer',lambda x: run(rf,x)),('hvi',hvi_run)]:
    out=f'{R}/outputs_{name}'; os.makedirs(out,exist_ok=True)
    for f in files:
        if not os.path.exists(f'{out}/{f}'): save(fn(L(f'{R}/inputs/{f}')),f'{out}/{f}')
    subprocess.run([sys.executable,'analysis/evaluate_alignment_outputs.py','--test_low',f'{D}/Low','--test_gt',f'{D}/Normal','--alignment_dir',f'{R}/inputs',
                    '--output_dir',out,'--name',name,'--save_dir',R],check=True)
    print(open(f'{R}/alignment_summary_{name}.txt').read(),flush=True)
