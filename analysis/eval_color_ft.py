# 色度微调实验评估
# 用法: python analysis/eval_color_ft.py --results Retinexformer/results_color_ft --gt_dir <LOL-v2-Real Test/Normal>
# 指标 (LOL-v2-Real 测试集全部图片):
#   PSNR / SSIM / ΔE00 ; 暗+饱和像素(GT C*>30 且 L*<45)色度保持率 ; 全部像素色度保持率
# 合成验证图 (assets/verify2, 5 张): 无噪声+无色偏 / 有噪声+无色偏 的暗+饱和色度保持率 (检验"模型本身压低饱和度")
import argparse, os
from glob import glob
import numpy as np
from PIL import Image
from natsort import natsorted
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--results', required=True)
parser.add_argument('--gt_dir', required=True)
parser.add_argument('--verify_gt', default='assets/gt')
args = parser.parse_args()
load = lambda p: np.asarray(Image.open(p).convert('RGB')).astype(np.float64) / 255
chroma = lambda lab: np.hypot(lab[..., 1], lab[..., 2])
gt_paths = natsorted(glob(os.path.join(args.gt_dir, '*.png')) + glob(os.path.join(args.gt_dir, '*.jpg')))
gt_cache = {}

rows = []
for name in ['orig', 'A', 'B', 'C', 'D']:
    test_dir = os.path.join(args.results, name, 'test')
    if not os.path.isdir(test_dir):
        continue
    out_paths = natsorted(glob(os.path.join(test_dir, '*.png')))
    assert len(out_paths) == len(gt_paths), f'{name}: {len(out_paths)} 张输出, {len(gt_paths)} 张 GT'
    psnr, ssim, delta_e = [], [], []
    num_dark = den_dark = num_all = den_all = 0.0
    for out_path, gt_path in zip(out_paths, gt_paths):
        output = load(out_path)
        if gt_path not in gt_cache:
            gt_image = load(gt_path); gt_cache[gt_path] = (gt_image, rgb2lab(gt_image))
        gt_image, gt_lab = gt_cache[gt_path]
        out_lab = rgb2lab(output)
        psnr.append(10 * np.log10(1 / np.mean((output - gt_image) ** 2)))
        ssim.append(structural_similarity(output, gt_image, channel_axis=2, data_range=1.0))
        delta_e.append(deltaE_ciede2000(gt_lab, out_lab).mean())
        chroma_gt, chroma_out = chroma(gt_lab), chroma(out_lab)
        mask = (chroma_gt > 30) & (gt_lab[..., 0] < 45)
        num_dark += chroma_out[mask].sum(); den_dark += chroma_gt[mask].sum()
        num_all += chroma_out.sum(); den_all += chroma_gt.sum()
    verify = {}
    for kind in ['cleanwb', 'noisywb']:
        num = den = 0.0
        for image_id in ['00739', '00749', '00750', '00751', '00752']:
            path = os.path.join(args.results, name, 'verify2', f'{image_id}_{kind}.png')
            if not os.path.exists(path):
                num = None; break
            gt_lab = rgb2lab(load(os.path.join(args.verify_gt, f'{image_id}.png')))
            out_lab = rgb2lab(load(path))
            mask = (chroma(gt_lab) > 30) & (gt_lab[..., 0] < 45)
            num += chroma(out_lab)[mask].sum(); den += chroma(gt_lab)[mask].sum()
        verify[kind] = None if num is None else num / den
    rows.append((name, np.mean(psnr), np.mean(ssim), np.mean(delta_e), num_dark / den_dark, num_all / den_all,
                 verify['cleanwb'], verify['noisywb']))

fmt = lambda v: '  -  ' if v is None else f'{v:.3f}'
print(f'{"组":6s} {"PSNR":>7s} {"SSIM":>6s} {"ΔE00":>6s} {"暗饱和保持":>8s} {"全图保持":>7s} {"合成:无噪无偏":>10s} {"合成:有噪无偏":>10s}')
for name, p, s, e, cd, ca, vc, vn in rows:
    print(f'{name:6s} {p:7.2f} {s:6.3f} {e:6.2f} {cd:8.3f} {ca:7.3f} {fmt(vc):>10s} {fmt(vn):>10s}')
