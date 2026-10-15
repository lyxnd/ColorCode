"""
[LLIE] 评测脚本：按文件名把增强结果和 GT 配对，计算 PSNR / SSIM / LPIPS，并给出颜色诊断指标。

用法示例：
  python eval_llie.py --gt_dir /mnt/d/LOLv2/Real_captured/Test/Normal \
      --pred_dirs output/AB/t100 output/AB/t500 output/AB/t900 \
      --show 00739

  --pred_dirs 可以给多个目录，每个目录输出一行汇总；逐张图的结果写到各自目录下的 metrics.csv。

指标说明：
  psnr / ssim         RGB、[0,1] 范围、整张图（不裁边）。SSIM 为 11x11 高斯窗(σ=1.5)、三通道平均。
  psnr_gtmean /       GT-mean 版本：先把预测图整体乘一个系数，使其平均亮度与 GT 相同，再计算。
  ssim_gtmean         用来排除"整体亮度差一点"对指标的影响，LLIE 论文常一起报告。
  lpips               默认 AlexNet 版本(LLIE 论文常用)；--lpips_net vgg 可切换。越低越好。
  bright_ratio        预测图平均亮度 / GT 平均亮度。<1 偏暗，>1 偏亮(PSNR 与 GT-mean PSNR 差距大时看它判断方向)。
  sat_ratio           预测图平均饱和度 / GT 平均饱和度(HSV 的 S)。<1 表示比 GT 发灰，>1 表示过饱和。
  chroma_err          在 YCbCr 的 Cb/Cr 两个色度通道上，与 GT 的平均绝对误差(0~255 刻度)，越低越好。
"""
import argparse
import csv
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(directory):
    files = {}
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        stem, extension = os.path.splitext(os.path.basename(path))
        if extension.lower() in IMAGE_EXTENSIONS:
            files[stem] = path
    return files


def load_image(path, device):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]


def compute_psnr(prediction, target):
    mse = F.mse_loss(prediction, target).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def compute_ssim(prediction, target, window):
    # 标准 SSIM(Wang et al. 2004)，数据范围 1，"valid" 卷积(不补边)，三通道分别计算后平均
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    channels = prediction.shape[1]
    mu_x = F.conv2d(prediction, window, groups=channels)
    mu_y = F.conv2d(target, window, groups=channels)
    sigma_xx = F.conv2d(prediction * prediction, window, groups=channels) - mu_x ** 2
    sigma_yy = F.conv2d(target * target, window, groups=channels) - mu_y ** 2
    sigma_xy = F.conv2d(prediction * target, window, groups=channels) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_xx + sigma_yy + c2))
    return ssim_map.mean().item()


def gt_mean_adjust(prediction, target):
    # 按灰度均值把预测图整体缩放到与 GT 同样的平均亮度
    prediction_gray = prediction.mean()
    target_gray = target.mean()
    return (prediction * (target_gray / prediction_gray.clamp_min(1e-6))).clamp(0, 1)


def mean_saturation(image):
    maximum = image.max(dim=1).values
    minimum = image.min(dim=1).values
    saturation = (maximum - minimum) / maximum.clamp_min(1e-6)
    return saturation.mean().item()


def chroma_channels(image):
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return cb, cr


def chroma_error(prediction, target):
    prediction_cb, prediction_cr = chroma_channels(prediction)
    target_cb, target_cr = chroma_channels(target)
    error = ((prediction_cb - target_cb).abs().mean() + (prediction_cr - target_cr).abs().mean()) / 2
    return error.item() * 255


def main():
    parser = argparse.ArgumentParser(description="LLIE evaluation: PSNR / SSIM / LPIPS + colour diagnostics")
    parser.add_argument("--gt_dir", default='/mnt/d/LOLv2/Real_captured/Test/Normal/', help="GT (normal-light) image folder")
    parser.add_argument("--pred_dirs", required=True, nargs="+", help="one or more result folders (file names must match GT)")
    parser.add_argument("--metrics", default="psnr,ssim,lpips", help="comma-separated subset of psnr,ssim,lpips")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    parser.add_argument("--show", nargs="*", default=[], help="file stems to print per-image results for, e.g. 00739")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    metrics = {name.strip() for name in args.metrics.split(",") if name.strip()}
    device = torch.device(args.device)
    window = gaussian_window(device=device)

    lpips_model = None
    if "lpips" in metrics:
        import lpips  # 训练环境里已安装
        lpips_model = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()

    gt_files = list_images(args.gt_dir)
    if not gt_files:
        raise SystemExit(f"No images found in --gt_dir {args.gt_dir}")

    columns = ["psnr", "psnr_gtmean", "ssim", "ssim_gtmean", "lpips", "bright_ratio", "sat_ratio", "chroma_err"]
    summary_rows = []
    for pred_dir in args.pred_dirs:
        pred_files = list_images(pred_dir)
        common = sorted(set(pred_files) & set(gt_files))
        missing = sorted(set(gt_files) - set(pred_files))
        if not common:
            print(f"[skip] {pred_dir}: no file names in common with GT")
            continue

        per_image = []
        for stem in common:
            prediction = load_image(pred_files[stem], device)
            target = load_image(gt_files[stem], device)
            if prediction.shape != target.shape:
                print(f"[skip] {pred_dir}/{stem}: size {tuple(prediction.shape[-2:])} != GT {tuple(target.shape[-2:])}")
                continue
            row = {"image": stem}
            with torch.no_grad():
                prediction_gtmean = gt_mean_adjust(prediction, target)
                if "psnr" in metrics:
                    row["psnr"] = compute_psnr(prediction, target)
                    row["psnr_gtmean"] = compute_psnr(prediction_gtmean, target)
                if "ssim" in metrics:
                    row["ssim"] = compute_ssim(prediction, target, window)
                    row["ssim_gtmean"] = compute_ssim(prediction_gtmean, target, window)
                if lpips_model is not None:
                    row["lpips"] = lpips_model(prediction * 2 - 1, target * 2 - 1).item()
                row["bright_ratio"] = prediction.mean().item() / max(target.mean().item(), 1e-6)
                row["sat_ratio"] = mean_saturation(prediction) / max(mean_saturation(target), 1e-6)
                row["chroma_err"] = chroma_error(prediction, target)
            per_image.append(row)

        used_columns = [c for c in columns if c in per_image[0]]
        with open(os.path.join(pred_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["image"] + used_columns)
            writer.writeheader()
            for row in per_image:
                writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

        summary = {"dir": pred_dir, "n": len(per_image)}
        for column in used_columns:
            summary[column] = float(np.mean([row[column] for row in per_image]))
        summary_rows.append((summary, used_columns))
        if missing:
            print(f"[note] {pred_dir}: {len(missing)} GT images have no result (e.g. {missing[:3]})")
        for stem in args.show:
            for row in per_image:
                if row["image"] == stem:
                    print(f"[{stem}] {pred_dir}: " + "  ".join(f"{c}={row[c]:.4f}" for c in used_columns))

    if not summary_rows:
        return
    used_columns = summary_rows[0][1]
    header = f"{'result dir':<40} {'n':>4} " + " ".join(f"{c:>11}" for c in used_columns)
    print("\n" + header + "\n" + "-" * len(header))
    for summary, _ in summary_rows:
        print(f"{summary['dir'][-40:]:<40} {summary['n']:>4} " + " ".join(f"{summary.get(c, float('nan')):>11.4f}" for c in used_columns))
    print("\nhigher is better: psnr, ssim | lower is better: lpips, chroma_err | bright_ratio / sat_ratio: 1.0 = same as GT (<1 darker / greyer)")


if __name__ == "__main__":
    main()
