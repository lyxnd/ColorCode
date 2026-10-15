"""
[LLIE] 最优 t 分析：判断"自动选 t"是否有收益，以及该做全局自动 t 还是空间自适应融合。

用同一批低光输入、多个 t 的推理结果，对照 GT 回答三个问题：
  1. 全局单一 t：所有图用同一个 t 时，哪个 t 最好；
  2. 逐图最优 t：每张图各自选最好的 t（用 GT 选，属于上限），比单一 t 好多少；最优 t 是否随输入的亮度/噪声变化；
  3. 逐块最优 t：把图切成小块，每块各自选最好的 t 再拼起来（同样是用 GT 的上限），比逐图最优又好多少；
     块的最优 t 是否随该块输入的局部信噪比变化。
  2 的收益明显 → 值得做全局自动 t；3 的收益明显大于 2 → 值得做空间自适应融合；都不明显 → 固定一个 t 即可。

用法：
  python analyze_oracle_t.py \
      --gt_dir /mnt/d/LOLv2/Real_captured/Test/Normal \
      --lq_dir /mnt/d/LOLv2/Real_captured/Test/Low \
      --pred 100=output/fifth/t100/skip1 250=output/fifth/t250/skip1 500=output/fifth/t500/skip1 \
             700=output/fifth/t700/skip1 900=output/fifth/t900/skip1 \
      --out_dir output/fifth/oracle_t

  --pred 的格式是 "t=结果目录"，可给任意多个 t；结果目录里的文件名需与 GT、低光图一致。

输出(都写到 --out_dir)：
  summary.txt          终端打印的汇总，同时存一份
  per_image.csv        每张图：各 t 的 PSNR / GT-mean PSNR / LPIPS，最优 t，输入亮度/噪声/信噪比
  per_patch_bins.csv   按输入局部信噪比分档：每档内块的最优 t 分布

说明：
  - "GT-mean PSNR"：先把预测整体缩放到 GT 的平均亮度再算（与 eval_llie.py 一致），排除训练/测试集曝光差异的影响。
  - 逐块比较用 GT-mean 缩放后的逐像素误差；装了 lpips 时还会用 LPIPS 的空间图(spatial=True)做一遍。
  - 噪声用 Immerkær 快速噪声估计(拉普拉斯型 3x3 卷积的平均绝对响应)。在纹理丰富的地方它也会把结构算进去，
    所以局部"噪声"更准确地说是"高频能量"；低光图里噪声通常占主导。局部信噪比 = 局部平均亮度 / 局部高频能量。
  - 逐图/逐块最优都是借助 GT 得到的上限，用来判断方案值不值得做，不是可部署的方法。
"""
import argparse
import csv
import glob
import math
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IMMERKAER_KERNEL = torch.tensor([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


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


def psnr_from_mse(mse):
    return float("inf") if mse <= 0 else 10 * math.log10(1.0 / mse)


def gt_mean_adjust(prediction, target):
    return (prediction * (target.mean() / prediction.mean().clamp_min(1e-6))).clamp(0, 1)


def pad_to_multiple(tensor, multiple):
    height, width = tensor.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor


def patch_mean(map_2d, patch):
    # map_2d: [1,1,H,W] -> [Ph,Pw]，边缘不足一块的部分用复制填充后再平均
    return F.avg_pool2d(pad_to_multiple(map_2d, patch), patch, stride=patch)[0, 0]


def immerkaer_abs_response(gray_255):
    # gray_255: [1,1,H,W]，0~255。返回 |拉普拉斯型响应| 图([1,1,H-2,W-2])，sigma = sqrt(pi/2)/6 * 其平均值
    kernel = IMMERKAER_KERNEL.to(gray_255.device)[None, None]
    return F.conv2d(gray_255, kernel).abs()


def rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    # 相同值取平均名次
    values = np.asarray(values)
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(description="Oracle timestep analysis for adaptive t selection")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--lq_dir", required=True, help="low-light input folder (for brightness / noise statistics)")
    parser.add_argument("--pred", required=True, nargs="+", help='list of "t=result_dir"')
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--patch", type=int, default=32, help="patch size (pixels) for the per-patch analysis")
    parser.add_argument("--no_lpips", action="store_true", help="skip LPIPS (faster)")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    parser.add_argument("--snr_bins", type=int, default=5, help="number of local-SNR quantile bins in the per-patch table")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    timestep_dirs = {}
    for item in args.pred:
        if "=" not in item:
            raise SystemExit(f'--pred entries must look like "t=dir", got: {item}')
        timestep, directory = item.split("=", 1)
        timestep_dirs[int(timestep)] = directory
    timesteps = sorted(timestep_dirs)
    if len(timesteps) < 2:
        raise SystemExit("need at least two timesteps")

    gt_files, lq_files = list_images(args.gt_dir), list_images(args.lq_dir)
    pred_files = {t: list_images(d) for t, d in timestep_dirs.items()}
    stems = sorted(set(gt_files) & set(lq_files) & set.intersection(*[set(p) for p in pred_files.values()]))
    if not stems:
        raise SystemExit("no file names shared by GT, low-light inputs and all result folders")

    # [LLIE] 第 10 轮：防止 --gt_dir / --lq_dir 写反(写反时 PSNR≈8dB、LPIPS≈0.55，结论全部无效)
    check_stems = stems[:min(10, len(stems))]
    gt_mean_check = sum(load_image(gt_files[s], device).mean().item() for s in check_stems) / len(check_stems)
    lq_mean_check = sum(load_image(lq_files[s], device).mean().item() for s in check_stems) / len(check_stems)
    if lq_mean_check >= gt_mean_check:
        raise SystemExit(f"--lq_dir looks brighter than --gt_dir (mean {lq_mean_check:.3f} vs {gt_mean_check:.3f} "
                         f"over {len(check_stems)} images). The two folders are probably swapped: "
                         f"--gt_dir should be the normal-light folder and --lq_dir the low-light folder.")

    lpips_model = None
    if not args.no_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net=args.lpips_net, spatial=True, verbose=False).to(device).eval()

    metric_names = ["psnr_gtmean"] + (["lpips"] if lpips_model is not None else [])
    higher_is_better = {"psnr_gtmean": True, "lpips": False}

    per_image_rows = []
    # 逐块：每个指标收集 (块的局部信噪比, 块的最优 t)
    patch_records = {name: [] for name in metric_names}
    # 汇总：单一 t / 逐图最优 / 逐块融合 的指标
    single_t_scores = {name: {t: [] for t in timesteps} for name in metric_names}
    single_t_psnr_raw = {t: [] for t in timesteps}
    per_image_oracle_scores = {name: [] for name in metric_names}
    fusion_scores = {name: [] for name in metric_names}            # 按该指标做逐块选择后，融合图在该指标上的得分
    fusion_cross_scores = {name: {} for name in metric_names}      # 按该指标做逐块选择后，融合图在另一指标上的得分

    for stem in stems:
        target = load_image(gt_files[stem], device)
        low = load_image(lq_files[stem], device)
        predictions = {}
        for t in timesteps:
            prediction = load_image(pred_files[t][stem], device)
            if prediction.shape != target.shape:
                raise SystemExit(f"size mismatch for {stem} at t={t}: {tuple(prediction.shape[-2:])} vs GT {tuple(target.shape[-2:])}")
            predictions[t] = prediction
        if low.shape != target.shape:
            low = F.interpolate(low, size=target.shape[-2:], mode="bilinear", align_corners=False)

        # 输入统计：平均亮度、Immerkær 噪声、信噪比；以及逐块的局部信噪比
        gray = low.mean(dim=1, keepdim=True) * 255
        response = immerkaer_abs_response(gray)
        sigma = math.sqrt(math.pi / 2) / 6 * response.mean().item()
        brightness = gray.mean().item()
        snr = brightness / max(sigma, 1e-6)
        response_full = F.pad(response, (1, 1, 1, 1), mode="replicate")
        patch_brightness = patch_mean(gray, args.patch)
        patch_sigma = math.sqrt(math.pi / 2) / 6 * patch_mean(response_full, args.patch)
        patch_snr = (patch_brightness / patch_sigma.clamp_min(1e-3)).flatten().cpu().numpy()

        row = {"image": stem, "lq_brightness": brightness, "lq_noise_sigma": sigma, "lq_snr": snr}
        error_maps = {name: {} for name in metric_names}
        with torch.no_grad():
            for t, prediction in predictions.items():
                adjusted = gt_mean_adjust(prediction, target)
                squared_error = ((adjusted - target) ** 2).mean(dim=1, keepdim=True)
                row[f"psnr_t{t}"] = psnr_from_mse(F.mse_loss(prediction, target).item())
                row[f"psnr_gtmean_t{t}"] = psnr_from_mse(squared_error.mean().item())
                single_t_psnr_raw[t].append(row[f"psnr_t{t}"])
                single_t_scores["psnr_gtmean"][t].append(row[f"psnr_gtmean_t{t}"])
                error_maps["psnr_gtmean"][t] = squared_error
                if lpips_model is not None:
                    lpips_map = lpips_model(prediction * 2 - 1, target * 2 - 1)
                    if lpips_map.shape[-2:] != target.shape[-2:]:
                        lpips_map = F.interpolate(lpips_map, size=target.shape[-2:], mode="bilinear", align_corners=False)
                    row[f"lpips_t{t}"] = lpips_map.mean().item()
                    single_t_scores["lpips"][t].append(row[f"lpips_t{t}"])
                    error_maps["lpips"][t] = lpips_map

            for name in metric_names:
                values = [row[f"{name}_t{t}"] for t in timesteps]
                best_index = int(np.argmax(values) if higher_is_better[name] else np.argmin(values))
                row[f"best_t_{name}"] = timesteps[best_index]
                per_image_oracle_scores[name].append(values[best_index])

                # 逐块：每块选误差最小的 t，拼出融合图
                patch_errors = torch.stack([patch_mean(error_maps[name][t], args.patch) for t in timesteps])  # [T,Ph,Pw]
                best_patch_index = patch_errors.argmin(dim=0)
                patch_records[name].append(np.stack([patch_snr, np.array(timesteps)[best_patch_index.flatten().cpu().numpy()]], axis=1))
                height, width = target.shape[-2:]
                index_map = best_patch_index.repeat_interleave(args.patch, 0).repeat_interleave(args.patch, 1)[:height, :width]
                stacked = torch.stack([predictions[t][0] for t in timesteps])  # [T,3,H,W]
                fused = torch.gather(stacked, 0, index_map[None, None].expand(1, 3, height, width).to(device))[0][None]
                fused_adjusted = gt_mean_adjust(fused, target)
                fused_psnr = psnr_from_mse(F.mse_loss(fused_adjusted, target).item())
                fused_lpips = lpips_model(fused * 2 - 1, target * 2 - 1).mean().item() if lpips_model is not None else None
                fusion_scores[name].append(fused_psnr if name == "psnr_gtmean" else fused_lpips)
                for other in metric_names:
                    if other != name:
                        fusion_cross_scores[name].setdefault(other, []).append(fused_lpips if other == "lpips" else fused_psnr)
                row[f"fused_{name}"] = fusion_scores[name][-1]
        per_image_rows.append(row)

    # ---------------- 写出逐图结果 ----------------
    columns = ["image", "lq_brightness", "lq_noise_sigma", "lq_snr"]
    for name in ["psnr"] + metric_names:
        columns += [f"{name}_t{t}" for t in timesteps]
    columns += [f"best_t_{name}" for name in metric_names] + [f"fused_{name}" for name in metric_names]
    with open(os.path.join(args.out_dir, "per_image.csv"), "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k]) for k in columns})

    # ---------------- 汇总 ----------------
    lines = []
    def emit(text=""):
        print(text)
        lines.append(text)

    emit(f"images: {len(per_image_rows)} | timesteps: {timesteps} | patch: {args.patch}px | metrics: {metric_names}")
    emit("")
    emit("1) single t for all images (mean over images)")
    header = f"   {'t':>6} {'PSNR':>9} {'GT-mean PSNR':>13}" + (f" {'LPIPS':>8}" if lpips_model is not None else "")
    emit(header)
    for t in timesteps:
        line = f"   {t:>6} {np.mean(single_t_psnr_raw[t]):>9.3f} {np.mean(single_t_scores['psnr_gtmean'][t]):>13.3f}"
        if lpips_model is not None:
            line += f" {np.mean(single_t_scores['lpips'][t]):>8.4f}"
        emit(line)
    best_single = {}
    for name in metric_names:
        means = {t: np.mean(single_t_scores[name][t]) for t in timesteps}
        best_single[name] = max(means, key=means.get) if higher_is_better[name] else min(means, key=means.get)
        emit(f"   best single t by {name}: t={best_single[name]} ({means[best_single[name]]:.4f})")

    emit("")
    emit("2) per-image oracle t (chosen with GT, upper bound of global adaptive t)")
    for name in metric_names:
        oracle = np.mean(per_image_oracle_scores[name]); single = np.mean(single_t_scores[name][best_single[name]])
        gain = oracle - single if higher_is_better[name] else single - oracle
        counts = Counter(row[f"best_t_{name}"] for row in per_image_rows)
        distribution = ", ".join(f"t{t}:{counts.get(t, 0)}" for t in timesteps)
        unit = "dB" if name == "psnr_gtmean" else ""
        emit(f"   {name}: oracle {oracle:.4f} vs best single {single:.4f} -> gain {gain:+.4f}{unit} | best-t counts: {distribution}")
        best_ts = [row[f"best_t_{name}"] for row in per_image_rows]
        emit(f"      Spearman(best t, input noise sigma) = {spearman(best_ts, [r['lq_noise_sigma'] for r in per_image_rows]):+.3f}   "
             f"(best t, input brightness) = {spearman(best_ts, [r['lq_brightness'] for r in per_image_rows]):+.3f}   "
             f"(best t, input SNR) = {spearman(best_ts, [r['lq_snr'] for r in per_image_rows]):+.3f}")

    emit("")
    emit(f"3) per-patch oracle fusion ({args.patch}px patches, chosen with GT, upper bound of spatially adaptive fusion)")
    bin_rows = []
    for name in metric_names:
        fused = np.mean(fusion_scores[name]); oracle = np.mean(per_image_oracle_scores[name])
        single = np.mean(single_t_scores[name][best_single[name]])
        gain_single = fused - single if higher_is_better[name] else single - fused
        gain_oracle = fused - oracle if higher_is_better[name] else oracle - fused
        unit = "dB" if name == "psnr_gtmean" else ""
        emit(f"   selected by {name}: fused {fused:.4f} | gain over best single t {gain_single:+.4f}{unit}, over per-image oracle {gain_oracle:+.4f}{unit}")
        for other, values in fusion_cross_scores[name].items():
            emit(f"      (the same fused images measured by {other}: {np.mean(values):.4f}; best single t by {other}: "
                 f"{np.mean(single_t_scores[other][best_single[other]]):.4f})")
        records = np.concatenate(patch_records[name], axis=0)
        local_snr, best_patch_t = records[:, 0], records[:, 1]
        emit(f"      Spearman(patch best t, local SNR) = {spearman(best_patch_t, local_snr):+.3f}  (negative = noisier patches prefer larger t)")
        edges = np.quantile(local_snr, np.linspace(0, 1, args.snr_bins + 1))
        emit(f"      {'local SNR bin':>22} {'patches':>8} {'mean best t':>12}  " + " ".join(f"{'t' + str(t):>7}" for t in timesteps))
        for index in range(args.snr_bins):
            low_edge, high_edge = edges[index], edges[index + 1]
            mask = (local_snr >= low_edge) & ((local_snr <= high_edge) if index == args.snr_bins - 1 else (local_snr < high_edge))
            if not mask.any():
                continue
            counts = Counter(best_patch_t[mask].astype(int))
            shares = [counts.get(t, 0) / mask.sum() for t in timesteps]
            emit(f"      {f'[{low_edge:.2f}, {high_edge:.2f}]':>22} {int(mask.sum()):>8} {best_patch_t[mask].mean():>12.1f}  "
                 + " ".join(f"{share:>7.1%}" for share in shares))
            bin_rows.append({"metric": name, "snr_low": f"{low_edge:.4f}", "snr_high": f"{high_edge:.4f}", "patches": int(mask.sum()),
                             "mean_best_t": f"{best_patch_t[mask].mean():.1f}", **{f"share_t{t}": f"{s:.4f}" for t, s in zip(timesteps, shares)}})

    emit("")
    emit("how to read: a clear gain in 2) -> a global adaptive t (predicted from the input) is worth building;")
    emit("             a gain in 3) clearly larger than in 2) -> spatially adaptive fusion is worth building;")
    emit("             small gains in both -> a fixed t is enough. Oracle numbers use GT and are upper bounds.")

    with open(os.path.join(args.out_dir, "per_patch_bins.csv"), "w", newline="", encoding="utf-8") as csv_file:
        if bin_rows:
            writer = csv.DictWriter(csv_file, fieldnames=list(bin_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bin_rows)
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")
    print(f"\nwritten: {os.path.join(args.out_dir, 'summary.txt')}, per_image.csv, per_patch_bins.csv")


if __name__ == "__main__":
    main()
