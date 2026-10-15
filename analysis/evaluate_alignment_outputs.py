"""
干预实验第 3 步: 评测对齐前后同场景输出的一致性

创建时间: 2026-10-14
功能说明:
  读取 make_alignment_inputs.py 生成的清单和模型在这些图上的输出, 对每一对比较
  "较亮输入的输出" 与 "较暗输入 (原图 / 四种对齐) 的输出" 之间的差异:
  全局色调差、亮度差 ΔL*、低频 ΔE00、逐像素 ΔE00; 同时检查输入端对齐后还剩多少色调差,
  以及较暗输入的输出相对 GT 的 PSNR (看对齐是否也让结果更准)。
  主表只用 "信息对" (输入本身有差别: 输入色调差 >= 1 或 |曝光差| >= 0.1 档), 全部对作为附表。
生成标记: Claude 生成

用法:
  python evaluate_alignment_outputs.py --test_low data/LOLv2/Real_captured/Test/Low \
      --test_gt data/LOLv2/Real_captured/Test/Normal --alignment_dir alignment_intervention/inputs \
      --output_dir alignment_intervention/outputs_official --name official --save_dir alignment_intervention
输出: save_dir 下的 alignment_summary_<name>.txt 和 alignment_pairs_<name>.csv
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import wilcoxon
from skimage.color import deltaE_ciede2000, rgb2lab

from diagnose_capture_difference import (block_average, index_images, linear_to_srgb, load_srgb, mean_luminance,
                                         srgb_to_linear)

VARIANTS = ['original', 'exposure', 'gain', 'gainoffset', 'matrix']
VARIANT_LABELS = {'original': '不对齐 (原图)', 'exposure': '只对齐曝光', 'gain': '逐通道增益',
                  'gainoffset': '增益+偏置', 'matrix': '3x3+偏置'}


def parse_arguments():
    parser = argparse.ArgumentParser(description='评测干预实验')
    parser.add_argument('--test_low', required=True)
    parser.add_argument('--test_gt', required=True)
    parser.add_argument('--alignment_dir', required=True, help='make_alignment_inputs.py 的输出目录 (含清单)')
    parser.add_argument('--output_dir', required=True, help='模型在 alignment_dir 上的推理输出')
    parser.add_argument('--name', default='model')
    parser.add_argument('--save_dir', default='alignment_intervention')
    parser.add_argument('--block_size', type=int, default=8)
    return parser.parse_args()


def exposure_normalized_lab_mean(low_srgb, ground_truth_srgb):
    """只用一个标量把输入提亮到 GT 的平均亮度 (不改通道比例), 返回平均 Lab; 与诊断脚本的输入色调差算法相同"""
    low_linear = srgb_to_linear(low_srgb)
    scale = mean_luminance(srgb_to_linear(ground_truth_srgb)) / max(mean_luminance(low_linear), 1e-8)
    return rgb2lab(linear_to_srgb(low_linear * scale)).reshape(-1, 3).mean(axis=0)


def compare_outputs(bright_output, bright_lab, dark_output, ground_truth_srgb, block_size):
    dark_lab = rgb2lab(dark_output)
    delta = bright_lab.reshape(-1, 3).mean(axis=0) - dark_lab.reshape(-1, 3).mean(axis=0)
    dark_lowpass = block_average(dark_output, block_size)
    return {
        'output_tone_difference': float(np.hypot(delta[1], delta[2])),
        'output_delta_lightness': float(delta[0]),
        'output_lowpass_delta_e': float(deltaE_ciede2000(rgb2lab(block_average(bright_output, block_size)), rgb2lab(dark_lowpass)).mean()),
        'output_pixel_delta_e': float(deltaE_ciede2000(bright_lab, dark_lab).mean()),
        'dark_output_psnr': float(10 * np.log10(1 / max(np.mean((dark_output - ground_truth_srgb) ** 2), 1e-12))),
    }


def describe(values, digits=2):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return (f'{np.mean(values):.{digits}f}', f'{np.median(values):.{digits}f}') if values.size else ('-', '-')


def summarize(rows, title, lines):
    lines.append(f'\n== {title} ({len(rows)} 对)')
    if not rows:
        return
    header = (f'  {"对齐方式":14s}{"输入色调差":>10s}{"输出色调差 均值/中位":>20s}{">3的对":>8s}{"|ΔL*| 均值":>11s}'
              f'{"低频ΔE":>9s}{"逐像素ΔE":>10s}{"色调差降幅":>11s}{"|ΔL*|降幅":>10s}{"较暗输出PSNR":>13s}')
    lines.append(header)
    baseline_tone = np.mean([row['original:output_tone_difference'] for row in rows])
    baseline_lightness = np.mean([abs(row['original:output_delta_lightness']) for row in rows])
    for variant in VARIANTS:
        tone = np.array([row[f'{variant}:output_tone_difference'] for row in rows])
        lightness = np.abs(np.array([row[f'{variant}:output_delta_lightness'] for row in rows]))
        tone_mean, tone_median = describe(tone)
        lines.append(f'  {VARIANT_LABELS[variant]:14s}{describe([row[f"{variant}:input_tone_difference"] for row in rows])[0]:>10s}'
                     f'{tone_mean + " / " + tone_median:>20s}{np.mean(tone > 3):>8.0%}{np.mean(lightness):>11.2f}'
                     f'{describe([row[f"{variant}:output_lowpass_delta_e"] for row in rows])[0]:>9s}'
                     f'{describe([row[f"{variant}:output_pixel_delta_e"] for row in rows])[0]:>10s}'
                     f'{1 - np.mean(tone) / max(baseline_tone, 1e-9):>11.0%}{1 - np.mean(lightness) / max(baseline_lightness, 1e-9):>10.0%}'
                     f'{describe([row[f"{variant}:dark_output_psnr"] for row in rows])[0]:>13s}')
    lines.append('  逐对比较 (Wilcoxon 符号秩检验, 输出色调差):')
    for first, second in [('original', 'gain'), ('gain', 'gainoffset'), ('gainoffset', 'matrix'), ('original', 'gainoffset')]:
        first_values = np.array([row[f'{first}:output_tone_difference'] for row in rows])
        second_values = np.array([row[f'{second}:output_tone_difference'] for row in rows])
        better = np.mean(second_values < first_values)
        p_value = wilcoxon(first_values, second_values).pvalue if len(rows) >= 6 and np.any(first_values != second_values) else np.nan
        lines.append(f'    {VARIANT_LABELS[second]} 比 {VARIANT_LABELS[first]} 更一致的对: {better:.0%}   p = {p_value:.2g}')


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.save_dir, exist_ok=True)
    ground_truth_index = index_images(arguments.test_gt)
    with open(os.path.join(arguments.alignment_dir, 'alignment_manifest.csv'), encoding='utf-8-sig') as handle:
        manifest = [row for row in csv.DictReader(handle) if row.get('skipped', '0') in ('0', '')]
    rows = []
    for pair in manifest:
        bright_id, dark_id, pair_name = pair['bright_id'], pair['dark_id'], pair['pair']
        ground_truth_srgb = load_srgb(ground_truth_index[pair['group'].split('/')[0]])
        size = (ground_truth_srgb.shape[1], ground_truth_srgb.shape[0])
        input_names = {'original': f'{dark_id}_original.png'}
        input_names.update({variant: f'{pair_name}_{variant}.png' for variant in VARIANTS[1:]})
        bright_output_path = os.path.join(arguments.output_dir, f'{bright_id}_original.png')
        if not os.path.exists(bright_output_path):
            print(f'缺少输出 {bright_output_path}, 跳过')
            continue
        bright_output = load_srgb(bright_output_path, size)
        bright_lab = rgb2lab(bright_output)
        bright_input_lab = exposure_normalized_lab_mean(load_srgb(os.path.join(arguments.alignment_dir, f'{bright_id}_original.png')),
                                                        ground_truth_srgb)
        row = {'pair': pair_name, 'bright_id': bright_id, 'dark_id': dark_id,
               'fitted_exposure_log2_ratio': float(pair['fitted_exposure_log2_ratio'])}
        for variant, file_name in input_names.items():
            input_delta = bright_input_lab - exposure_normalized_lab_mean(
                load_srgb(os.path.join(arguments.alignment_dir, file_name)), ground_truth_srgb)
            row[f'{variant}:input_tone_difference'] = float(np.hypot(input_delta[1], input_delta[2]))
            metrics = compare_outputs(bright_output, bright_lab, load_srgb(os.path.join(arguments.output_dir, file_name), size),
                                      ground_truth_srgb, arguments.block_size)
            row.update({f'{variant}:{key}': value for key, value in metrics.items()})
        rows.append(row)
    informative = [row for row in rows if row['original:input_tone_difference'] >= 1 or abs(row['fitted_exposure_log2_ratio']) >= 0.1]

    lines = [f'干预实验: 对齐较暗输入后, 同场景输出是否变一致 (模型 {arguments.name})',
             '读法: "增益+偏置" 的降幅明显大于 "逐通道增益" -> 不一致主要由采集间的增益和偏置差造成, 一致性训练的扰动应包含这两项;',
             '      对齐后仍剩下的差异来自噪声和局部内容, 逐像素 ΔE 不会降到 0 (只换噪声样本时约 2.4), 主要看全局色调差和 |ΔL*|;',
             '      "输入色调差" 一列应在对齐后接近 0, 用来确认对齐本身有效;',
             '      较暗输入里被压到 0 的像素 (常见于暗部地面) 信息已丢失, 任何全局对齐都补不回, 对齐后剩下的差异有一部分来自这里.']
    summarize(informative, '信息对: 输入本身有差别', lines)
    summarize(rows, '全部对', lines)
    lines.append('\n== 信息对中原始色调差最大的 12 对: 输出色调差 / |ΔL*| (不对齐 -> 只曝光 -> 增益 -> 增益+偏置 -> 3x3)')
    for row in sorted(informative, key=lambda row: -row['original:output_tone_difference'])[:12]:
        cells = '  '.join(f'{row[f"{variant}:output_tone_difference"]:5.2f}/{abs(row[f"{variant}:output_delta_lightness"]):4.1f}'
                          for variant in VARIANTS)
        lines.append(f'  {row["pair"]:16s} {cells}')

    keys = list(rows[0]) if rows else []
    with open(os.path.join(arguments.save_dir, f'alignment_pairs_{arguments.name}.csv'), 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: (f'{value:.6g}' if isinstance(value, float) else value) for key, value in row.items()} for row in rows])
    summary = '\n'.join(lines)
    with open(os.path.join(arguments.save_dir, f'alignment_summary_{arguments.name}.txt'), 'w', encoding='utf-8') as handle:
        handle.write(summary + '\n')
    print(summary)


if __name__ == '__main__':
    main()
