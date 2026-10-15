"""
干预实验第 1 步: 把同场景的较暗输入按不同的全局模型对齐到较亮输入, 生成待推理的图片

创建时间: 2026-10-14
功能说明:
  对测试集每一对同场景输入 (较亮者 a, 较暗者 b), 在线性光空间拟合 b -> a 的全局变换 (与诊断脚本完全相同的拟合方式),
  然后把 b 变换后另存, 共四种对齐:
    exposure    只对齐整体曝光 (一个标量增益)
    gain        逐通道增益 (相当于白平衡 + 曝光)
    gainoffset  逐通道增益 + 逐通道偏置
    matrix      3x3 矩阵 + 偏置 (上限对照)
  同时把每张原始低光图另存为 <编号>_original.png, 让原图和对齐图走同一个推理脚本。
  如果 "增益+偏置" 对齐后两张输出的差异大幅下降, 而 "只用增益" 降得少, 就说明不一致是由采集间的增益和偏置差造成的。
生成标记: Claude 生成

用法 (与 diagnose_capture_difference.py 放在同一目录):
  python make_alignment_inputs.py --test_low data/LOLv2/Real_captured/Test/Low \
      --test_gt data/LOLv2/Real_captured/Test/Normal --save_dir alignment_intervention/inputs
输出: save_dir 下的 png (直接作为推理输入) 和 alignment_manifest.csv (每对的拟合参数, 评测脚本要用)
"""
import argparse
import csv
import itertools
import os

import numpy as np
from PIL import Image

from diagnose_capture_difference import (LUMINANCE_WEIGHTS, block_average, fit_global_models, group_by_ground_truth,
                                         index_images, linear_to_srgb, load_srgb, mean_luminance,
                                         select_inlier_blocks, srgb_to_linear)

ALIGNMENT_SUFFIXES = {'scalar_gain': 'exposure', 'channel_gain': 'gain',
                      'channel_gain_offset': 'gainoffset', 'matrix_offset': 'matrix'}


def parse_arguments():
    parser = argparse.ArgumentParser(description='生成干预实验的对齐输入')
    parser.add_argument('--test_low', required=True)
    parser.add_argument('--test_gt', required=True)
    parser.add_argument('--save_dir', default='alignment_intervention/inputs')
    parser.add_argument('--block_size', type=int, default=8)
    parser.add_argument('--valid_low', type=int, default=2)
    parser.add_argument('--valid_high', type=int, default=250)
    return parser.parse_args()


def save_srgb(image, path):
    Image.fromarray(np.clip(np.round(image * 255), 0, 255).astype(np.uint8)).save(path)


def apply_alignment(model_name, parameters, linear_image):
    """把拟合出的全局变换作用到整张线性光图像上"""
    if model_name == 'scalar_gain':
        return linear_image * parameters[0]
    if model_name == 'channel_gain':
        return linear_image * parameters
    if model_name == 'channel_gain_offset':
        return linear_image * parameters[:, 0] + parameters[:, 1]
    if model_name == 'matrix_offset':
        ones = np.ones(linear_image.shape[:2] + (1,))
        return np.concatenate([linear_image, ones], axis=2) @ parameters
    raise ValueError(model_name)


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.save_dir, exist_ok=True)
    low_index, ground_truth_index = index_images(arguments.test_low), index_images(arguments.test_gt)
    groups, _ = group_by_ground_truth(low_index, ground_truth_index)
    manifest_rows, saved_originals = [], set()
    for group in groups:
        size = Image.open(ground_truth_index[group[0]]).size
        images = {image_id: load_srgb(low_index[image_id], size) for image_id in group}
        for image_id, image in images.items():
            if image_id not in saved_originals:
                save_srgb(image, os.path.join(arguments.save_dir, f'{image_id}_original.png'))
                saved_originals.add(image_id)
        for first_id, second_id in itertools.combinations(group, 2):
            first_linear, second_linear = srgb_to_linear(images[first_id]), srgb_to_linear(images[second_id])
            if mean_luminance(first_linear) >= mean_luminance(second_linear):
                bright_id, dark_id, bright_linear, dark_linear = first_id, second_id, first_linear, second_linear
            else:
                bright_id, dark_id, bright_linear, dark_linear = second_id, first_id, second_linear, first_linear
            bright_srgb, dark_srgb = images[bright_id], images[dark_id]
            valid_pixels = np.all((bright_srgb >= arguments.valid_low / 255) & (bright_srgb <= arguments.valid_high / 255)
                                  & (dark_srgb >= arguments.valid_low / 255) & (dark_srgb <= arguments.valid_high / 255), axis=2)
            block_valid = block_average(valid_pixels[..., None].astype(np.float64), arguments.block_size)[..., 0] >= 0.8
            bright_blocks = block_average(bright_linear, arguments.block_size)[block_valid]
            dark_blocks = block_average(dark_linear, arguments.block_size)[block_valid]
            row = {'pair': f'{dark_id}_to_{bright_id}', 'bright_id': bright_id, 'dark_id': dark_id,
                   'group': '/'.join(group), 'valid_block_count': int(bright_blocks.shape[0])}
            if bright_blocks.shape[0] < 50:
                row['skipped'] = 1
                manifest_rows.append(row)
                print(f'跳过 {row["pair"]}: 有效块只有 {bright_blocks.shape[0]} 个')
                continue
            inliers = select_inlier_blocks(dark_blocks, bright_blocks)
            fits = fit_global_models(dark_blocks[inliers], bright_blocks[inliers])
            for model_name, suffix in ALIGNMENT_SUFFIXES.items():
                aligned = apply_alignment(model_name, fits[model_name][0], dark_linear)
                save_srgb(linear_to_srgb(aligned), os.path.join(arguments.save_dir, f'{row["pair"]}_{suffix}.png'))
            gain_offset = fits['channel_gain_offset'][0]
            row.update(skipped=0,
                       fitted_exposure_log2_ratio=float(np.log2(max(gain_offset[:, 0] @ LUMINANCE_WEIGHTS, 1e-8))),
                       gain_red=float(fits['channel_gain'][0][0]), gain_green=float(fits['channel_gain'][0][1]),
                       gain_blue=float(fits['channel_gain'][0][2]),
                       offset_red=float(gain_offset[0, 1]), offset_green=float(gain_offset[1, 1]), offset_blue=float(gain_offset[2, 1]))
            manifest_rows.append(row)
    keys = []
    for row in manifest_rows:
        keys += [key for key in row if key not in keys]
    with open(os.path.join(arguments.save_dir, 'alignment_manifest.csv'), 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f'{len(groups)} 个场景, {len(manifest_rows)} 对 (跳过 {sum(row["skipped"] for row in manifest_rows)} 对); '
          f'原图 {len(saved_originals)} 张 + 对齐图 {4 * sum(1 - row["skipped"] for row in manifest_rows)} 张 -> {arguments.save_dir}')


if __name__ == '__main__':
    main()
