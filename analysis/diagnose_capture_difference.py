"""
诊断同一场景多次低光采集之间的差异, 以及这些差异与增强输出不一致的关系 (LOL-v2-Real)

创建时间: 2026-10-14
功能说明:
  1. 按 GT 图像内容把低光图分组 (同一组 = 同一场景、同一 GT、不同的低光采集), 测试集和训练集都做
  2. 对组内每一对输入 (较亮者记为 a, 较暗者记为 b), 在线性光空间里拟合 b -> a 的五种全局模型
     (整体增益 / 逐通道增益 / 逐通道增益+偏置 / 3x3 矩阵+偏置 / 逐通道三次曲线), 并量化全局模型拟合后
     剩下的低频亮度、低频色度和高频(噪声)差异, 以及两张输入的噪声水平和压暗像素比例
  3. 对每个模型输出目录, 计算同组输出之间的全局色调差、亮度差和 ΔE00, 再看输出不一致与哪种输入差异相关
  4. 输出结论汇总和逐对数据, 训练集的逐对参数可直接用于之后构造数据增强
生成标记: Claude 生成

用法 (在服务器上, 路径按实际情况修改):
  python diagnose_capture_difference.py \
      --test_low  data/LOLv2/Real_captured/Test/Low \
      --test_gt   data/LOLv2/Real_captured/Test/Normal \
      --train_low data/LOLv2/Real_captured/Train/Low \
      --train_gt  data/LOLv2/Real_captured/Train/Normal \
      --outputs Retinexformer=results_color_exp/orig/test A=results_color_exp/A/test \
      --save_dir capture_diagnosis
  文件按名字里的数字编号配对 (low00690.png / normal00690.png / 00690.png / 00690_normal.png 都能认出)。
  输出尺寸与 GT 不同时 (如 ReDDiT 的 592x400) 会先缩放到 GT 尺寸。

输出 (save_dir 下):
  summary.txt      结论汇总 (开头有读法说明)
  test_pairs.csv   测试集每一对: 输入差异特征 + 各模型的输出不一致
  train_pairs.csv  训练集每一对: 输入差异特征 (不含模型输出)

依赖: numpy, scipy, scikit-image, Pillow
"""
import argparse
import csv
import hashlib
import itertools
import os
import re
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from skimage.color import deltaE_ciede2000, rgb2lab

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
MODEL_NAMES = ['scalar_gain', 'channel_gain', 'channel_gain_offset', 'matrix_offset', 'channel_cubic']

# 用于相关分析的输入差异特征 (全部按 "较亮输入 a 相对较暗输入 b" 计算)
FEATURE_DESCRIPTIONS = {
    'exposure_log2_ratio': '平均亮度差 log2(Ya/Yb)',
    'fitted_exposure_log2_ratio': '曝光差 (增益+偏置拟合的增益, 不受偏置影响)',
    'input_tone_difference': '输入色调差 (曝光归一化后平均 a*b* 之差)',
    'gain_chroma_spread': '逐通道增益的通道间差异 std(log g)',
    'offset_magnitude': '逐通道偏置大小 mean|o/mean(a)|',
    'offset_channel_spread': '偏置的通道间差异 std(o/mean(a))',
    'offset_improvement': '加偏置后残差下降比例',
    'cross_channel_improvement': '3x3 相对 增益+偏置 的残差下降比例',
    'nonlinear_improvement': '三次曲线 相对 增益+偏置 的残差下降比例',
    'low_frequency_luminance_residual': '全局对齐后的低频亮度残差',
    'low_frequency_chroma_residual': '全局对齐后的低频色度残差 (彩色斑块)',
    'high_frequency_residual': '全局对齐后的高频残差 (噪声)',
    'noise_log_ratio': '相对噪声之比 log(σb/σa)',
    'minimum_signal_to_noise': '两张中较低的信噪比',
    'crushed_fraction_difference': '压暗像素比例之差 (较暗减较亮)',
}
# 多变量回归只用互不重复的特征 (去掉与其他特征几乎等价的几项), 防止 80 对样本上过拟合
MULTIVARIATE_FEATURES = ['fitted_exposure_log2_ratio', 'input_tone_difference', 'offset_magnitude', 'offset_channel_spread',
                         'cross_channel_improvement', 'nonlinear_improvement', 'low_frequency_luminance_residual',
                         'low_frequency_chroma_residual', 'noise_log_ratio', 'crushed_fraction_difference']
# 这几项在物理上随曝光差一起变化, 另外报告控制曝光差后的偏相关
EXPOSURE_LINKED_FEATURES = ['exposure_log2_ratio', 'noise_log_ratio', 'crushed_fraction_difference', 'minimum_signal_to_noise']
OUTPUT_TARGET_DESCRIPTIONS = {
    'output_tone_difference': '输出全局色调差',
    'output_absolute_delta_lightness': '输出亮度差 |ΔL*|',
    'output_lowpass_delta_e': '输出低频 ΔE00',
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='诊断同场景多次低光采集的差异及其与输出不一致的关系')
    parser.add_argument('--test_low', required=True, help='测试集低光图目录')
    parser.add_argument('--test_gt', required=True, help='测试集 GT 目录')
    parser.add_argument('--train_low', default=None, help='训练集低光图目录 (可选)')
    parser.add_argument('--train_gt', default=None, help='训练集 GT 目录 (可选)')
    parser.add_argument('--outputs', nargs='*', default=[], help='模型输出目录, 格式 名字=路径, 可给多个')
    parser.add_argument('--save_dir', default='capture_diagnosis')
    parser.add_argument('--block_size', type=int, default=8, help='全局模型拟合前的块平均大小 (压噪声)')
    parser.add_argument('--lowpass_sigma', type=float, default=8.0, help='低频残差的高斯核标准差 (像素)')
    parser.add_argument('--valid_low', type=int, default=2, help='有效像素下限 (8bit), 低于此视为压暗')
    parser.add_argument('--valid_high', type=int, default=250, help='有效像素上限 (8bit), 高于此视为过曝')
    parser.add_argument('--max_train_pairs', type=int, default=0, help='训练集最多分析多少对, 0 = 全部')
    return parser.parse_args()


# ---------------------------------------------------------------- 读图与基础变换
def extract_image_id(file_name):
    """取文件名里第一段不少于 3 位的数字作为编号, 例如 low00690 / normal00690 / 00690_normal -> 00690"""
    match = re.search(r'\d{3,}', os.path.splitext(os.path.basename(file_name))[0])
    return match.group(0) if match else None


def index_images(directory):
    index = {}
    for file_name in sorted(os.listdir(directory)):
        if file_name.lower().endswith(IMAGE_EXTENSIONS):
            image_id = extract_image_id(file_name)
            if image_id is not None:
                index[image_id] = os.path.join(directory, file_name)
    return index


def load_srgb(path, target_size=None):
    image = Image.open(path).convert('RGB')
    if target_size is not None and image.size != target_size:
        image = image.resize(target_size, Image.BICUBIC)
    return np.asarray(image).astype(np.float64) / 255.0


def srgb_to_linear(image):
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image):
    image = np.clip(image, 0, 1)
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1 / 2.4) - 0.055)


def mean_luminance(linear_image):
    return float((linear_image.reshape(-1, 3) @ LUMINANCE_WEIGHTS).mean())


def block_average(image, block_size):
    height = image.shape[0] // block_size * block_size
    width = image.shape[1] // block_size * block_size
    cropped = image[:height, :width]
    return cropped.reshape(height // block_size, block_size, width // block_size, block_size, -1).mean(axis=(1, 3))


def group_by_ground_truth(low_index, ground_truth_index):
    """按 GT 像素内容的哈希分组, 返回 (多张一组的分组列表, 单张场景数)"""
    groups_by_hash = {}
    for image_id in sorted(set(low_index) & set(ground_truth_index)):
        pixels = np.asarray(Image.open(ground_truth_index[image_id]).convert('RGB'))
        digest = hashlib.md5(pixels.tobytes()).hexdigest()
        groups_by_hash.setdefault(digest, []).append(image_id)
    groups = sorted(group for group in groups_by_hash.values() if len(group) >= 2)
    single_count = sum(1 for group in groups_by_hash.values() if len(group) == 1)
    return groups, single_count


# ---------------------------------------------------------------- 单张图的统计量
def compute_image_statistics(low_srgb, ground_truth_srgb, valid_low, valid_high):
    """单张低光图: 平均亮度、相对噪声、压暗比例、曝光归一化后的平均 Lab"""
    low_linear = srgb_to_linear(low_srgb)
    ground_truth_linear = srgb_to_linear(ground_truth_srgb)
    luminance = mean_luminance(low_linear)
    valid_pixels = np.all((low_srgb >= valid_low / 255) & (low_srgb <= valid_high / 255), axis=2)
    relative_noise_per_channel = []
    for channel in range(3):
        high_pass = low_linear[..., channel] - gaussian_filter(low_linear[..., channel], 1.5)
        values = high_pass[valid_pixels] if valid_pixels.sum() > 1000 else high_pass.ravel()
        noise_sigma = 1.4826 * np.median(np.abs(values - np.median(values)))
        channel_mean = low_linear[..., channel][valid_pixels].mean() if valid_pixels.sum() > 1000 else low_linear[..., channel].mean()
        relative_noise_per_channel.append(noise_sigma / max(channel_mean, 1e-6))
    # 只用一个标量把输入提亮到 GT 的平均亮度, 不改通道比例 -> 保留输入本身的色偏
    exposure_scale = mean_luminance(ground_truth_linear) / max(luminance, 1e-8)
    brightened_lab = rgb2lab(linear_to_srgb(low_linear * exposure_scale)).reshape(-1, 3).mean(axis=0)
    return {
        'luminance': luminance,
        'relative_noise': float(np.mean(relative_noise_per_channel)),
        'crushed_fraction': float(np.mean(np.any(low_srgb < valid_low / 255, axis=2))),
        'normalized_lab': brightened_lab,
    }


# ---------------------------------------------------------------- 两张输入之间的全局模型拟合
def least_squares(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def select_inlier_blocks(source_blocks, target_blocks, keep_fraction=0.9):
    """先用 逐通道增益+偏置 粗拟合, 去掉残差最大的 10% 的块 (局部运动、高光等), 所有模型都在同一批内点上比较"""
    ones = np.ones((source_blocks.shape[0], 1))
    residual = np.zeros(source_blocks.shape[0])
    for channel in range(3):
        design = np.hstack([source_blocks[:, channel:channel + 1], ones])
        residual += (design @ least_squares(design, target_blocks[:, channel]) - target_blocks[:, channel]) ** 2
    return residual <= np.quantile(residual, keep_fraction)


def fit_global_models(source_blocks, target_blocks):
    """拟合 source(b) -> target(a) 的五种全局模型; 返回 {模型名: (参数, 预测值 N x 3)}"""
    ones = np.ones((source_blocks.shape[0], 1))
    results = {}
    scalar = least_squares(source_blocks.reshape(-1, 1), target_blocks.reshape(-1))
    results['scalar_gain'] = (scalar, source_blocks * scalar[0])
    channel_gains, gain_offset_parameters, cubic_parameters = [], [], []
    channel_gain_prediction = np.zeros_like(target_blocks)
    gain_offset_prediction = np.zeros_like(target_blocks)
    cubic_prediction = np.zeros_like(target_blocks)
    for channel in range(3):
        source = source_blocks[:, channel:channel + 1]
        target = target_blocks[:, channel]
        gain = least_squares(source, target)
        channel_gains.append(gain[0])
        channel_gain_prediction[:, channel] = source[:, 0] * gain[0]
        gain_offset_design = np.hstack([source, ones])
        gain_offset = least_squares(gain_offset_design, target)
        gain_offset_parameters.append(gain_offset)
        gain_offset_prediction[:, channel] = gain_offset_design @ gain_offset
        normalized = source / max(source.max(), 1e-8)
        cubic_design = np.hstack([ones, normalized, normalized ** 2, normalized ** 3])
        cubic = least_squares(cubic_design, target)
        cubic_parameters.append(cubic)
        cubic_prediction[:, channel] = cubic_design @ cubic
    results['channel_gain'] = (np.array(channel_gains), channel_gain_prediction)
    results['channel_gain_offset'] = (np.array(gain_offset_parameters), gain_offset_prediction)
    results['channel_cubic'] = (np.array(cubic_parameters), cubic_prediction)
    matrix_design = np.hstack([source_blocks, ones])
    matrix_parameters = np.stack([least_squares(matrix_design, target_blocks[:, channel]) for channel in range(3)], axis=1)
    results['matrix_offset'] = (matrix_parameters, matrix_design @ matrix_parameters)
    return results


def relative_residual(prediction, target_blocks):
    return float(np.sqrt(np.mean((prediction - target_blocks) ** 2)) / max(target_blocks.mean(), 1e-8))


def spatial_residuals(bright_linear, dark_linear, gain_offset_parameters, valid_pixels, lowpass_sigma):
    """把较暗输入按 逐通道增益+偏置 对齐到较亮输入后, 剩余差异的低频亮度/低频色度/高频部分 (相对平均亮度)"""
    aligned = dark_linear * gain_offset_parameters[:, 0] + gain_offset_parameters[:, 1]
    residual = bright_linear - aligned
    weight = valid_pixels.astype(np.float64)
    smoothed_weight = gaussian_filter(weight, lowpass_sigma)
    lowpass = np.stack([gaussian_filter(residual[..., channel] * weight, lowpass_sigma) / np.maximum(smoothed_weight, 1e-6)
                        for channel in range(3)], axis=2)
    reliable = smoothed_weight > 0.5
    reference = max(bright_linear[valid_pixels].mean(), 1e-8) if valid_pixels.any() else max(bright_linear.mean(), 1e-8)
    if reliable.sum() < 100:
        return np.nan, np.nan, np.nan
    lowpass_luminance = lowpass.mean(axis=2)[reliable]
    lowpass_chroma_red = (lowpass[..., 0] - lowpass[..., 1])[reliable]
    lowpass_chroma_blue = (lowpass[..., 2] - lowpass[..., 1])[reliable]
    high_frequency = (residual - lowpass)[reliable & valid_pixels] if (reliable & valid_pixels).any() else (residual - lowpass)[reliable]
    return (float(np.sqrt(np.mean(lowpass_luminance ** 2)) / reference),
            float(np.sqrt(np.mean(lowpass_chroma_red ** 2 + lowpass_chroma_blue ** 2)) / reference),
            float(np.sqrt(np.mean(high_frequency ** 2)) / reference))


def compute_pair_input_features(first_id, second_id, low_index, ground_truth_path, statistics_cache, arguments):
    """一对输入的差异特征; 较亮者为 a (bright), 较暗者为 b (dark)"""
    ground_truth_srgb = load_srgb(ground_truth_path)
    images = {}
    for image_id in (first_id, second_id):
        images[image_id] = load_srgb(low_index[image_id], (ground_truth_srgb.shape[1], ground_truth_srgb.shape[0]))
        if image_id not in statistics_cache:
            statistics_cache[image_id] = compute_image_statistics(images[image_id], ground_truth_srgb,
                                                                  arguments.valid_low, arguments.valid_high)
    if statistics_cache[first_id]['luminance'] >= statistics_cache[second_id]['luminance']:
        bright_id, dark_id = first_id, second_id
    else:
        bright_id, dark_id = second_id, first_id
    bright_srgb, dark_srgb = images[bright_id], images[dark_id]
    bright_linear, dark_linear = srgb_to_linear(bright_srgb), srgb_to_linear(dark_srgb)
    bright_statistics, dark_statistics = statistics_cache[bright_id], statistics_cache[dark_id]

    valid_pixels = np.all((bright_srgb >= arguments.valid_low / 255) & (bright_srgb <= arguments.valid_high / 255)
                          & (dark_srgb >= arguments.valid_low / 255) & (dark_srgb <= arguments.valid_high / 255), axis=2)
    block_valid = block_average(valid_pixels[..., None].astype(np.float64), arguments.block_size)[..., 0] >= 0.8
    bright_blocks = block_average(bright_linear, arguments.block_size)[block_valid]
    dark_blocks = block_average(dark_linear, arguments.block_size)[block_valid]

    features = {'bright_id': bright_id, 'dark_id': dark_id,
                'valid_block_fraction': float(block_valid.mean()),
                'exposure_log2_ratio': float(np.log2(bright_statistics['luminance'] / max(dark_statistics['luminance'], 1e-8)))}
    input_delta = bright_statistics['normalized_lab'] - dark_statistics['normalized_lab']
    features.update(input_delta_a_star=float(input_delta[1]), input_delta_b_star=float(input_delta[2]),
                    input_tone_difference=float(np.hypot(input_delta[1], input_delta[2])))
    features.update(noise_log_ratio=float(np.log(max(dark_statistics['relative_noise'], 1e-8) / max(bright_statistics['relative_noise'], 1e-8))),
                    minimum_signal_to_noise=float(1 / max(dark_statistics['relative_noise'], bright_statistics['relative_noise'], 1e-8)),
                    crushed_fraction_bright=bright_statistics['crushed_fraction'],
                    crushed_fraction_dark=dark_statistics['crushed_fraction'],
                    crushed_fraction_difference=dark_statistics['crushed_fraction'] - bright_statistics['crushed_fraction'])

    if bright_blocks.shape[0] < 50:  # 有效区域太少, 拟合不可靠
        for model_name in MODEL_NAMES:
            features[f'residual_{model_name}'] = np.nan
        return features
    inliers = select_inlier_blocks(dark_blocks, bright_blocks)
    dark_blocks, bright_blocks = dark_blocks[inliers], bright_blocks[inliers]
    fits = fit_global_models(dark_blocks, bright_blocks)
    for model_name in MODEL_NAMES:
        features[f'residual_{model_name}'] = relative_residual(fits[model_name][1], bright_blocks)
    channel_gains = fits['channel_gain'][0]
    gain_offset_parameters = fits['channel_gain_offset'][0]
    channel_means = bright_blocks.mean(axis=0)
    relative_offsets = gain_offset_parameters[:, 1] / np.maximum(channel_means, 1e-8)
    for channel, channel_name in enumerate(('red', 'green', 'blue')):
        features[f'gain_{channel_name}'] = float(channel_gains[channel])
        features[f'offset_gain_{channel_name}'] = float(gain_offset_parameters[channel, 0])
        features[f'offset_{channel_name}_relative'] = float(relative_offsets[channel])
    features['fitted_exposure_log2_ratio'] = float(np.log2(max(gain_offset_parameters[:, 0] @ LUMINANCE_WEIGHTS, 1e-8)))
    features['gain_chroma_spread'] = float(np.std(np.log(np.maximum(channel_gains, 1e-8))))
    features['offset_magnitude'] = float(np.mean(np.abs(relative_offsets)))
    features['offset_channel_spread'] = float(np.std(relative_offsets))
    base = features['residual_channel_gain_offset']
    features['offset_improvement'] = 1 - base / max(features['residual_channel_gain'], 1e-12)
    features['cross_channel_improvement'] = 1 - features['residual_matrix_offset'] / max(base, 1e-12)
    features['nonlinear_improvement'] = 1 - features['residual_channel_cubic'] / max(base, 1e-12)
    (features['low_frequency_luminance_residual'], features['low_frequency_chroma_residual'],
     features['high_frequency_residual']) = spatial_residuals(bright_linear, dark_linear, gain_offset_parameters,
                                                              valid_pixels, arguments.lowpass_sigma)
    return features


# ---------------------------------------------------------------- 输出端的不一致
def compute_pair_output_metrics(bright_id, dark_id, output_index, ground_truth_srgb, block_size):
    size = (ground_truth_srgb.shape[1], ground_truth_srgb.shape[0])
    bright_output = load_srgb(output_index[bright_id], size)
    dark_output = load_srgb(output_index[dark_id], size)
    bright_lab, dark_lab = rgb2lab(bright_output), rgb2lab(dark_output)
    delta = bright_lab.reshape(-1, 3).mean(axis=0) - dark_lab.reshape(-1, 3).mean(axis=0)
    bright_lowpass_lab = rgb2lab(block_average(bright_output, block_size))
    dark_lowpass_lab = rgb2lab(block_average(dark_output, block_size))
    ground_truth_lowpass_lab = rgb2lab(block_average(ground_truth_srgb, block_size))
    return {
        'output_delta_lightness': float(delta[0]),
        'output_absolute_delta_lightness': float(abs(delta[0])),
        'output_delta_a_star': float(delta[1]),
        'output_delta_b_star': float(delta[2]),
        'output_tone_difference': float(np.hypot(delta[1], delta[2])),
        'output_lowpass_delta_e': float(deltaE_ciede2000(bright_lowpass_lab, dark_lowpass_lab).mean()),
        'output_pixel_delta_e': float(deltaE_ciede2000(bright_lab, dark_lab).mean()),
        'output_lowpass_error_to_ground_truth': float((deltaE_ciede2000(bright_lowpass_lab, ground_truth_lowpass_lab).mean()
                                                       + deltaE_ciede2000(dark_lowpass_lab, ground_truth_lowpass_lab).mean()) / 2),
    }


# ---------------------------------------------------------------- 统计与汇总
def column(rows, key):
    return np.array([row.get(key, np.nan) for row in rows], dtype=np.float64)


def percentile_text(values, digits=3):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return '-'
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return f'{p50:.{digits}f} [{p10:.{digits}f}, {p90:.{digits}f}]'


def leave_one_out_r2(feature_matrix, target, ridge):
    """标准化特征的岭回归, 用帽子矩阵算留一法 R2"""
    standardized = (feature_matrix - feature_matrix.mean(axis=0)) / np.maximum(feature_matrix.std(axis=0), 1e-12)
    design = np.hstack([np.ones((standardized.shape[0], 1)), standardized])
    penalty = ridge * standardized.shape[0] * np.eye(design.shape[1])
    penalty[0, 0] = 0
    inverse = np.linalg.inv(design.T @ design + penalty)
    coefficients = inverse @ design.T @ target
    hat_diagonal = np.einsum('ij,jk,ik->i', design, inverse, design)
    residual = target - design @ coefficients
    loo_residual = residual / np.maximum(1 - hat_diagonal, 1e-6)
    r2 = 1 - np.sum(loo_residual ** 2) / max(np.sum((target - target.mean()) ** 2), 1e-12)
    return float(r2), coefficients[1:]


def best_leave_one_out_r2(feature_matrix, target):
    """在几档正则强度里取留一法 R2 最高的一档 (诊断用, 略偏乐观)"""
    candidates = [(ridge,) + leave_one_out_r2(feature_matrix, target, ridge) for ridge in (0.01, 0.1, 0.3, 1.0, 3.0)]
    return max(candidates, key=lambda candidate: candidate[1])


def partial_spearman(feature_values, target_values, control_values):
    """控制一个变量后的 Spearman 偏相关: 三者先转秩, 再各自对控制变量做线性回归取残差, 求残差的相关"""
    ranks = [np.argsort(np.argsort(values)).astype(np.float64) for values in (feature_values, target_values, control_values)]
    control_design = np.column_stack([np.ones_like(ranks[2]), ranks[2]])
    residuals = [values - control_design @ least_squares(control_design, values) for values in ranks[:2]]
    return float(np.corrcoef(residuals[0], residuals[1])[0, 1])


def summarize_input_differences(rows, title, lines):
    lines.append(f'\n== {title}: 两次采集之间差在哪 ({len(rows)} 对; 数值为 中位数 [P10, P90])')
    lines.append(f'  有效块比例                {percentile_text(column(rows, "valid_block_fraction"), 2)}')
    lines.append(f'  平均亮度差 log2(Ya/Yb)    {percentile_text(column(rows, "exposure_log2_ratio"), 2)}')
    lines.append(f'  曝光差 (拟合增益, log2)   {percentile_text(column(rows, "fitted_exposure_log2_ratio"), 2)}')
    lines.append(f'  输入色调差 (曝光归一化后)  {percentile_text(column(rows, "input_tone_difference"), 2)}')
    lines.append('  b->a 全局模型的相对残差 (越小越能解释两次采集的差别):')
    for model_name, label in zip(MODEL_NAMES, ['整体增益', '逐通道增益', '逐通道增益+偏置', '3x3矩阵+偏置', '逐通道三次曲线']):
        lines.append(f'    {label:12s} {percentile_text(column(rows, f"residual_{model_name}"))}')
    for key, label in [('offset_improvement', '加偏置后残差下降'), ('cross_channel_improvement', '3x3 再下降'),
                       ('nonlinear_improvement', '三次曲线再下降')]:
        values = column(rows, key)
        finite = values[np.isfinite(values)]
        share = np.mean(finite > 0.2) if finite.size else np.nan
        lines.append(f'  {label:14s} {percentile_text(values, 2)}   下降超过 20% 的对: {share:.0%}')
    lines.append(f'  逐通道偏置 R/G/B (相对通道均值): {percentile_text(column(rows, "offset_red_relative"))} / '
                 f'{percentile_text(column(rows, "offset_green_relative"))} / {percentile_text(column(rows, "offset_blue_relative"))}')
    lines.append(f'  偏置的通道间差异          {percentile_text(column(rows, "offset_channel_spread"))}')
    lines.append(f'  增益的通道间差异          {percentile_text(column(rows, "gain_chroma_spread"))}')
    lines.append(f'  全局对齐后 低频亮度残差   {percentile_text(column(rows, "low_frequency_luminance_residual"))}')
    lines.append(f'  全局对齐后 低频色度残差   {percentile_text(column(rows, "low_frequency_chroma_residual"))}')
    lines.append(f'  全局对齐后 高频残差       {percentile_text(column(rows, "high_frequency_residual"))}')
    lines.append(f'  相对噪声之比 log(σb/σa)   {percentile_text(column(rows, "noise_log_ratio"), 2)}')
    lines.append(f'  压暗像素比例 较亮/较暗    {percentile_text(column(rows, "crushed_fraction_bright"), 2)} / '
                 f'{percentile_text(column(rows, "crushed_fraction_dark"), 2)}')


def summarize_model(rows, model_name, lines):
    prefix = f'{model_name}:'
    valid_rows = [row for row in rows if np.isfinite(row.get(prefix + 'output_tone_difference', np.nan))]
    lines.append(f'\n== 模型 {model_name} ({len(valid_rows)} 对有输出)')
    if len(valid_rows) < 3:
        lines.append('  对数太少, 跳过统计')
        return
    tone = column(valid_rows, prefix + 'output_tone_difference')
    lightness = column(valid_rows, prefix + 'output_delta_lightness')
    lines.append(f'  输出全局色调差   均值 {np.mean(tone):.2f}  中位数 {np.median(tone):.2f}  P90 {np.percentile(tone, 90):.2f}  >3 的对 {np.mean(tone > 3):.0%}')
    lines.append(f'  输出亮度差|ΔL*|  均值 {np.mean(np.abs(lightness)):.2f}  中位数 {np.median(np.abs(lightness)):.2f}  >3 的对 {np.mean(np.abs(lightness) > 3):.0%}')
    lines.append(f'  输出低频 ΔE00    均值 {np.mean(column(valid_rows, prefix + "output_lowpass_delta_e")):.2f}   '
                 f'逐像素 ΔE00 均值 {np.mean(column(valid_rows, prefix + "output_pixel_delta_e")):.2f}')

    # 亮度泄漏: 较亮的输入是否得到较亮的输出
    exposure = column(valid_rows, 'fitted_exposure_log2_ratio')
    lines.append(f'  亮度泄漏: 较亮输入的输出更亮的对 {np.mean(lightness > 0):.0%};  平均 ΔL* {np.mean(lightness):+.2f};  '
                 f'ΔL* 与曝光差的 Spearman r = {spearmanr(exposure[np.isfinite(exposure)], lightness[np.isfinite(exposure)])[0]:+.2f}')
    # 色偏传递: 输入的色调差 (曝光归一化后) 有多少进入了输出
    input_vector = np.concatenate([column(valid_rows, 'input_delta_a_star'), column(valid_rows, 'input_delta_b_star')])
    output_vector = np.concatenate([column(valid_rows, prefix + 'output_delta_a_star'), column(valid_rows, prefix + 'output_delta_b_star')])
    finite = np.isfinite(input_vector) & np.isfinite(output_vector)
    slope = np.sum(input_vector[finite] * output_vector[finite]) / max(np.sum(input_vector[finite] ** 2), 1e-12)
    correlation = np.corrcoef(input_vector[finite], output_vector[finite])[0, 1]
    lines.append(f'  真实成对的色偏传递: 输出 Δa*Δb* 对输入 Δa*Δb* 的过原点斜率 {slope:.2f} (1 = 完全照搬), 相关 r = {correlation:+.2f}')

    lines.append('  输出不一致与输入差异的 Spearman 相关 (|r| 越大越相关; * 表示 p<0.01):')
    header = '    ' + f'{"输入差异特征":38s}' + ''.join(f'{OUTPUT_TARGET_DESCRIPTIONS[target]:>16s}' for target in OUTPUT_TARGET_DESCRIPTIONS)
    lines.append(header)
    for feature, description in FEATURE_DESCRIPTIONS.items():
        feature_values = column(valid_rows, feature)
        cells = []
        for target in OUTPUT_TARGET_DESCRIPTIONS:
            target_values = column(valid_rows, prefix + target)
            finite = np.isfinite(feature_values) & np.isfinite(target_values)
            if finite.sum() < 5 or np.std(feature_values[finite]) == 0:
                cells.append(f'{"-":>16s}')
                continue
            rho, p_value = spearmanr(feature_values[finite], target_values[finite])
            cells.append(f'{rho:+15.2f}{"*" if p_value < 0.01 else " "}')
        lines.append(f'    {description:38s}' + ''.join(cells))

    lines.append('  控制曝光差 (拟合增益) 后的偏相关 (这几项随曝光一起变, 看它们是否有独立作用):')
    exposure = column(valid_rows, 'fitted_exposure_log2_ratio')
    for feature in EXPOSURE_LINKED_FEATURES:
        feature_values = column(valid_rows, feature)
        cells = []
        for target in OUTPUT_TARGET_DESCRIPTIONS:
            target_values = column(valid_rows, prefix + target)
            finite = np.isfinite(feature_values) & np.isfinite(target_values) & np.isfinite(exposure)
            cells.append(f'{partial_spearman(feature_values[finite], target_values[finite], exposure[finite]):+16.2f}'
                         if finite.sum() >= 5 else f'{"-":>16s}')
        lines.append(f'    {FEATURE_DESCRIPTIONS[feature]:38s}' + ''.join(cells))

    matrix = np.stack([column(valid_rows, name) for name in MULTIVARIATE_FEATURES], axis=1)
    finite_rows = np.all(np.isfinite(matrix), axis=1)
    if finite_rows.sum() >= 3 * len(MULTIVARIATE_FEATURES):
        lines.append(f'  多变量 ({len(MULTIVARIATE_FEATURES)} 个不重复特征, 岭回归, 留一法 R2; {int(finite_rows.sum())} 对):')
        for target, description in OUTPUT_TARGET_DESCRIPTIONS.items():
            target_values = column(valid_rows, prefix + target)
            usable = finite_rows & np.isfinite(target_values)
            ridge, r2, coefficients = best_leave_one_out_r2(matrix[usable], target_values[usable])
            top = np.argsort(-np.abs(coefficients))[:3]
            top_text = ', '.join(f'{FEATURE_DESCRIPTIONS[MULTIVARIATE_FEATURES[index]]} {coefficients[index]:+.2f}' for index in top)
            lines.append(f'    {description}: 留一法 R2 = {r2:.2f} (正则 {ridge});  标准化系数最大的三项: {top_text}')
    else:
        lines.append(f'  多变量回归跳过 (有效对 {int(finite_rows.sum())} 少于 {3 * len(MULTIVARIATE_FEATURES)})')

    lines.append('  输出色调差最大的 8 对 (a = 较亮输入):')
    for row in sorted(valid_rows, key=lambda row: -row[prefix + 'output_tone_difference'])[:8]:
        lines.append(f'    {row["bright_id"]}/{row["dark_id"]}  色调差 {row[prefix + "output_tone_difference"]:5.2f}  ΔL* {row[prefix + "output_delta_lightness"]:+5.1f} | '
                     f'曝光差 {row.get("fitted_exposure_log2_ratio", np.nan):.2f}  输入色调差 {row["input_tone_difference"]:.2f}  '
                     f'加偏置残差下降 {row.get("offset_improvement", np.nan):.2f}  低频色度残差 {row.get("low_frequency_chroma_residual", np.nan):.3f}  '
                     f'噪声比 {row["noise_log_ratio"]:+.2f}  压暗差 {row["crushed_fraction_difference"]:+.2f}')


def write_csv(rows, path):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f'{value:.6g}' if isinstance(value, float) else value) for key, value in row.items()})


def analyze_split(low_directory, ground_truth_directory, outputs, arguments, split_name, max_pairs=0):
    low_index, ground_truth_index = index_images(low_directory), index_images(ground_truth_directory)
    groups, single_count = group_by_ground_truth(low_index, ground_truth_index)
    pairs = [(group, first, second) for group in groups for first, second in itertools.combinations(group, 2)]
    if max_pairs and len(pairs) > max_pairs:
        selection = np.random.default_rng(0).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[index] for index in sorted(selection)]
    sizes = [len(group) for group in groups]
    group_text = (f'{split_name}: 低光图 {len(low_index)} 张, 有 GT 的 {len(set(low_index) & set(ground_truth_index))} 张; '
                  f'多采集场景 {len(groups)} 个 (组大小分布 ' + ', '.join(f'{size}张x{sizes.count(size)}' for size in sorted(set(sizes)))
                  + f'), 单张场景 {single_count} 个, 共 {len(pairs)} 对')
    print(group_text, flush=True)
    output_indexes = {name: index_images(path) for name, path in outputs.items()}
    rows, statistics_cache, start_time = [], {}, time.time()
    for pair_number, (group, first_id, second_id) in enumerate(pairs, 1):
        ground_truth_path = ground_truth_index[group[0]]
        row = {'group': '/'.join(group)}
        row.update(compute_pair_input_features(first_id, second_id, low_index, ground_truth_path, statistics_cache, arguments))
        ground_truth_srgb = load_srgb(ground_truth_path) if output_indexes else None
        for name, output_index in output_indexes.items():
            if row['bright_id'] in output_index and row['dark_id'] in output_index:
                metrics = compute_pair_output_metrics(row['bright_id'], row['dark_id'], output_index, ground_truth_srgb, arguments.block_size)
                row.update({f'{name}:{key}': value for key, value in metrics.items()})
        rows.append(row)
        if pair_number % 20 == 0 or pair_number == len(pairs):
            print(f'  {split_name} {pair_number}/{len(pairs)} 对, 用时 {time.time() - start_time:.0f}s', flush=True)
    return rows, group_text


READING_GUIDE = """读法:
  - "b->a 全局模型残差": 逐通道增益+偏置 明显好于 逐通道增益 (加偏置后残差下降 > 20% 的对多) -> 两次采集之间有逐通道偏置,
    只做增益/白平衡扰动不够, 扰动模型要加偏置; 三次曲线再下降多 -> 存在非线性 (色调曲线/截断); 3x3 再下降多 -> 存在通道串扰
  - "全局对齐后 低频色度残差" 大 -> 两次采集之间有空间上不均匀的色度差 (彩色斑块), 全局模型消不掉
  - "Spearman 相关" 表: 哪个输入差异特征与输出不一致相关最强, 就是方法首先要针对的干扰;
    多变量留一法 R2 表示这些输入差异合起来能解释多少输出不一致 (R2 低 -> 不一致更多来自模型本身的不稳定)
  - 噪声比、压暗比例、信噪比在物理上随曝光差一起变, 单看相关会和曝光差混在一起; "控制曝光差后的偏相关" 仍明显 -> 可能有独立作用.
    曝光差本身也是估计值, 偏相关只作参考; 两者都强时, 以多变量回归里谁的标准化系数更大为准
  - 曝光差优先看 "拟合增益" 那一项; "平均亮度差" 会被偏置抬高, 只作对照
  - "亮度泄漏": 较亮输入的输出也更亮的比例远高于 50% -> 模型没有把曝光完全归一化
  - "色偏传递": 斜率接近 1 -> 输入色偏直接带到输出; 接近 0 -> 输出色调差不是由输入的全局色偏造成
  - 测试集 "模型" 一节的输出全局色调差/亮度差, 应与本地 consistency_results.txt 一致 (Retinexformer 官方权重: 色调差均值 1.64, |ΔL*| 1.56)
"""


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.save_dir, exist_ok=True)
    outputs = {}
    for item in arguments.outputs:
        name, _, path = item.partition('=')
        if not path:
            raise SystemExit(f'--outputs 的格式应为 名字=路径, 收到: {item}')
        outputs[name] = path
    lines = ['同场景多次低光采集的差异诊断', READING_GUIDE]

    test_rows, test_group_text = analyze_split(arguments.test_low, arguments.test_gt, outputs, arguments, '测试集')
    lines.append('== 分组\n  ' + test_group_text)
    train_rows = []
    if arguments.train_low and arguments.train_gt:
        train_rows, train_group_text = analyze_split(arguments.train_low, arguments.train_gt, {}, arguments, '训练集',
                                                     arguments.max_train_pairs)
        lines.append('  ' + train_group_text)
    if test_rows:
        summarize_input_differences(test_rows, '测试集', lines)
    if train_rows:
        summarize_input_differences(train_rows, '训练集', lines)
    for name in outputs:
        summarize_model(test_rows, name, lines)

    write_csv(test_rows, os.path.join(arguments.save_dir, 'test_pairs.csv'))
    if train_rows:
        write_csv(train_rows, os.path.join(arguments.save_dir, 'train_pairs.csv'))
    summary = '\n'.join(lines)
    with open(os.path.join(arguments.save_dir, 'summary.txt'), 'w', encoding='utf-8') as handle:
        handle.write(summary + '\n')
    print(summary)


if __name__ == '__main__':
    main()
