"""[LLIE] 第 11 轮：输入预提亮。

低光图直接送进 SD 的 VAE 编码器时，像素几乎都挤在 [-1,1] 的 -0.9 附近，暗部几个灰阶的颜色差异只有约 0.02，
预训练编码器(只见过正常曝光的图)要靠 rank=4 的 LoRA 学出 10~30 倍增益，颜色细节容易丢。
这里在送入编码器之前，先用一个确定的、不含可学习参数的变换把输入提到正常曝光范围。

设计要点：
- 按**亮度**算增益，三个通道乘**同一个**比例 -> 色相和 R:G:B 比例不变(gamma 会让通道比例向 1 收缩，即变灰，所以不用)。
- 高光用软拐点压缩亮度，并限制比例不超过 1/max(R,G,B)，不会因为单通道截断而改变色相。
- 增益由**整张图**计算：训练时在裁剪之前做，推理时在整图上做，两边一致。
"""
import torch

BRIGHTEN_MODES = ("none", "gain", "fixed")

_LUMA = (0.299, 0.587, 0.114)


def _soft_knee(y, knee):
    """y<=knee 时不变；y>knee 时平滑压缩到 <1(拐点处一阶导数连续，单调递增)。"""
    if knee >= 1.0:
        return y.clamp(max=1.0)
    span = 1.0 - knee
    compressed = knee + span * (1.0 - torch.exp(-(y - knee) / span))
    return torch.where(y > knee, compressed, y)


@torch.no_grad()
def brighten_input(image_01, mode="gain", target=0.4, max_gain=20.0, fixed_gain=8.0, knee=0.8):
    """image_01: [C,H,W] 或 [B,C,H,W]，取值 [0,1]。返回同形状、同 dtype 的提亮结果。

    mode:
      none  : 原样返回
      gain  : 每张图一个增益 k = target / 平均亮度，限制在 [1, max_gain]；已经够亮的图(k<1)不变暗
      fixed : 所有图使用同一个增益 fixed_gain(保留不同图之间的相对亮度)
    """
    if mode == "none":
        return image_01
    if mode not in BRIGHTEN_MODES:
        raise ValueError(f"unknown brighten mode {mode!r}, expected one of {BRIGHTEN_MODES}")

    squeeze = image_01.dim() == 3
    x = image_01[None] if squeeze else image_01
    dtype = x.dtype
    x = x.float().clamp(0.0, 1.0)

    weights = torch.tensor(_LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    luma = (x * weights).sum(dim=1, keepdim=True)                       # [B,1,H,W]

    if mode == "gain":
        mean_luma = luma.mean(dim=(2, 3), keepdim=True)
        gain = (target / mean_luma.clamp_min(1e-4)).clamp(1.0, max_gain)  # [B,1,1,1]
    else:
        gain = torch.full((x.shape[0], 1, 1, 1), float(fixed_gain), device=x.device, dtype=x.dtype)

    new_luma = _soft_knee(luma * gain, knee)
    eps = 1e-4
    ratio = torch.where(luma > eps, new_luma / luma.clamp_min(eps), gain.expand_as(luma))
    # 不让任何通道超过 1：三个通道同比例缩放，色相不变
    max_channel = x.max(dim=1, keepdim=True).values
    ratio = torch.minimum(ratio, 1.0 / max_channel.clamp_min(eps))
    out = (x * ratio).clamp(0.0, 1.0).to(dtype)
    return out[0] if squeeze else out
