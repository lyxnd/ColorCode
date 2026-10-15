# SID (Sony .ARW) 短曝光 raw -> sRGB PNG, 避免"负噪声截断 x 白平衡增益"导致的品红偏色
# 依赖: pip install rawpy numpy pillow
# 用法:
#   python sid_raw_to_srgb.py short.ARW out.png --wb_from long.ARW              # 保持真实暗度 (更接近 LOL 低光输入的分布)
#   python sid_raw_to_srgb.py short.ARW out.png --wb_from long.ARW --ratio 300  # 按曝光比放大 (0.1s->30s 为300)
#   --target_mean 0.04 : 把线性亮度均值缩放到指定值 (可选, 用来对齐 LOL-v2-real low 的亮度水平)
#   --size 960 512     : 最后 resize 到指定宽高 (与你之前的数据保持一致时使用)
# 说明: 未在你的 raw 文件上实际运行过, 第一次用请先对 1 张图检查通道均值。
import argparse, numpy as np, rawpy
from PIL import Image

def pack_half(raw):
    im = raw.raw_image_visible.astype(np.float32)
    bl = np.array(raw.black_level_per_channel, np.float32)
    white = float(raw.white_level)
    colors = raw.raw_colors_visible            # 每个像素的 CFA 颜色索引 (0=R,1=G,2=B,3=G2)
    out = np.zeros((im.shape[0]//2, im.shape[1]//2, 3), np.float32)
    cnt = np.zeros_like(out)
    for dy in (0, 1):
        for dx in (0, 1):
            ch = colors[dy, dx]
            v = (im[dy::2, dx::2] - bl[ch]) / (white - bl[ch])     # 注意: 这里不截断负值
            c = {0: 0, 1: 1, 3: 1, 2: 2}[ch]
            h, w = out.shape[:2]
            out[..., c] += v[:h, :w]; cnt[..., c] += 1
    return out / np.maximum(cnt, 1)

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--wb_from', default=None, help='用长曝光 raw 的相机白平衡 (更稳定)')
ap.add_argument('--ratio', type=float, default=1.0)
ap.add_argument('--target_mean', type=float, default=None)
ap.add_argument('--size', type=int, nargs=2, default=None)
a = ap.parse_args()

with rawpy.imread(a.src) as raw:
    x = pack_half(raw)
    rgb_cam = np.array(raw.color_matrix, np.float32)[:, :3]   # camera RGB -> sRGB(线性)
    wb = np.array((rawpy.imread(a.wb_from) if a.wb_from else raw).camera_whitebalance, np.float32)
wb = wb[:3] / wb[1]
x = x * wb * a.ratio                    # 白平衡和放大都在"未截断"的线性数据上做, 噪声均值仍为0
x = x @ rgb_cam.T
if a.target_mean: x = x * (a.target_mean / max(x.mean(), 1e-6))
x = np.clip(x, 0, 1)                    # 只在最后截断一次
x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
img = Image.fromarray((x * 255 + 0.5).astype(np.uint8))
if a.size: img = img.resize(tuple(a.size), Image.BICUBIC)
img.save(a.dst)
print('saved', a.dst, 'mean RGB (8bit) =', np.asarray(img).reshape(-1, 3).mean(0).round(1))
