import glob
import os

import torch
from PIL import Image
import random
import numpy as np

from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

from .realesrgan import RealESRGAN_degradation
from my_utils.llie_brighten import brighten_input  # [LLIE] 第 11 轮：输入预提亮

class PairedCaptionDataset(data.Dataset):
    def __init__(self, split=None, args=None, txt_path=None):
        super(PairedCaptionDataset, self).__init__()
        self.args = args
        self.split = split

        # [LLIE] 可显式传入列表文件(验证集用)；不传则沿用 --txt_path 的第一个
        if txt_path is None:
            txt_path = self.args.txt_path[0]

        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []

        # [LLIE] 每行格式: GT路径  LQ路径  tag
        # 优先按制表符切分(路径中允许有空格)，否则按空白切分且最多切2次(tag里的空格保留)
        with open(txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                if '\t' in line:
                    parts = line.split('\t', 2)
                else:
                    parts = line.strip().split(maxsplit=2)

                if len(parts) >= 2:
                    self.gt_list.append(parts[0].strip())
                    self.lr_list.append(parts[1].strip())
                    self.tag_path_list.append(parts[2].strip() if len(parts) == 3 else "")
                else:
                    print(f"skip: {line.strip()}")

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)
        indices = list(range(len(self.lr_list)))
        # [LLIE] 只打乱训练集；验证集保持列表顺序，保证每次验证用的是同一批图
        if self.split == "train":
            random.shuffle(indices)
        self.lr_list = [self.lr_list[i] for i in indices]
        self.gt_list = [self.gt_list[i] for i in indices]
        self.tag_path_list = [self.tag_path_list[i] for i in indices]

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
        # [LLIE] 训练裁剪尺寸，需为8的倍数(VAE下采样8倍)
        self.crop_size = int(getattr(self.args, "resolution", 384))
        assert self.crop_size % 8 == 0, "--resolution must be a multiple of 8"

    def _paired_random_crop(self, gt_img, lq_img):
        # [LLIE] 对 GT/LQ 使用同一个裁剪位置；短边不足时先等比放大
        _, h, w = gt_img.shape
        cs = self.crop_size
        if min(h, w) < cs:
            scale = cs / min(h, w)
            new_h, new_w = max(cs, round(h * scale)), max(cs, round(w * scale))
            gt_img = F.interpolate(gt_img[None], size=(new_h, new_w), mode="bicubic", align_corners=False)[0].clamp(0, 1)
            lq_img = F.interpolate(lq_img[None], size=(new_h, new_w), mode="bicubic", align_corners=False)[0].clamp(0, 1)
            h, w = new_h, new_w
        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        gt_img = gt_img[:, top:top + cs, left:left + cs]
        lq_img = lq_img[:, top:top + cs, left:left + cs]
        return gt_img, lq_img

    def _paired_augment(self, gt_img, lq_img):
        # [LLIE] 水平/垂直翻转，不改变亮度与颜色
        if random.random() < 0.5:
            gt_img, lq_img = gt_img.flip(-1), lq_img.flip(-1)
        if random.random() < 0.5:
            gt_img, lq_img = gt_img.flip(-2), lq_img.flip(-2)
        return gt_img, lq_img

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')

        # [LLIE] 始终使用配对的低光图作为输入；删除原 SR 的 bicubic x4 退化分支(deg_prob 不再使用)
        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        if lq_img.size != gt_img.size:
            lq_img = lq_img.resize(gt_img.size, Image.Resampling.LANCZOS)

        gt_img = self.img_preproc(gt_img)
        lq_img = self.img_preproc(lq_img)

        # [LLIE] 第 11 轮：在**整图**上预提亮低光输入(在裁剪之前，增益与推理时整图计算的一致)；GT 不动
        lq_img = brighten_input(
            lq_img,
            mode=getattr(self.args, "input_brighten", "none"),
            target=float(getattr(self.args, "input_brighten_target", 0.4)),
            max_gain=float(getattr(self.args, "input_brighten_max_gain", 20.0)),
            fixed_gain=float(getattr(self.args, "input_brighten_fixed_gain", 8.0)),
            knee=float(getattr(self.args, "input_brighten_knee", 0.8)),
        )

        if self.split == "train":
            gt_img, lq_img = self._paired_random_crop(gt_img, lq_img)
            gt_img, lq_img = self._paired_augment(gt_img, lq_img)

        tag = self.tag_path_list[index]

        example = dict()

        example["neg_prompt"] = self.args.neg_prompt
        example["null_prompt"] = ""
        example["prompt"] = tag
        example["output_pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["conditioning_pixel_values"] = lq_img.squeeze(0) * 2.0 - 1.0


        return example

    def __len__(self):
        return len(self.gt_list)
