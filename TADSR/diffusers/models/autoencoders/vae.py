# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math  # [LLIE] 第 7 轮：低频颜色通路用
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # [LLIE] 第 7 轮：低频颜色通路用

from ..embeddings import Timesteps, TimestepEmbedding
from ...utils import BaseOutput, is_torch_version
from ...utils.torch_utils import randn_tensor
from ..activations import get_activation
from ..attention_processor import SpatialNorm
from ..unet_2d_blocks import (
    AutoencoderTinyBlock,
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class TimeAwareEncoder(nn.Module):
    r"""
    The `TimeAwareEncoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
        resnet_time_scale_shift (`str`, defaults to `"default"`)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        resnet_time_scale_shift: str = "scale_shift",
        temb_channels: int = 256,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        attention_head_dim: int = 512,
    ):
        super().__init__()
        if isinstance(layers_per_block, int):
            layers_per_block = (layers_per_block,) * len(down_block_types)
        self.layers_per_block = layers_per_block

        timestep_input_dim = max(128, block_out_channels[0])
        self.time_proj = Timesteps(timestep_input_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(timestep_input_dim, temb_channels)

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                temb_channels=temb_channels,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            attention_head_dim=attention_head_dim,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
            encoder_resnet=True 
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

        # [LLIE] time-aware skip：为 True 时，forward 会把每个 down block 的输入特征
        # (分辨率依次为 1, 1/2, 1/4, 1/8) 存到 self.current_skip_features，供解码器使用
        self.store_skip_features = False
        self.current_skip_features = None

    def forward(
            self,
            sample: torch.Tensor,
            timesteps: int,
            ) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        # time embedding
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=list(self.time_embedding.parameters())[0].dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # [LLIE] 收集 skip 特征(每个 down block 的输入)
        skip_features = [] if self.store_skip_features else None

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    if skip_features is not None:
                        skip_features.append(sample)
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, temb=emb, use_reentrant=False
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, temb=emb, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    if skip_features is not None:
                        skip_features.append(sample)
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, temb=emb
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, temb=emb
                )

        else:
            # down
            for down_block in self.down_blocks:
                if skip_features is not None:
                    skip_features.append(sample)
                sample = down_block(sample, temb=emb)

            # middle
            sample = self.mid_block(sample, temb=emb)

        self.current_skip_features = skip_features

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

        # [LLIE] time-aware skip。skip_convs 由 enable_skip_connections() 在加载预训练权重之后创建
        # (放在 __init__ 里会让 from_pretrained 因缺少权重而报错)。
        self.skip_convs = None
        # 每次解码前由外部设置：编码器特征列表(None 表示本次不使用 skip)和强度(标量或 [B,1,1,1])
        self.incoming_skip_features = None
        self.skip_gate = 1.0
        # [LLIE] 旁路特征是否先做逐通道归一化(见 _add_skip)；由 enable_skip_connections 设置
        self.skip_feature_norm = False
        # [LLIE] 第 7 轮：低频颜色通路。旁路特征拆成低频(颜色/曝光，不归一化、不乘 gate)和高频(细节，
        # 归一化 + gate(t))两部分。skip_convs_color 为低频部分的 1x1 卷积；color_pool_factor 为低通的
        # 等效下采样倍数(相对原图)；color_scale 由外部设置(训练时为 1，推理时等于 skip_scale)。
        self.skip_color_path = False
        self.skip_convs_color = None
        self.color_pool_factor = 32
        self.color_scale = 1.0

    def enable_skip_connections(self, block_out_channels, init_value: float = 1e-5, feature_norm: bool = False,
                                color_path: bool = False, color_pool_factor: int = 32):
        r"""[LLIE] 创建编码器->解码器的 1x1 skip 卷积。

        编码器第 i 个 down block 的输入特征(分辨率 1/2^i)接到解码器同分辨率 up block 的输入上。
        权重初始化为接近 0 的常数，训练开始时解码器的行为与没有 skip 时几乎一致。
        feature_norm=True 时，旁路特征先做逐通道归一化再进入 1x1 卷积。
        color_path=True 时，旁路按频率拆分：低频部分经单独的 1x1 卷积(skip_convs_color)直接接入，
        不归一化、不乘 gate(t)，负责把输入的颜色/曝光带过去；高频部分走原来的 skip_convs。
        """
        self.skip_feature_norm = bool(feature_norm)
        self.skip_color_path = bool(color_path)
        self.color_pool_factor = int(color_pool_factor)
        block_out_channels = list(block_out_channels)
        # 编码器各 down block 的输入通道数，按分辨率从高到低
        encoder_channels = [block_out_channels[0]] + block_out_channels[:-1]
        # 解码器各 up block 的输入通道数，按分辨率从低到高
        reversed_channels = list(reversed(block_out_channels))
        decoder_channels = [reversed_channels[0]] + reversed_channels[:-1]

        skip_convs = []
        for encoder_channel, decoder_channel in zip(reversed(encoder_channels), decoder_channels):
            skip_conv = nn.Conv2d(encoder_channel, decoder_channel, kernel_size=1, bias=False)
            nn.init.constant_(skip_conv.weight, init_value)
            skip_convs.append(skip_conv)
        self.skip_convs = nn.ModuleList(skip_convs)
        reference_parameter = next(self.conv_in.parameters())
        self.skip_convs.to(device=reference_parameter.device, dtype=reference_parameter.dtype)

        if self.skip_color_path:
            # [LLIE] 第 7 轮：低频颜色通路的 1x1 卷积，通道对应关系与 skip_convs 相同。
            # 参数名含 "skip_conv"，所以训练、保存、加载时会和 skip_convs 一起处理。
            color_convs = []
            for encoder_channel, decoder_channel in zip(reversed(encoder_channels), decoder_channels):
                color_conv = nn.Conv2d(encoder_channel, decoder_channel, kernel_size=1, bias=False)
                nn.init.constant_(color_conv.weight, init_value)
                color_convs.append(color_conv)
            self.skip_convs_color = nn.ModuleList(color_convs)
            self.skip_convs_color.to(device=reference_parameter.device, dtype=reference_parameter.dtype)

    def _lowpass(self, feature: torch.Tensor, block_index: int) -> torch.Tensor:
        # [LLIE] 低通：平均池化到约 原图/color_pool_factor 的分辨率，再双线性放大回原尺寸。
        # 第 block_index 个 up block 处的特征分辨率为 原图/2^(n-1-block_index)，据此换算这一层的池化倍数。
        level_downsample = 2 ** (len(self.up_blocks) - 1 - block_index)
        pool = max(1, self.color_pool_factor // level_downsample)
        if pool == 1:
            return feature
        height, width = feature.shape[-2:]
        pooled = F.adaptive_avg_pool2d(feature, (max(1, math.ceil(height / pool)), max(1, math.ceil(width / pool))))
        return F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)

    def _add_skip(self, sample: torch.Tensor, block_index: int) -> torch.Tensor:
        # [LLIE] 在第 block_index 个 up block 之前加上对应分辨率的编码器特征
        if self.skip_convs is None or self.incoming_skip_features is None:
            return sample
        feature = self.incoming_skip_features[-1 - block_index]
        if feature.shape[-2:] != sample.shape[-2:]:
            raise ValueError(
                f"Skip feature size {tuple(feature.shape[-2:])} does not match decoder feature size "
                f"{tuple(sample.shape[-2:])}. The input height and width must be multiples of 8."
            )
        color_term = None
        if self.skip_color_path:
            # [LLIE] 第 7 轮：频率拆分。低频(颜色/曝光)不归一化、不乘 gate(t)，任何 t 下都把输入的颜色带过去；
            # 平均还能压低暗部色度的量化和噪声。高频 = 原特征 - 低频，继续走下面的归一化 + gate(t)。
            feature_float = feature.float()
            low_frequency = self._lowpass(feature_float, block_index)
            color_term = self.skip_convs_color[block_index](low_frequency.to(sample.dtype))
            feature = feature_float - low_frequency
        if self.skip_feature_norm:
            # [LLIE] 逐通道、在空间维度上标准化(无可学习参数的 instance norm)：
            # 1) 特征幅度固定，编码器无法通过放大特征来抵消 gate(t)，t 才能真正控制旁路强度；
            # 2) 去掉每个通道的均值，整体亮度/颜色无法经旁路从暗图直接传过去，只能由 latent 提供，
            #    旁路只负责结构和细节。用 float32 计算，避免 fp16 下方差下溢。
            feature_float = feature.float()
            mean = feature_float.mean(dim=(2, 3), keepdim=True)
            variance = feature_float.var(dim=(2, 3), keepdim=True, unbiased=False)
            feature = (feature_float - mean) * torch.rsqrt(variance + 1e-5)
        skip = self.skip_convs[block_index](feature.to(sample.dtype))
        gate = self.skip_gate
        if torch.is_tensor(gate):
            gate = gate.to(device=sample.device, dtype=sample.dtype)
        sample = sample + skip * gate
        if color_term is not None:
            sample = sample + color_term * self.color_scale
        return sample

    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for block_index, up_block in enumerate(self.up_blocks):
                    sample = self._add_skip(sample, block_index)
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for block_index, up_block in enumerate(self.up_blocks):
                    sample = self._add_skip(sample, block_index)
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for block_index, up_block in enumerate(self.up_blocks):
                sample = self._add_skip(sample, block_index)
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class UpSample(nn.Module):
    r"""
    The `UpSample` layer of a variational autoencoder that upsamples its input.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `UpSample` class."""
        x = torch.relu(x)
        x = self.deconv(x)
        return x


class MaskConditionEncoder(nn.Module):
    """
    used in AsymmetricAutoencoderKL
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 192,
        res_ch: int = 768,
        stride: int = 16,
    ) -> None:
        super().__init__()

        channels = []
        while stride > 1:
            stride = stride // 2
            in_ch_ = out_ch * 2
            if out_ch > res_ch:
                out_ch = res_ch
            if stride == 1:
                in_ch_ = res_ch
            channels.append((in_ch_, out_ch))
            out_ch *= 2

        out_channels = []
        for _in_ch, _out_ch in channels:
            out_channels.append(_out_ch)
        out_channels.append(channels[-1][0])

        layers = []
        in_ch_ = in_ch
        for l in range(len(out_channels)):
            out_ch_ = out_channels[l]
            if l == 0 or l == 1:
                layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
            in_ch_ = out_ch_

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor, mask=None) -> torch.FloatTensor:
        r"""The forward method of the `MaskConditionEncoder` class."""
        out = {}
        for l in range(len(self.layers)):
            layer = self.layers[l]
            x = layer(x)
            out[str(tuple(x.shape))] = x
            x = torch.relu(x)
        return out


class MaskConditionDecoder(nn.Module):
    r"""The `MaskConditionDecoder` should be used in combination with [`AsymmetricAutoencoderKL`] to enhance the model's
    decoder with a conditioner on the mask and masked image.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # condition encoder
        self.condition_encoder = MaskConditionEncoder(
            in_ch=out_channels,
            out_ch=block_out_channels[0],
            res_ch=block_out_channels[-1],
        )

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        z: torch.FloatTensor,
        image: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `MaskConditionDecoder` class."""
        sample = z
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # condition encoder
                if image is not None and mask is not None:
                    masked_image = (1 - mask) * image
                    im_x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.condition_encoder),
                        masked_image,
                        mask,
                        use_reentrant=False,
                    )

                # up
                for up_block in self.up_blocks:
                    if image is not None and mask is not None:
                        sample_ = im_x[str(tuple(sample.shape))]
                        mask_ = nn.functional.interpolate(mask, size=sample.shape[-2:], mode="nearest")
                        sample = sample * mask_ + sample_ * (1 - mask_)
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
                if image is not None and mask is not None:
                    sample = sample * mask + im_x[str(tuple(sample.shape))] * (1 - mask)
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # condition encoder
                if image is not None and mask is not None:
                    masked_image = (1 - mask) * image
                    im_x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.condition_encoder),
                        masked_image,
                        mask,
                    )

                # up
                for up_block in self.up_blocks:
                    if image is not None and mask is not None:
                        sample_ = im_x[str(tuple(sample.shape))]
                        mask_ = nn.functional.interpolate(mask, size=sample.shape[-2:], mode="nearest")
                        sample = sample * mask_ + sample_ * (1 - mask_)
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
                if image is not None and mask is not None:
                    sample = sample * mask + im_x[str(tuple(sample.shape))] * (1 - mask)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # condition encoder
            if image is not None and mask is not None:
                masked_image = (1 - mask) * image
                im_x = self.condition_encoder(masked_image, mask)

            # up
            for up_block in self.up_blocks:
                if image is not None and mask is not None:
                    sample_ = im_x[str(tuple(sample.shape))]
                    mask_ = nn.functional.interpolate(mask, size=sample.shape[-2:], mode="nearest")
                    sample = sample * mask_ + sample_ * (1 - mask_)
                sample = up_block(sample, latent_embeds)
            if image is not None and mask is not None:
                sample = sample * mask + im_x[str(tuple(sample.shape))] * (1 - mask)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e: int,
        vq_embed_dim: int,
        beta: float,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple]:
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q: torch.FloatTensor = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.FloatTensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q: torch.FloatTensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class EncoderTiny(nn.Module):
    r"""
    The `EncoderTiny` layer is a simpler version of the `Encoder` layer.

    Args:
        in_channels (`int`):
            The number of input channels.
        out_channels (`int`):
            The number of output channels.
        num_blocks (`Tuple[int, ...]`):
            Each value of the tuple represents a Conv2d layer followed by `value` number of `AutoencoderTinyBlock`'s to
            use.
        block_out_channels (`Tuple[int, ...]`):
            The number of output channels for each block.
        act_fn (`str`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        act_fn: str,
    ):
        super().__init__()

        layers = []
        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]

            if i == 0:
                layers.append(nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
            else:
                layers.append(
                    nn.Conv2d(
                        num_channels,
                        num_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                )

            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

        layers.append(nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `EncoderTiny` class."""
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)

        else:
            # scale image from [-1, 1] to [0, 1] to match TAESD convention
            x = self.layers(x.add(1).div(2))

        return x


class DecoderTiny(nn.Module):
    r"""
    The `DecoderTiny` layer is a simpler version of the `Decoder` layer.

    Args:
        in_channels (`int`):
            The number of input channels.
        out_channels (`int`):
            The number of output channels.
        num_blocks (`Tuple[int, ...]`):
            Each value of the tuple represents a Conv2d layer followed by `value` number of `AutoencoderTinyBlock`'s to
            use.
        block_out_channels (`Tuple[int, ...]`):
            The number of output channels for each block.
        upsampling_scaling_factor (`int`):
            The scaling factor to use for upsampling.
        act_fn (`str`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
            get_activation(act_fn),
        ]

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]

            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor))

            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(
                nn.Conv2d(
                    num_channels,
                    conv_out_channel,
                    kernel_size=3,
                    padding=1,
                    bias=is_final_block,
                )
            )

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `DecoderTiny` class."""
        # Clamp.
        x = torch.tanh(x / 3) * 3

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)

        else:
            x = self.layers(x)

        # scale image from [0, 1] to [-1, 1] to match diffusers convention
        return x.mul(2).sub(1)
