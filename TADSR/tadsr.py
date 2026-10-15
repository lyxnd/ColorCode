import os
import sys
sys.path.append(os.getcwd())
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
from diffusers import TimeAwareAutoencoderKL

from my_utils.vaehook import VAEHook, perfcount
from my_utils.llie_brighten import brighten_input  # [LLIE] 第 11 轮：输入预提亮


def get_brighten_config(source):
    """[LLIE] 第 11 轮：从训练参数(args)或权重文件(dict)里取预提亮设置；旧权重没有这些键，按 none 处理。"""
    getter = source.get if isinstance(source, dict) else (lambda key, default: getattr(source, key, default))
    return {
        "mode": getter("input_brighten", "none"),
        "target": float(getter("input_brighten_target", 0.4)),
        "max_gain": float(getter("input_brighten_max_gain", 20.0)),
        "fixed_gain": float(getter("input_brighten_fixed_gain", 8.0)),
        "knee": float(getter("input_brighten_knee", 0.8)),
    }


# ---------------------------------------------------------------------------------------------
# [LLIE] time-aware skip：编码器 -> 解码器的旁路，用来绕过 VAE latent 的 8 倍压缩瓶颈，保留文字和细节。
# 旁路强度随 t 递减：小 t(保真)时全开，大 t(生成)时减弱，与 t 原有的语义保持一致。
# ---------------------------------------------------------------------------------------------
SKIP_GATE_MODES = ("sqrt_alpha", "linear", "none")


def compute_skip_gate(mode, timesteps, alphas_cumprod, num_train_timesteps=1000):
    """返回 skip 强度。

    sqrt_alpha: sqrt(alpha_cumprod[t])，即扩散过程在时间步 t 保留的信号比例(t=1 约 1.0，t=999 约 0.07)
    linear:     1 - t / num_train_timesteps
    none:       恒为 1(普通的 skip，不随 t 变化，用作对照)
    结果形状为 [B, 1, 1, 1]，mode 为 none 时返回标量 1.0。
    """
    if mode == "none":
        return 1.0
    timesteps = torch.as_tensor(timesteps, device=alphas_cumprod.device).long().reshape(-1)
    if mode == "sqrt_alpha":
        gate = alphas_cumprod[timesteps].float() ** 0.5
    elif mode == "linear":
        gate = 1.0 - timesteps.float() / num_train_timesteps
    else:
        raise ValueError(f"Unknown skip gate mode: {mode}. Choose from {SKIP_GATE_MODES}.")
    return gate.view(-1, 1, 1, 1)


def get_vae_decoder_lora_modules(vae):
    """解码器中加 LoRA 的模块名(卷积和注意力投影)，不含 skip 卷积(skip 卷积整体训练)。"""
    patterns = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    module_names = []
    for name, _ in vae.named_parameters():
        if not name.startswith("decoder."):
            continue
        if "bias" in name or "norm" in name or "lora" in name or "skip_conv" in name:
            continue
        if any(pattern in name for pattern in patterns):
            module_name = name.replace(".weight", "")
            if module_name not in module_names:
                module_names.append(module_name)
    return module_names


def enable_vae_skip(vae, lora_rank, decoder_lora_modules=None, feature_norm=False, color_path=False, color_pool_factor=32):
    """给已加载好权重、且已有 default_encoder LoRA 的 VAE 加上 skip 卷积和解码器 LoRA。
    feature_norm: 旁路特征是否先做逐通道归一化([LLIE] 第 4 轮的 B)。
    color_path / color_pool_factor: 旁路是否按频率拆分出低频颜色通路、低通的等效下采样倍数([LLIE] 第 7 轮)。"""
    vae.decoder.enable_skip_connections(vae.config.block_out_channels, feature_norm=feature_norm,
                                        color_path=color_path, color_pool_factor=color_pool_factor)
    vae.encoder.store_skip_features = True
    if decoder_lora_modules is None:
        decoder_lora_modules = get_vae_decoder_lora_modules(vae)
    lora_conf_decoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian", target_modules=decoder_lora_modules)
    vae.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    # add_adapter 只会激活最后加入的 adapter，必须显式把两个都激活，否则编码器 LoRA 会失效
    vae.set_adapter(["default_encoder", "default_decoder"])
    return decoder_lora_modules


def initialize_vae_time(args):
    vae = TimeAwareAutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="time_vae")
    vae.requires_grad_(False)
    vae.train()

    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    l_time = "time"
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n) and (l_time not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))

    for name, param in vae.named_parameters():
        if l_time in name:
            param.requires_grad = True

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    # [LLIE] time-aware skip + 解码器 LoRA
    l_target_modules_decoder = []
    if getattr(args, "use_skip", False):
        l_target_modules_decoder = enable_vae_skip(vae, args.lora_rank,
                                                   feature_norm=getattr(args, "skip_feature_norm", False),
                                                   color_path=getattr(args, "skip_color_path", False),
                                                   color_pool_factor=getattr(args, "skip_color_pool", 32))
        for name, param in vae.named_parameters():
            if "lora" in name or "skip_conv" in name or l_time in name:
                param.requires_grad = True

    return vae, l_target_modules_encoder, l_target_modules_decoder

def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = args.pretrained_model_name_or_path
    else:
        print(f"pretrained_model_name_or_path: {pretrained_model_name_or_path}")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others



class TADSR_gen(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        # [LLIE] time-aware skip 的开关和强度模式
        self.use_skip = bool(getattr(args, "use_skip", False))
        self.skip_gate_mode = getattr(args, "skip_gate", "sqrt_alpha")
        # [LLIE] 第 4 轮：B 旁路特征归一化；A 训练时额外做一次不走旁路的解码，用于辅助重建损失
        self.skip_feature_norm = self.use_skip and bool(getattr(args, "skip_feature_norm", False))
        self.noskip_reconstruction = self.use_skip and float(getattr(args, "lambda_noskip", 0.0)) > 0
        # [LLIE] 第 7 轮：低频颜色通路
        self.skip_color_path = self.use_skip and bool(getattr(args, "skip_color_path", False))
        self.skip_color_pool = int(getattr(args, "skip_color_pool", 32))
        # [LLIE] 第 11 轮：输入预提亮的设置。提亮本身在数据集里(整图、裁剪前)完成，这里只记录下来存进权重，
        # 推理时 TADSR_test 按同样的设置处理整图
        self.brighten_config = get_brighten_config(args)
        self.vae, self.lora_vae_modules_encoder, self.lora_vae_modules_decoder = initialize_vae_time(self.args)


        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args)
        self.load_pretrained_model()
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([1], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def load_pretrained_model(self):
        """加载预训练的LoRA权重
        
        Args:
            pretrained_model_name_or_path (str): 预训练模型的路径
        """
        if hasattr(self.args, 'pretrained_lora_path') and self.args.pretrained_lora_path is not None:
            print(f"Loading pretrained LoRA weights from {self.args.pretrained_lora_path}")
            
            # 加载预训练权重
            pretrained_weights = torch.load(self.args.pretrained_lora_path, map_location="cuda")
            
            # 加载UNet的LoRA权重
            if "state_dict_unet" in pretrained_weights:
                print("Loading UNet LoRA weights...")
                for n, p in self.unet.named_parameters():
                    if "lora" in n and n in pretrained_weights["state_dict_unet"]:
                        p.data.copy_(pretrained_weights["state_dict_unet"][n])
                        print(f"Loaded UNet weight: {n}")
                
                # 加载conv_in权重
                if "conv_in.weight" in pretrained_weights["state_dict_unet"]:
                    self.unet.conv_in.weight.data.copy_(pretrained_weights["state_dict_unet"]["conv_in.weight"])
                    print("Loaded UNet conv_in.weight")
                if "conv_in.bias" in pretrained_weights["state_dict_unet"]:
                    self.unet.conv_in.bias.data.copy_(pretrained_weights["state_dict_unet"]["conv_in.bias"])
                    print("Loaded UNet conv_in.bias")
            
            # 加载VAE的LoRA权重
            if "state_dict_vae" in pretrained_weights:
                print("Loading VAE LoRA weights...")
                for n, p in self.vae.named_parameters():
                    if "lora" in n and n in pretrained_weights["state_dict_vae"]:
                        p.data.copy_(pretrained_weights["state_dict_vae"][n])
                        print(f"Loaded VAE weight: {n}")
            
            # 加载VAE的time权重（如果存在）
            if "state_dict_vae_time" in pretrained_weights:
                print("Loading VAE time weights...")
                for n, p in self.vae.named_parameters():
                    if "time" in n and n in pretrained_weights["state_dict_vae_time"]:
                        p.data.copy_(pretrained_weights["state_dict_vae_time"][n])
                        print(f"Loaded VAE time weight: {n}")

            # [LLIE] 加载 skip 卷积权重(预训练权重带 skip 且当前也启用 skip 时)
            if self.use_skip and "state_dict_vae_skip" in pretrained_weights:
                print("Loading VAE skip weights...")
                for n, p in self.vae.named_parameters():
                    if "skip_conv" in n and n in pretrained_weights["state_dict_vae_skip"]:
                        p.data.copy_(pretrained_weights["state_dict_vae_skip"][n])
                        print(f"Loaded VAE skip weight: {n}")

            print("Successfully loaded pretrained LoRA weights")
        else:
            print("No pretrained LoRA path specified, using random initialization")

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n or "time" in n or "skip_conv" in n:  # [LLIE] skip 卷积也参与训练
                _p.requires_grad = True

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def normal_latent(self, latent):
        mean = latent.mean(dim=(2, 3), keepdim=True)
        std = latent.std(dim=(2, 3), keepdim=True)
        latent = (latent - mean) / (std + 1e-8)
        return latent

    def shift_latent(self, latent, target_latent):
        latent = latent * target_latent.std(dim=(2, 3), keepdim=True) + target_latent.mean(dim=(2, 3), keepdim=True)
        return latent

    def get_x0_from_res(self, latent_lq, model_pred, timesteps):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=latent_lq.device, dtype=latent_lq.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(latent_lq.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        x0 = (latent_lq / alpha_prod_t ** (0.5)) - model_pred
        return x0

    def forward(self, c_t, batch=None, timesteps=None):

        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])

        # [LLIE] 第 9 轮：训练时照旧从 VAE 后验采样；eval 模式(训练中的验证)改用后验均值，结果确定、与推理默认一致
        latent_distribution = self.vae.encode(c_t, timesteps).latent_dist
        encoded_control = (latent_distribution.sample() if self.training else latent_distribution.mode()) * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
        x_denoised = self.get_x0_from_res(encoded_control, model_pred, timesteps)

        # [LLIE] 把编码器特征交给解码器，强度随 t 变化；解码后立即清空，避免被之后的 decode 误用
        if self.use_skip:
            self.vae.decoder.incoming_skip_features = self.vae.encoder.current_skip_features
            self.vae.decoder.skip_gate = compute_skip_gate(
                self.skip_gate_mode, timesteps, self.noise_scheduler.alphas_cumprod,
                self.noise_scheduler.config.num_train_timesteps)
            self.vae.decoder.color_scale = 1.0  # [LLIE] 第 7 轮：低频颜色通路不随 t 变化
        try:
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        finally:
            self.vae.decoder.incoming_skip_features = None
            self.vae.encoder.current_skip_features = None

        # [LLIE] A：训练时把同一个 latent 关掉旁路再解码一次(解码器 LoRA 共用)。
        # 对这次输出也算重建损失，迫使 latent 本身带有完整图像信息，旁路只在此基础上补细节，
        # 防止模型把内容全部走旁路、架空扩散部分。验证/推理(eval 模式)时不计算，返回 None。
        output_image_noskip = None
        if self.noskip_reconstruction and self.training:
            output_image_noskip = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds, output_image_noskip

    def save_model(self, outf):
        sd = {}
        sd["state_dict_vae_time"] = {k: v for k, v in self.vae.state_dict().items() if "time" in k}

        # [LLIE] skip 相关：开关、强度模式、解码器 LoRA 模块名、skip 卷积权重
        # (解码器 LoRA 的权重名里带 "lora"，已包含在下面的 state_dict_vae 中)
        sd["use_skip"] = self.use_skip
        sd["skip_gate"] = self.skip_gate_mode
        sd["skip_feature_norm"] = self.skip_feature_norm  # [LLIE] 第 4 轮
        sd["skip_color_path"] = self.skip_color_path      # [LLIE] 第 7 轮
        sd["skip_color_pool"] = self.skip_color_pool
        # [LLIE] 第 11 轮：输入预提亮设置
        sd["input_brighten"] = self.brighten_config["mode"]
        sd["input_brighten_target"] = self.brighten_config["target"]
        sd["input_brighten_max_gain"] = self.brighten_config["max_gain"]
        sd["input_brighten_fixed_gain"] = self.brighten_config["fixed_gain"]
        sd["input_brighten_knee"] = self.brighten_config["knee"]
        sd["vae_lora_decoder_modules"] = self.lora_vae_modules_decoder
        sd["state_dict_vae_skip"] = {k: v for k, v in self.vae.state_dict().items() if "skip_conv" in k}

        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        torch.save(sd, outf)


class TADSR_reg(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(args, pretrained_model_name_or_path=args.pretrained_model_name_or_path_reg)
        self.unet_update.to(accelerator.device)


    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def diff_loss(self, latents, prompt_embeds, args):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        self.unet_update.enable_adapters()
        noise_pred = self.unet_update(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample
    
    
    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args, model_timesteps=None):
        bsz = latents.shape[0]
        if model_timesteps is not None:
            peak = model_timesteps * 0.5 + 0 
            timesteps = peak.long().to(latents.device)
        else:
            timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
         
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            self.unet_update.enable_adapters()
            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            self.unet_update.disable_adapters()
            noise_pred_fix = self.unet_update(
                noisy_latents_input,
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)

            cfg_vsd = torch.tensor(args.cfg_vsd, device=latents.device) + timesteps / 200
            cfg_vsd = cfg_vsd.view(bsz, 1, 1, 1)
            noise_pred_fix = noise_pred_uncond + cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss
    

    
class TADSR_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")

        self.vae = TimeAwareAutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="time_vae")   
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")


        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        tadsr_weight = torch.load(args.tadsr_path)
        self.load_ckpt(tadsr_weight)

        # [LLIE] 启用 skip 时，编码器必须走原始 forward 才能拿到各层特征。
        # 分块 VAE 的钩子会让编码器始终走它自己的任务队列(不经过 forward)，所以这里把编码器的钩子去掉；
        # 代价是编码器不再分块，特别大的图会更占显存。解码器的钩子本来就总是调用原始 forward，不受影响。
        self.skip_scale = float(getattr(args, "skip_scale", 1.0))
        # [LLIE] 第 9 轮：VAE 编码取后验均值(mean，默认，结果确定)还是采样(sample，原 TADSR 的做法，每次运行会有随机波动)
        self.latent_mode = getattr(args, "latent_mode", "mean")
        if self.latent_mode not in ("mean", "sample"):
            raise ValueError(f"latent_mode must be 'mean' or 'sample', got {self.latent_mode}")
        if self.use_skip:
            self.vae.encoder.forward = self.vae.encoder.original_forward
            print(f"===> time-aware skip enabled (gate: {self.skip_gate_mode}, feature norm: {self.skip_feature_norm}, "
                  f"color path: {self.skip_color_path}, scale: {self.skip_scale}); VAE encoder tiling is disabled <===")
        print(f"===> input brighten: {self.brighten_config} <===")  # [LLIE] 第 11 轮


        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")

        # [LLIE] 权重里带 skip 时，重建 skip 卷积和解码器 LoRA(旧权重没有这些键，按不带 skip 处理)
        self.use_skip = bool(model.get("use_skip", False))
        self.skip_gate_mode = model.get("skip_gate", "none")
        # [LLIE] 第 4 轮：旁路特征归一化开关(第 3 轮的权重没有这个键，按关闭处理)
        self.skip_feature_norm = bool(model.get("skip_feature_norm", False))
        # [LLIE] 第 7 轮：低频颜色通路(之前的权重没有这些键，按关闭处理)
        self.skip_color_path = bool(model.get("skip_color_path", False))
        self.skip_color_pool = int(model.get("skip_color_pool", 32))
        # [LLIE] 第 11 轮：输入预提亮(之前的权重没有这些键，按 none 处理，结果与之前一致)
        self.brighten_config = get_brighten_config(model)
        if self.use_skip:
            enable_vae_skip(self.vae, model["rank_vae"], decoder_lora_modules=model["vae_lora_decoder_modules"],
                            feature_norm=self.skip_feature_norm, color_path=self.skip_color_path,
                            color_pool_factor=self.skip_color_pool)

        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        if self.use_skip:
            self.vae.set_adapter(['default_encoder', 'default_decoder'])
            for n, p in self.vae.named_parameters():
                if "skip_conv" in n:
                    p.data.copy_(model["state_dict_vae_skip"][n])
        else:
            self.vae.set_adapter(['default_encoder'])
        self.vae.load_state_dict(model["state_dict_vae_time"], strict=False)


    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds


    def normal_latent(self, latent):
        mean = latent.mean(dim=(2, 3), keepdim=True)
        std = latent.std(dim=(2, 3), keepdim=True)
        latent = (latent - mean) / (std + 1e-8)
        return latent

    def shift_latent(self, latent, target_latent):
        latent = latent * target_latent.std(dim=(2, 3), keepdim=True) + target_latent.mean(dim=(2, 3), keepdim=True)
        return latent

    
    def get_x0_from_res(self, latent_lq, model_pred, timesteps):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=latent_lq.device, dtype=latent_lq.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(latent_lq.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        x0 = (latent_lq / alpha_prod_t ** (0.5)) - model_pred
        return x0
    

    # @perfcount
    @torch.no_grad()
    def forward(self, lq, prompt, timesteps):

        prompt_embeds = self.encode_prompt([prompt])

        # [LLIE] 第 11 轮：按权重里记录的设置在整图上预提亮(与训练时数据集里的处理一致)，float32 计算
        if self.brighten_config["mode"] != "none":
            lq_01 = (lq.float() + 1) / 2
            lq = brighten_input(lq_01, **self.brighten_config) * 2 - 1
        
        # [LLIE] 第 9 轮：默认取后验均值，去掉采样带来的随机性(大 t 时这部分噪声会被 1/sqrt(alpha_t) 放大)
        latent_distribution = self.vae.encode(lq.to(self.weight_dtype), timesteps).latent_dist
        lq_latent = (latent_distribution.mode() if self.latent_mode == "mean" else latent_distribution.sample()) * self.vae.config.scaling_factor
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred = self.unet(lq_latent, timesteps, encoder_hidden_states=prompt_embeds).sample
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        # [LLIE] 原为 self.timesteps(TADSR_test 中未定义，大图走分块时会报错)，改为传入的 timesteps
                        model_out = self.unet(input_list_t, timesteps, encoder_hidden_states=prompt_embeds.to(self.weight_dtype),).sample
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.get_x0_from_res(lq_latent, model_pred, timesteps)

        # [LLIE] 把编码器特征交给解码器；强度 = gate(t) * skip_scale。skip_scale=0 相当于关掉 skip(用来对比)
        if self.use_skip:
            gate = compute_skip_gate(self.skip_gate_mode, timesteps, self.noise_scheduler.alphas_cumprod,
                                     self.noise_scheduler.config.num_train_timesteps)
            self.vae.decoder.incoming_skip_features = self.vae.encoder.current_skip_features
            self.vae.decoder.skip_gate = gate * self.skip_scale
            # [LLIE] 第 7 轮：低频颜色通路不乘 gate(t)，只乘 skip_scale；skip_scale=0 时两条通路都关闭(与训练中不走旁路的情形一致)
            self.vae.decoder.color_scale = self.skip_scale
        try:
            output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        finally:
            self.vae.decoder.incoming_skip_features = None
            self.vae.encoder.current_skip_features = None

        return output_image

    def decode_latent(self, latent):
        output_image = (self.vae.decode(latent.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image
    
    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu, time_vae=True)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))


