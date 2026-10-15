import os
import gc
import lpips
import clip
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from datetime import datetime

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from tadsr import TADSR_reg, TADSR_gen
from dataloaders.paired_dataset import PairedCaptionDataset  

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")

def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")

def parse_str_list(arg):
    return arg.split(',')

def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # training details
    parser.add_argument("--output_dir", default='experience/tadsr')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    
    
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff", help="The name of the wandb project to log to.")
    parser.add_argument('--txt_path', type=parse_str_list, default=['YOUR TXT FILE PATH'], help='A comma-separated list of integers')
    parser.add_argument('--dataset_prob_paths_list', type=parse_int_list, default=[1], help='A comma-separated list of integers')
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)
    parser.add_argument("--pretrained_model_name_or_path_reg", default=None, type=str)
    parser.add_argument("--pretrained_model_name_or_path_fix", default=None, type=str)
    parser.add_argument("--pretrained_lora_path", default=None, type=str)
    
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--neg_prompt", default="", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)
    parser.add_argument("--deg_prob", default=0.99, type=float)

    # lora setting
    parser.add_argument("--lora_rank", default=4, type=int)
    # ram path
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM model')

    # [LLIE] validation / best checkpoint
    parser.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True,
        help="Save checkpoints/model_best.pkl according to the validation metric (needs --val_txt_path). Use --no-save_best to disable.")
    parser.add_argument("--val_txt_path", type=str, default=None,
        help="Validation list file, same format as --txt_path (GT<TAB>LQ<TAB>tag). Generate it with get_tag.py on the test split.")
    parser.add_argument("--validation_steps", type=int, default=500, help="Run validation every N optimization steps.")
    parser.add_argument("--val_timesteps", type=parse_int_list, default=[500],
        help="Comma-separated timesteps used for validation; metrics are averaged over them.")
    parser.add_argument("--best_metric", type=str, default="psnr", choices=["psnr", "psnr_gtmean", "lpips"],
        help="Metric that decides the best checkpoint (psnr / psnr_gtmean: higher is better, lpips: lower is better). "
             "psnr_gtmean rescales each prediction to the GT mean brightness first, so a global exposure mismatch "
             "between train and test GT does not decide the best checkpoint.")
    parser.add_argument("--val_num_images", type=int, default=0, help="Use only the first N validation images; 0 = all.")

    # [LLIE] time-aware skip (encoder -> decoder bypass around the VAE latent bottleneck)
    parser.add_argument("--use_skip", action=argparse.BooleanOptionalAction, default=True,
        help="Add encoder->decoder skip connections and a decoder LoRA to preserve text and fine details. Use --no-use_skip to disable.")
    parser.add_argument("--skip_gate", type=str, default="sqrt_alpha", choices=["sqrt_alpha", "linear", "none"],
        help="How the skip strength depends on t: sqrt_alpha = sqrt(alpha_cumprod[t]), linear = 1 - t/1000, none = constant 1.")
    # [LLIE] round 4: B = per-channel normalization of skip features, A = auxiliary reconstruction loss without skip
    parser.add_argument("--skip_feature_norm", action=argparse.BooleanOptionalAction, default=True,
        help="Normalize each skip feature per channel (no learnable params) before the 1x1 conv, so the gate really controls the skip "
             "strength and global brightness/color must come from the latent. Use --no-skip_feature_norm to disable.")
    parser.add_argument("--lambda_noskip", type=float, default=1.0,
        help="Weight of the extra reconstruction loss (blurred L2 + LPIPS) on the image decoded WITHOUT skip from the same latent. "
             "Forces the latent itself to carry the full image. 0 disables it.")
    # [LLIE] round 6: global colour statistics loss against desaturated ("greyish") results
    parser.add_argument("--lambda_color", type=float, default=1.0,
        help="Weight of the global colour statistics loss: per-image mean/std of the Cb and Cr chroma channels and the mean "
             "chroma magnitude are matched to the GT (no pixel-wise term, no brightness term). Applied to both the skip and "
             "the no-skip outputs. 0 disables it.")
    # [LLIE] round 7: frequency-split skip -- low frequencies (colour/exposure) bypass the gate, high frequencies stay gated
    parser.add_argument("--skip_color_path", action=argparse.BooleanOptionalAction, default=True,
        help="Split each skip feature into a low-frequency part (colour/exposure: not normalized, NOT multiplied by gate(t), "
             "own 1x1 conv) and a high-frequency part (details: normalized + gate(t)). Lets the input colours reach the output "
             "at every t while t keeps controlling the details. Use --no-skip_color_path to disable.")
    parser.add_argument("--skip_color_pool", type=int, default=32,
        help="Low-pass strength of the colour path: features are average-pooled to about 1/N of the image resolution.")
    # [LLIE] round 11: deterministic pre-brightening of the low-light input before the VAE encoder
    parser.add_argument("--input_brighten", type=str, default="gain", choices=["none", "gain", "fixed"],
        help="Brighten the low-light input before the VAE encoder (no learnable params; same scale on R,G,B so hue is kept). "
             "gain = per-image gain to reach --input_brighten_target mean luminance (computed on the FULL image, before cropping); "
             "fixed = the same gain --input_brighten_fixed_gain for every image; none = feed the raw dark image (old behaviour). "
             "Stored in the checkpoint; test_tadsr.py applies the same transform automatically.")
    parser.add_argument("--input_brighten_target", type=float, default=0.4, help="Target mean luminance for --input_brighten gain.")
    parser.add_argument("--input_brighten_max_gain", type=float, default=20.0, help="Upper bound of the per-image gain.")
    parser.add_argument("--input_brighten_fixed_gain", type=float, default=8.0, help="Gain used by --input_brighten fixed.")
    parser.add_argument("--input_brighten_knee", type=float, default=0.8,
        help="Luminance above this is softly compressed instead of clipped (1 = hard clip).")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

import torch
import torch.nn as nn


def gaussian_blur(img, kernel_size=5, sigma=1.0, timesteps=None):
    import torchvision.transforms.functional as TF
    from torchvision.transforms import GaussianBlur
    if timesteps is not None:
        out = []
        for i in range(img.shape[0]):
            k = kernel_size + int(timesteps[i].item() / 200) * 4
            s = sigma + timesteps[i].item() / 200
            blur = GaussianBlur(kernel_size=k, sigma=s)
            out.append(blur(img[i].unsqueeze(0)))
        img_blur = torch.cat(out, dim=0)
        return img_blur
    else:
        blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return blur(img)


# [LLIE] 第 6 轮：全局颜色统计损失
# 每个统计量的典型量级(0~1 刻度下)，用来把各项归一化到相近的量级：
# 色度类(Cb/Cr 均值、标准差、平均幅度)约 0.05；HSV 平均饱和度约 0.25(LOL-v2 Real GT 为 0.23~0.27)
COLOR_STATISTIC_SCALES = (0.05, 0.05, 0.05, 0.05, 0.05, 0.25)


def chroma_statistics(image_01):
    """每张图的 6 个颜色统计量：
    Cb 均值、Cr 均值(整体色偏)；Cb 标准差、Cr 标准差、平均色度幅度 sqrt(Cb^2+Cr^2)(颜色浓度)；
    HSV 平均饱和度 (max-min)/max(对"整体抬亮导致的发灰"也敏感：RGB 同时加同一个数时 Cb/Cr 不变，但饱和度下降)。
    image_01: [B,3,H,W]，范围 [0,1]。返回 [B,6]。"""
    red, green, blue = image_01[:, 0], image_01[:, 1], image_01[:, 2]
    chroma_blue = -0.168736 * red - 0.331264 * green + 0.5 * blue
    chroma_red = 0.5 * red - 0.418688 * green - 0.081312 * blue
    chroma_magnitude = torch.sqrt(chroma_blue ** 2 + chroma_red ** 2 + 1e-8)
    maximum, minimum = image_01.max(dim=1).values, image_01.min(dim=1).values
    saturation = (maximum - minimum) / (maximum + 1e-2)  # +1e-2：避免极暗像素的噪声主导
    return torch.stack([chroma_blue.mean(dim=(1, 2)), chroma_red.mean(dim=(1, 2)),
                        chroma_blue.std(dim=(1, 2)), chroma_red.std(dim=(1, 2)),
                        chroma_magnitude.mean(dim=(1, 2)), saturation.mean(dim=(1, 2))], dim=1)


def color_statistics_loss(prediction, target):
    """[LLIE] 让结果的整体颜色分布与 GT 一致，用来纠正整体偏灰(饱和度不足)。

    - Cb/Cr 均值：整体色偏；Cb/Cr 标准差、平均色度幅度、HSV 平均饱和度：整体饱和度/颜色丰富程度。
    - 各统计量先除以各自的典型量级(COLOR_STATISTIC_SCALES)再取平均。
    - 只比较全图统计量，不做逐像素约束：偏灰本身来自逐像素回归在颜色不确定时"取平均"，
      逐像素颜色损失容易重蹈覆辙；全局统计量只要求整体颜色"够浓"、色偏一致。
    - 不含亮度项：LOL-v2 Real 训练集 GT 比测试集亮约 20%，亮度对齐到训练集 GT 没有意义。
    - 与 t 无关，不影响 t 对细节程度的调节。
    prediction / target: [B,3,H,W]，范围 [-1,1]。
    """
    prediction_01 = (prediction.float() + 1) / 2
    target_01 = (target.float() + 1) / 2
    scales = torch.tensor(COLOR_STATISTIC_SCALES, device=prediction_01.device)
    return ((chroma_statistics(prediction_01) - chroma_statistics(target_01)).abs() / scales).mean()


@torch.no_grad()
def run_validation(model, net_lpips, val_dataset, args, accelerator):
    """[LLIE] 在验证集上整图推理，返回平均 PSNR / LPIPS（对所有图片和 --val_timesteps 取平均）。

    model 是未被 DDP 包装的 TADSR_gen。验证时固定随机种子(VAE latent 采样有随机性)，
    保证不同 step 之间的指标可比；结束后恢复训练的随机状态和 train 模式。
    """
    device = accelerator.device
    unet_was_training, vae_was_training = model.unet.training, model.vae.training
    model.unet.eval()
    model.vae.eval()

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(0)

    num_images = len(val_dataset)
    if args.val_num_images > 0:
        num_images = min(num_images, args.val_num_images)

    psnr_sum, lpips_sum, count = 0.0, 0.0, 0
    psnr_gtmean_sum, saturation_ratio_sum = 0.0, 0.0  # [LLIE] 第 6 轮

    def mean_saturation(image_01):
        maximum, minimum = image_01.max(dim=1).values, image_01.min(dim=1).values
        return ((maximum - minimum) / maximum.clamp_min(1e-6)).mean().item()

    try:
        for image_index in range(num_images):
            example = val_dataset[image_index]
            lq = example["conditioning_pixel_values"].unsqueeze(0).to(device)
            gt = example["output_pixel_values"].unsqueeze(0).to(device)
            # VAE 下采样 8 倍，裁到 8 的倍数
            height, width = lq.shape[-2:]
            height, width = height - height % 8, width - width % 8
            lq, gt = lq[..., :height, :width], gt[..., :height, :width]
            batch = {"prompt": [example["prompt"]], "neg_prompt": [example["neg_prompt"]]}

            for timestep in args.val_timesteps:
                timesteps = torch.tensor([timestep], device=device).long()
                with accelerator.autocast():
                    prediction = model(lq, batch=batch, timesteps=timesteps)[0]
                prediction = prediction.float().clamp(-1, 1)
                # 量化到 8bit，与保存成 png 后再算的指标一致
                prediction_01 = ((prediction + 1) / 2 * 255).round() / 255
                gt_01 = (gt.float() + 1) / 2
                mse = F.mse_loss(prediction_01, gt_01).clamp_min(1e-10)
                psnr_sum += (10 * torch.log10(1.0 / mse)).item()
                lpips_sum += net_lpips(prediction_01 * 2 - 1, gt.float()).mean().item()
                # [LLIE] GT-mean PSNR：先把预测整体缩放到 GT 的平均亮度(与 eval_llie.py 一致)；以及饱和度比
                prediction_gtmean = (prediction_01 * (gt_01.mean() / prediction_01.mean().clamp_min(1e-6))).clamp(0, 1)
                mse_gtmean = F.mse_loss(prediction_gtmean, gt_01).clamp_min(1e-10)
                psnr_gtmean_sum += (10 * torch.log10(1.0 / mse_gtmean)).item()
                saturation_ratio_sum += mean_saturation(prediction_01) / max(mean_saturation(gt_01), 1e-6)
                count += 1
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        model.unet.train(unet_was_training)
        model.vae.train(vae_was_training)

    count = max(count, 1)
    return {"psnr": psnr_sum / count, "lpips": lpips_sum / count,
            "psnr_gtmean": psnr_gtmean_sum / count, "sat_ratio": saturation_ratio_sum / count}


def main(args):

    def sanitize_config(config):
        sanitized = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                sanitized[k] = v
            elif isinstance(v, (list, tuple)):
                sanitized[k] = str(v)
            else:
                try:
                    sanitized[k] = str(v)
                except:
                    continue
        return sanitized

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    model_gen = TADSR_gen(args)
    model_gen.set_train()

    model_reg = TADSR_reg(args=args, accelerator=accelerator)
    model_reg.set_train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # set gen adapter
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            # model_reg.unet_fix.enable_xformers_memory_efficient_attention()
            model_reg.unet_update.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        # model_reg.unet_fix.enable_gradient_checkpointing()
        model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = []
    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    optimizer_name = []
    for n, _p in model_gen.vae.named_parameters():
        if "lora" in n or _p.requires_grad:
            layers_to_opt.append(_p)
            if "time" in n:
                optimizer_name.append(n)


    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    layers_to_opt_reg = []
    for n, _p in model_reg.unet_update.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedCaptionDataset(split="train", args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=False)

    # [LLIE] 验证集(用于保存 best)。split="val" 不裁剪不翻转，整图推理
    dataset_val = None
    if args.save_best:
        if args.val_txt_path is not None and os.path.isfile(args.val_txt_path):
            dataset_val = PairedCaptionDataset(split="val", args=args, txt_path=args.val_txt_path)
            if len(dataset_val) == 0:
                dataset_val = None
        if dataset_val is None and accelerator.is_main_process:
            print(f"[WARNING] --save_best is on but no usable validation list was found (--val_txt_path={args.val_txt_path}). "
                  f"model_best.pkl will NOT be saved; only the periodic checkpoints are kept.")


    
    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg)

    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        args.txt_path = str(args.txt_path)
        tracker_config = sanitize_config(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    
    timesteps_list = np.linspace(1, 999, 50, dtype=int)
    timesteps_list = torch.tensor(timesteps_list)

    # [LLIE] best checkpoint 记录
    best_value = None
    best_step = -1
    last_validated_step = -1
    last_saved_step = -1
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    def validate_and_save_best(step):
        # 只在主进程调用。验证 -> 记录日志 -> 指标更好时覆盖 model_best.pkl
        nonlocal best_value, best_step, last_validated_step
        if dataset_val is None or step == last_validated_step:
            return
        last_validated_step = step
        metrics = run_validation(accelerator.unwrap_model(model_gen), net_lpips, dataset_val, args, accelerator)
        value = metrics[args.best_metric]
        higher_is_better = args.best_metric in ("psnr", "psnr_gtmean")  # [LLIE] 第 6 轮新增 psnr_gtmean
        is_better = best_value is None or (value > best_value if higher_is_better else value < best_value)
        if is_better:
            best_value, best_step = value, step
            accelerator.unwrap_model(model_gen).save_model(os.path.join(checkpoint_dir, "model_best.pkl"))
            with open(os.path.join(checkpoint_dir, "best.txt"), "w", encoding="utf-8") as best_file:
                best_file.write(f"step={step}\nmetric={args.best_metric}\npsnr={metrics['psnr']:.4f}\n"
                                f"psnr_gtmean={metrics['psnr_gtmean']:.4f}\nlpips={metrics['lpips']:.4f}\n"
                                f"sat_ratio={metrics['sat_ratio']:.4f}\nval_timesteps={args.val_timesteps}\n")
        accelerator.log({"val/psnr": metrics["psnr"], "val/lpips": metrics["lpips"],
                         "val/psnr_gtmean": metrics["psnr_gtmean"], "val/sat_ratio": metrics["sat_ratio"],
                         f"val/best_{args.best_metric}": best_value}, step=step)
        tqdm.write(f"[val] step {step}: psnr={metrics['psnr']:.3f} psnr_gtmean={metrics['psnr_gtmean']:.3f} "
                   f"lpips={metrics['lpips']:.4f} sat_ratio={metrics['sat_ratio']:.3f} | "
                   f"best {args.best_metric}={best_value:.4f} @ step {best_step}" + ("  <- new best" if is_better else ""))

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, model_reg]
            with accelerator.accumulate(*m_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape

                index = torch.randint(0, len(timesteps_list), (B,))
                timesteps = timesteps_list[index].to(accelerator.device)
                timesteps = timesteps.long()


                # forward pass
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds, x_tgt_pred_noskip = model_gen(x_src, batch=batch, timesteps=timesteps)
                # Reconstruction loss
                loss_l2 = F.mse_loss(gaussian_blur(x_tgt_pred, timesteps=timesteps).float(), gaussian_blur(x_tgt, timesteps=timesteps).float(), reduction="mean") * args.lambda_l2

                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                # [LLIE] A：不走旁路的那次解码也要还原 GT(形式与上面的重建损失相同，再乘 lambda_noskip)
                loss_noskip = None
                if x_tgt_pred_noskip is not None:
                    loss_noskip_l2 = F.mse_loss(gaussian_blur(x_tgt_pred_noskip, timesteps=timesteps).float(), gaussian_blur(x_tgt, timesteps=timesteps).float(), reduction="mean") * args.lambda_l2
                    loss_noskip_lpips = net_lpips(x_tgt_pred_noskip.float(), x_tgt.float()).mean() * args.lambda_lpips
                    loss_noskip = (loss_noskip_l2 + loss_noskip_lpips) * args.lambda_noskip
                    loss = loss + loss_noskip

                # [LLIE] 第 6 轮：全局颜色统计损失(两次输出都加；不走旁路那次再乘 lambda_noskip，保证 latent 这条路也学到正确的颜色)
                loss_color = None
                if args.lambda_color > 0:
                    loss_color = color_statistics_loss(x_tgt_pred, x_tgt) * args.lambda_color
                    if x_tgt_pred_noskip is not None:
                        loss_color = loss_color + color_statistics_loss(x_tgt_pred_noskip, x_tgt) * args.lambda_color * args.lambda_noskip
                    loss = loss + loss_color
                # KL loss
                if torch.cuda.device_count() > 1:
                    loss_kl = model_reg.module.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, args=args, model_timesteps=timesteps)
                    loss_kl = loss_kl * args.lambda_vsd
                else:
                    loss_kl = model_reg.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, args=args, model_timesteps=timesteps)
                    loss_kl = loss_kl * args.lambda_vsd
                loss = loss + loss_kl

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                diff loss: let lora model closed to generator 
                """
                if torch.cuda.device_count() > 1:
                    loss_d = model_reg.module.diff_loss(latents=latents_pred, prompt_embeds=prompt_embeds, args=args)*args.lambda_vsd_lora
                else:
                    loss_d = model_reg.diff_loss(latents=latents_pred, prompt_embeds=prompt_embeds, args=args)*args.lambda_vsd_lora
                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    
                    logs = {}
                    # log all the losses
                   
                    logs["loss_d"] = loss_d.detach().item()
                    logs["loss_kl"] = loss_kl.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if loss_noskip is not None:
                        logs["loss_noskip"] = loss_noskip.detach().item()  # [LLIE] A
                    if loss_color is not None:
                        logs["loss_color"] = loss_color.detach().item()  # [LLIE] 第 6 轮
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)
                        last_saved_step = global_step

                    accelerator.log(logs, step=global_step)

                    # [LLIE] 定期验证并保存 best
                    if global_step % args.validation_steps == 0:
                        validate_and_save_best(global_step)

                # [LLIE] 到达 max_train_steps 后停止(原代码只把它用于进度条和学习率调度，不会停)
                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    # [LLIE] 训练结束：保存最终权重，做最后一次验证，关闭日志
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if last_saved_step != global_step:
            outf = os.path.join(checkpoint_dir, f"model_{global_step}.pkl")
            accelerator.unwrap_model(model_gen).save_model(outf)
        validate_and_save_best(global_step)
        tqdm.write(f"Training finished at step {global_step}."
                   + (f" Best {args.best_metric}={best_value:.4f} at step {best_step} (checkpoints/model_best.pkl)." if best_value is not None else ""))
    progress_bar.close()
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)