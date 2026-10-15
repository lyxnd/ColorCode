import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from tadsr import TADSR_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def brighten_for_tagging(img, target_mean=0.4):
    # [LLIE] 仅用于 RAM 打 tag：用 gamma 把暗图平均亮度拉到 target_mean；送入网络的仍是原图
    if target_mean is None or target_mean <= 0:
        return img
    img = img.clamp(1e-4, 1.0)
    mean = img.mean().clamp(1e-4, 0.999)
    if mean >= target_mean:
        return img
    gamma = (torch.log(torch.tensor(target_mean, device=img.device)) / torch.log(mean)).clamp(0.1, 1.0)
    return img ** gamma


def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(brighten_for_tagging(lq, args.ram_brighten)).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"
    
    return validation_prompt, lq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str,
                        default='',
                        help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                        help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='./weights', help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    # [LLIE] 原默认 512：短边<512 的图(如 LOL 600x400)会被先放大再处理；LLIE 默认按原分辨率处理，设 0 关闭放大
    parser.add_argument("--process_size", type=int, default=0)
    parser.add_argument("--upscale", type=int, default=1)  # [LLIE] 原默认 4 (SR)，LLIE 不放大
    # [LLIE] 原默认 adain；adain/wavelet 会把输出颜色与亮度对齐回暗输入，LLIE 必须用 nofix
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='nofix')
    parser.add_argument("--ram_brighten", type=float, default=0.4, help='[LLIE] target mean brightness of the copy used for RAM tagging; <=0 disables')
    # [LLIE] 推理时对 skip 强度再乘一个系数(只对带 skip 的权重有效)：1 = 按训练时的强度，0 = 关掉 skip(用来对比)
    parser.add_argument("--skip_scale", type=float, default=1.0)
    # [LLIE] 第 9 轮：VAE 编码取后验均值(mean，默认，结果确定)还是采样(sample，原做法；配合 --seed 可复现)
    parser.add_argument("--latent_mode", type=str, choices=['mean', 'sample'], default='mean')
    parser.add_argument("--tadsr_path", type=str, default='./weights/osediff_my.pkl')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_path', type=str, default='./weights/ram_swin_large_14m.pth')
    parser.add_argument('--ram_ft_path', type=str, default='./weights/DAPE.pth')
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False) # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--timesteps", type=int, default=500)

    args = parser.parse_args()

    # [LLIE] 第 9 轮：训练时 t 只取 np.linspace(1, 999, 50, dtype=int) 这 50 个值；用其他 t 推理时给出提示
    trained_timesteps = np.linspace(1, 999, 50, dtype=int)
    if args.timesteps not in trained_timesteps:
        nearest = int(trained_timesteps[np.abs(trained_timesteps - args.timesteps).argmin()])
        print(f"[note] --timesteps {args.timesteps} was never used in training; the nearest trained value is {nearest}. "
              f"Trained values: {trained_timesteps.tolist()}")

    # initialize the model
    model = TADSR_test(args)

    # get all input images
    if os.path.isdir(args.input_image):
        # [LLIE] 支持常见图片后缀(原来只读 *.png)
        image_names = sorted(set(p for ext in ('png', 'jpg', 'jpeg', 'bmp', 'PNG', 'JPG', 'JPEG', 'BMP')
                                 for p in glob.glob(os.path.join(args.input_image, f'*.{ext}'))))
    else:
        image_names = [args.input_image]


    # get ram model
    DAPE = ram(pretrained=args.ram_path,
            pretrained_condition=args.ram_ft_path,
            image_size=384,
            vit='swin_l')
    DAPE.eval()
    DAPE.to("cuda")

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # set weight type
    DAPE = DAPE.to(dtype=weight_dtype)
    
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)
    
    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')


    for i in range(len(image_names)):
        # make sure that the input image is a multiple of 8
        image_name = image_names[i]
        input_image = Image.open(image_name).convert('RGB')


        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False
        if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
            scale = (args.process_size//rscale)/min(ori_width, ori_height)
            input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))

        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # get caption
        validation_prompt, lq = get_validation_prompt(args, input_image, DAPE)
        if args.save_prompts:
            txt_save_path = f"{txt_path}/{bname.split('.')[0]}.txt"
            with open(txt_save_path, 'w', encoding='utf-8') as f:
                f.write(validation_prompt)
                f.close()
        print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8'))

        # translate the image
        with torch.no_grad():
            lq = lq*2-1
            # [LLIE] 第 9 轮：每张图前重置随机种子(原来 --seed 没有被使用)。latent_mode=sample 时结果可复现，
            # 且与图片的处理顺序无关；latent_mode=mean 时本来就是确定的
            torch.manual_seed(args.seed)
            output_image = model(lq, prompt=validation_prompt, timesteps=args.timesteps)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)
            else:
                pass
            # [LLIE] 原代码 resize 结果未赋值(无效)，且只在 resize_flag 时执行；
            # 现在总是还原到 原图尺寸*upscale，保证与 GT 逐像素对齐(也抵消了 %8 裁边)
            target_size = (int(args.upscale*ori_width), int(args.upscale*ori_height))
            if output_pil.size != target_size:
                output_pil = output_pil.resize(target_size, Image.BICUBIC)


        # [LLIE] 统一存为 png，避免 jpg 输入时二次有损压缩影响 PSNR/SSIM
        output_pil.save(os.path.join(args.output_dir, os.path.splitext(bname)[0] + '.png'))

