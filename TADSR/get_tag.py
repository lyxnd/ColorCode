'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution
 * Modified for unified dataset list generation
 * [LLIE] 输出格式改为 "GT路径<TAB>LQ路径<TAB>tag"，与 PairedCaptionDataset 读取顺序一致；
 *        默认在 GT(正常光) 图上打 tag，暗图上 RAM 几乎识别不出内容
'''
import torch
from torchvision import transforms
import os
import glob
import argparse
from PIL import Image
from ram.models.ram import ram
from ram import inference_ram as inference
from tqdm import tqdm


ram_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

parser = argparse.ArgumentParser()

parser.add_argument("--ram_path", type=str, default='', help='RAM weight')
parser.add_argument("--hr_path", type=str, default='', help='GT (normal-light) images directory')
parser.add_argument("--lr_path", type=str, default='', help='LQ (low-light) images directory')
parser.add_argument("--out_txt", type=str, default='', help='Path to save the combined metadata txt')
parser.add_argument("--tag_source", type=str, default='hr', choices=['hr', 'lr'], help='[LLIE] which image RAM tags are computed on')
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()


# [LLIE] 支持常见图片后缀
IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP')
lr_lists = sorted(set(p for ext in IMG_EXTS for p in glob.glob(os.path.join(args.lr_path, ext))))
num_imgs = len(lr_lists)
print(f"Found {num_imgs} LQ images.")

device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
model = ram(pretrained=args.ram_path, image_size=384, vit='swin_l')
model = model.eval().to(device)
# Start processing and writing to file
# Using 'w' mode will overwrite the file if it already exists
num_written = 0
with torch.no_grad(), open(args.out_txt, "w", encoding="utf-8") as f:
    # Use tqdm for a progress bar with estimated time
    for lq_path in tqdm(lr_lists, desc="Tagging progress"):
        try:
            basename = os.path.basename(lq_path)

            # --- Filename matching logic ---
            hr_path = os.path.join(args.hr_path, basename)

            # Check if corresponding HR exists
            if not os.path.exists(hr_path):
                print(f"Warning: GT image not found for: {hr_path}")
                continue

            tag_img_path = hr_path if args.tag_source == 'hr' else lq_path
            img_input = ram_transforms(Image.open(tag_img_path).convert('RGB')).unsqueeze(0).to(device)
            captions = inference(img_input, model)

            tag_text = captions[0].replace('\t', ' ').replace('\n', ' ')
            # Absolute paths are used to ensure the training script can locate files from any directory
            lq_abs = os.path.abspath(lq_path)
            hr_abs = os.path.abspath(hr_path)
            # [LLIE] 顺序: GT, LQ, tag (制表符分隔，路径中可含空格)
            f.write(f"{hr_abs}\t{lq_abs}\t{tag_text}\n")
            num_written += 1

        except Exception as e:
            print(f"\nError processing {lq_path}: {e}")

print(f"\n>> Task completed! {num_written} pairs saved to: {args.out_txt}")
