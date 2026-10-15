#!/bin/bash
# 干预实验: 生成对齐输入 -> Retinexformer 推理 -> 评测 (2026-10-14 新增, Claude 生成)
# 用法 (在 Retinexformer 目录下, 与 diagnose_capture_difference.py / make_alignment_inputs.py / evaluate_alignment_outputs.py 同目录):
#   bash run_alignment_intervention.sh [官方权重, 默认 pretrained_weights/LOL_v2_real.pth] [GPU 编号, 默认 0]
# 结果: alignment_intervention/alignment_summary_official.txt
set -e
WEIGHTS=${1:-pretrained_weights/LOL_v2_real.pth}
GPU=${2:-0}
DATA=data/LOLv2/Real_captured/Test
ROOT=alignment_intervention

python make_alignment_inputs.py --test_low $DATA/Low --test_gt $DATA/Normal --save_dir $ROOT/inputs

CUDA_VISIBLE_DEVICES=$GPU python Enhancement/infer_folder.py --opt Options/RetinexFormer_LOL_v2_real.yml \
    --weights $WEIGHTS --input_dir $ROOT/inputs --output_dir $ROOT/outputs_official
python evaluate_alignment_outputs.py --test_low $DATA/Low --test_gt $DATA/Normal --alignment_dir $ROOT/inputs \
    --output_dir $ROOT/outputs_official --name official --save_dir $ROOT

# 可选: 重训模型也跑一遍 (去掉注释即可), 看结论是否对不同训练结果都成立
# CUDA_VISIBLE_DEVICES=$GPU python Enhancement/infer_folder.py --opt Options/color_exp/Color_A.yml \
#     --weights experiments/Color_A_LOL_v2_real/models/net_g_latest.pth --input_dir $ROOT/inputs --output_dir $ROOT/outputs_A
# python evaluate_alignment_outputs.py --test_low $DATA/Low --test_gt $DATA/Normal --alignment_dir $ROOT/inputs \
#     --output_dir $ROOT/outputs_A --name A --save_dir $ROOT
