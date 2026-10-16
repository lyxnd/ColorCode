#!/bin/bash
# 依次跑三组 (可重复执行: 已完成的组跳过, 未完成的从最近存档续训). 用法 (仓库根目录): bash analysis/loss_data_ft/run_all.sh
cd "$(dirname "$0")/../.."
for arm in l1 gan bias; do
  [ -f analysis/loss_data_ft/runs/$arm/G_2000.pth ] && continue
  mkdir -p analysis/loss_data_ft/runs/$arm
  python analysis/loss_data_ft/train.py --arm $arm --iters 2000 >> analysis/loss_data_ft/runs/$arm/train.log 2>&1
done
