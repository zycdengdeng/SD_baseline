#!/bin/bash

# 设置环境变量
export PYTHONPATH=/mnt/vdb1/lyt/ArbiViewGen-main:$PYTHONPATH

# 运行指标计算
echo "开始计算PSNR、SSIM和LPIPS指标..."

python scripts/calculate_metrics.py \
    --output_dir /mnt/vdb1/lyt/ArbiViewGen-main/scripts/here \
    --version version_11

echo "指标计算完成！" 