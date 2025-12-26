#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

n_nodes=1
n_gpus_per_node=2 # 双GPU配置
torch_num_workers=10 # 平衡数据加载和内存使用
batch_size=6 # 在5和8之间取平衡，避免内存爆炸
exp_name="train_pano_outpaint_dual_gpu_optimized=$(($n_gpus_per_node * $n_nodes * $batch_size))"

# 使用GPU 0和1
export CUDA_VISIBLE_DEVICES=0,1

# 移除内存分配限制，提高性能
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

python -u /mnt/vdb1/lyt/ArbiViewGen-main/train.py /mnt/vdb1/lyt/ArbiViewGen-main/configs/pano_generation_outpaint.yaml \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" --strategy="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --log_every_n_steps=50 \
    --num_sanity_val_steps=0 \
    --limit_val_batches 4 \
    --benchmark=True \
    --max_epochs=10 \
    --val_check_interval 1.0 \
    --accumulate_grad_batches=2