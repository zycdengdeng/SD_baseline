#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

n_nodes=1
n_gpus_per_node=2 # number of gpus
torch_num_workers=4 # 减少数据加载线程数，避免内存竞争
batch_size=1 # 保守的batch size，考虑到GPU内存使用率较高
exp_name="test_pano_outpaint_dual_gpu_conservative"

export CUDA_VISIBLE_DEVICES=0,1
python -u /mnt/vdb1/lyt/ArbiViewGen-main/test.py /mnt/vdb1/lyt/ArbiViewGen-main/configs/pano_generation_outpaint_test.yaml \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" --strategy="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --log_every_n_steps=100 \
    --num_sanity_val_steps=1 \
    --limit_val_batches=1.0 \
    --benchmark=True \
    --max_epochs=10 \
    --val_check_interval 1.0  \
    --ckpt_path /mnt/vdb1/lyt/logs/tb_logs/train_pano_outpaint_dual_gpu_optimized=12/version_4/checkpoints/epoch=epoch=3-loss=train_loss=0.0466.ckpt
