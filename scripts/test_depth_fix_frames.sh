#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR=$(dirname "$SCRIPTPATH")   # go one level up from scripts/
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

n_nodes=1
n_gpus_per_node=1
torch_num_workers=0
batch_size=1
exp_name="test_depth_gen_fix_frames=$(($n_gpus_per_node * $n_nodes * $batch_size))"

CUDA_VISIBLE_DEVICES='0' python -u ${PROJECT_DIR}/test.py /mnt/vdb1/lyt/ArbiViewGen-main/configs/depth_generation_train.yaml \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" --strategy="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --mode test \
    --ckpt_path /mnt/vdc1/lyt/depth_gen.ckpt
