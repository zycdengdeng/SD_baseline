#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH


n_nodes=1
n_gpus_per_node=2 # number of gpus
torch_num_workers=2
batch_size=12 # make the batch size as large as possible
exp_name="train_pano_outpaint_release=$(($n_gpus_per_node * $n_nodes * $batch_size))"

export CUDA_VISIBLE_DEVICES=0,1
python -u /mnt/vdb1/lyt/ArbiViewGen-main/train.py /mnt/vdb1/lyt/ArbiViewGen-main/configs/pano_generation_outpaint.yaml \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" --strategy="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --log_every_n_steps=100 \
    --num_sanity_val_steps=1 \
    --limit_val_batches 4 \
    --benchmark=True \
    --max_epochs=10 \
    --val_check_interval 1.0 