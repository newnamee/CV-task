#!/usr/bin/env bash
# chmod +x autodl-tmp/project/FarSeg/scripts/train_farseg50.sh
# bash autodl-tmp/project/FarSeg/scripts/train_farseg50.sh
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
#export PYTHONPATH=$PYTHONPATH:`pwd`
export PYTHONPATH=$PYTHONPATH:/autodl-tmp/project/FarSeg
config_path='isaid.farseg50'
model_dir='autodl-tmp/project/FarSeg/log/isaid_segm/farseg50'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 9996 autodl-tmp/project/FarSeg/apex_train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    --opt_level='O1'

#python autodl-tmp/project/FarSeg/apex_train.py \
#    --local_rank=0 \
#    --config_path=${config_path} \
#    --model_dir=${model_dir} \
#    --opt_level='O1'