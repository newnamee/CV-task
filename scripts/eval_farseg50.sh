#!/usr/bin/env bash
# chmod +x autodl-tmp/project/FarSeg/scripts/eval_farseg50.sh
# bash autodl-tmp/project/FarSeg/scripts/eval_farseg50.sh
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='isaid.farseg50'
model_dir='autodl-tmp/project/FarSeg/log/isaid_segm/farseg50'
ckpt_path='autodl-tmp/project/FarSeg/log/isaid_segm/farseg50/model-60000.pth'
vis_dir='autodl-tmp/project/FarSeg/log/isaid_segm/farseg50/vis-60000'

image_dir='/root/autodl-tmp/Data/iSAID/val/images'
mask_dir='/root/autodl-tmp/Data/iSAID/val/masks'

python autodl-tmp/project/FarSeg/isaid_eval.py \
    --config_path=${config_path} \
    --ckpt_path=${ckpt_path} \
    --image_dir=${image_dir} \
    --mask_dir=${mask_dir} \
    --vis_dir=${vis_dir} \
    --log_dir=${model_dir} \
    --patch_size=896