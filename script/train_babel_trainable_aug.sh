#!/bin/bash

# $1 : experiment name
# $2 : cuda id

CONFIG="babel_202_trainable_aug"

DIR="/home/darong/other_storage/e2e_model/"

echo "Start running training process of E2E ASR"
CUDA_VISIBLE_DEVICES=$2 python3 main.py --config config/${CONFIG}.yaml \
    --name $1 \
    --njobs 4 \
    --seed 0 \
    --logdir ${DIR}/log/ \
    --ckpdir ${DIR}/ckpt/ \
    --outdir ${DIR}/result/ \
    --reserve_gpu 0 \
    --babel babel \
    --no_cudnn \
    # --load_aug /Home/daniel094144/E2E-ASR-Pytorch/ckpt/nofast_sgd0.1lr_-3/best_aug.pth

