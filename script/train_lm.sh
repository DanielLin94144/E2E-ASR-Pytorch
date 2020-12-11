#!/bin/bash

CONFIG="librispeech_lm"
DIR="/Home/daniel094144/E2E-ASR-Pytorch/libri_LM" # where you want to save your LM

echo "Start running training process of RNNLM"
CUDA_VISIBLE_DEVICES=$2 python3 main.py --config config/${CONFIG}.yaml \
    --name $1 \
    --njobs 16 \
    --seed 0 \
    --lm \
    --logdir ${DIR}/log/ \
    --ckpdir ${DIR}/ckpt/ \
    --outdir ${DIR}/result/ \
    --reserve_gpu 0 \
