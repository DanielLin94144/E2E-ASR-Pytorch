#!/bin/bash

CONFIG="librispeech_lm"
DIR="/Home/daniel094144/End-to-End-jointCTC-Attention-ASR/libri_LM"

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
