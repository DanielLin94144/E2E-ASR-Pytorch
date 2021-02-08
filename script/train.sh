#!/bin/bash

# $1 : experiment name
# $2 : cuda id

CONFIG="librispeech_asr"

DIR="/Home/daniel094144/E2E-ASR-Pytorch/"

echo "Start running training process of E2E ASR"
CUDA_VISIBLE_DEVICES=$0 python3 main.py --config config/${CONFIG}.yaml \
    --name $1 \
    --njobs 10 \
    --seed 0 \
    --logdir ${DIR}/log/ \
    --ckpdir ${DIR}/ckpt/ \
    --outdir ${DIR}/result/ \
    --reserve_gpu 0 \

