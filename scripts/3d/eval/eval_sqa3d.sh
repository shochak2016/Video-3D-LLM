#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="./ckpt/$1"
ANWSER_FILE="results/sqa3d/${1}_${2}_fsr${3}.jsonl"

TEST_SIZE=${4:-10000000}

CUDA_VISIBLE_DEVICES=0 python3 llava/eval/model_sqa3d.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 1 \
    --question-file data/processed/sqa3d_test_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --test_size $TEST_SIZE \
    --overwrite_cfg true

python llava/eval/eval_sqa3d.py --input-file $ANWSER_FILE
