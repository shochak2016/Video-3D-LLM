#!/bin/bash
# Experiment 7 — ScanQA, DINOv3 fusion replaces 3D PE.
#   - PE OFF:   no --world_position_embedding_type
#   - DINO IN:  --spatial_feature_type dinov3-concat  (A/B: dinov3-add) (+ --dino_feature_dir)
#   - Trainable: LoRA on the LLM + full-Adam dino_fusion MLP; SigLIP + projector frozen.
#   - DeepSpeed ZeRO-2 (not 3): base fits 80GB under LoRA; avoids PCIe all-gather tax.
# Run the precompute FIRST (in the dino env), pointing --dino_feature_dir at its output:
#   /path/to/.venvs/dino/bin/python scripts/3d/preprocessing/precompute_dino_features.py \
#       --split train --frames_upbound 32 --out_dir /mnt/local/dino_features/dinov3-vitb16
#   (also --split val for eval)

IMAGE_FOLDER="data"
VIDEO_FOLDER="data"
DATA_YAML="scripts/3d/train/scanqa.yaml"
DINO_FEATURE_DIR="/mnt/local/dino_features/dinov3-vitb16"   # <-- precompute output

alias python=python3
nvidia-smi

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="exp7-scanqa-dinov3-fuse"
PREV_STAGE_CHECKPOINT="data/models/LLaVA-Video-7B-Qwen2"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

NUM_GPUS=8
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE/NUM_GPUS))

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port 43000 \
    llava/train/train_3d.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --embodiedscan_folder data/embodiedscan/ \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --spatial_feature_type dinov3-concat \
    --dino_feature_dim 768 \
    --dino_feature_dir $DINO_FEATURE_DIR \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./ckpt/$MID_RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 4 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --frame_sampling_strategy mc \
    --frames_upbound 32 \
    > "./ckpt/${MID_RUN_NAME}.log" 2>&1
exit 0;
