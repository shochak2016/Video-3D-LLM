#!/bin/bash
# Generic single-GPU LoRA fine-tune launcher for Video-3D-LLM.
#
# What it does:
#   - Full-trains mm_projector
#   - LoRA-trains LLM attention (q/k/v/o)
#   - Freezes SigLIP vision tower
#   - DeepSpeed Zero-2 (better than Zero-3 for single-GPU)
#
# How to use it:
#   - Don't invoke directly for an experiment. Use a thin wrapper script
#     (e.g. train/lora/exp4_train.sh) that exports the experiment-specific
#     env vars and then `exec`s this script.
#   - Every knob below is env-overridable. Anything required (DATA_YAML,
#     RUN_NAME, PE_TYPE) has a default that works for sanity testing the
#     LoRA infrastructure against the existing baseline.

set -e
cd "$(dirname "$0")/../.."

# ----- Required-ish (have defaults but you usually want to override) ------
DATA_YAML="${DATA_YAML:-train/lora/exp4.yaml}"
RUN_NAME="${RUN_NAME:-lora-run}"
PE_TYPE="${PE_TYPE:-avg-discrete-sin3d}"

# ----- Base model + vision encoder ---------------------------------------
LLM_VERSION="${LLM_VERSION:-Qwen/Qwen2-7B-Instruct}"
PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT:-data/models/LLaVA-Video-7B-Qwen2}"
VISION_TOWER="${VISION_TOWER:-google/siglip-so400m-patch14-384}"

# ----- Training schedule -------------------------------------------------
EPOCHS="${EPOCHS:-2}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"        # effective batch = PER_DEVICE_BS * GRAD_ACCUM
NUM_FRAMES="${NUM_FRAMES:-16}"
SAMPLING="${SAMPLING:-mc}"

# ----- LoRA hyperparams --------------------------------------------------
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LR_LORA="${LR_LORA:-5e-5}"
LR_PROJECTOR="${LR_PROJECTOR:-5e-5}"

# ----- Checkpointing -----------------------------------------------------
# Default 1875 = ~25% of an epoch at ~60k samples / effective batch 8.
# Recompute for your data size: SAVE_STEPS = (n_samples / eff_batch) / 4
SAVE_STEPS="${SAVE_STEPS:-1875}"
SAVE_LIMIT="${SAVE_LIMIT:-12}"

# ----- Infra -------------------------------------------------------------
DS_CONFIG="${DS_CONFIG:-scripts/zero2.json}"
MASTER_PORT="${MASTER_PORT:-43001}"

set -o pipefail   # so a torchrun failure isn't masked by the tee pipe
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"   # was a no-op typo (python3WARNINGS); mutes torch/transformers warnings
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p ./ckpt

echo "[train_lora] RUN_NAME=$RUN_NAME  PE_TYPE=$PE_TYPE  DATA_YAML=$DATA_YAML"
echo "[train_lora] EPOCHS=$EPOCHS  EFF_BATCH=$((PER_DEVICE_BS * GRAD_ACCUM))  FRAMES=$NUM_FRAMES  SAMPLING=$SAMPLING"
echo "[train_lora] LORA r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT lr=$LR_LORA  projector_lr=$LR_PROJECTOR"
echo "[train_lora] save_steps=$SAVE_STEPS  save_limit=$SAVE_LIMIT  ds=$DS_CONFIG"

torchrun --nnodes=1 --nproc_per_node=1 --master_port "$MASTER_PORT" \
    llava/train/train_3d.py \
    --deepspeed "$DS_CONFIG" \
    --model_name_or_path "$PREV_STAGE_CHECKPOINT" \
    --version qwen_1_5 \
    --data_path "$DATA_YAML" \
    --image_folder data \
    --video_folder data \
    --embodiedscan_folder data/embodiedscan/ \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),(1x2),(1x3),(1x4),(1x5),(1x6),(2x2),(2x3),(2x4),(2x5),(2x6),(3x3),(3x4),(3x5),(3x6),(4x4),(4x5),(4x6),(5x5),(5x6),(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --attn_implementation "${ATTN_IMPL:-flash_attention_2}" \
    --run_name "$RUN_NAME" \
    --output_dir "./ckpt/$RUN_NAME" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_LIMIT" \
    --learning_rate "$LR_LORA" \
    --mm_projector_lr "$LR_PROJECTOR" \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing "${GRAD_CKPT:-True}" \
    --dataloader_num_workers "${DATALOADER_WORKERS:-1}" \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --lora_enable True \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --mm_tunable_parts "mm_mlp_adapter" \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type "$PE_TYPE" \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy "$SAMPLING" \
    --frames_upbound "$NUM_FRAMES" \
    --report_to "${REPORT_TO:-none}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "./ckpt/${RUN_NAME}.log"
