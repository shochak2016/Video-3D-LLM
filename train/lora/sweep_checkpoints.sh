#!/bin/bash
# Eval every saved LoRA checkpoint of a run on a small ScanQA + Scan2Cap subset.
# Generic: works for any LoRA run produced by train/lora/train_lora.sh (or wrappers).
#
# Outputs:
#   results/sweep/<RUN_NAME>/<checkpoint_id>_{scanqa,scan2cap}.log + .jsonl
#   plus a final summary table to stdout.
#
# Usage:
#   RUN_NAME=exp4-lora bash train/lora/sweep_checkpoints.sh
#   RUN_NAME=exp4-lora N=200 TASKS="scanqa" bash train/lora/sweep_checkpoints.sh

set -e
cd "$(dirname "$0")/../.."

RUN_NAME="${RUN_NAME:-exp4-lora}"
BASE_MODEL="${BASE_MODEL:-data/models/LLaVA-Video-7B-Qwen2}"
N="${N:-200}"                                # samples per task
SAMPLING="${SAMPLING:-mc}"
FRAMES="${FRAMES:-16}"
TASKS="${TASKS:-scanqa scan2cap}"

CKPT_DIR="./ckpt/${RUN_NAME}"
OUT_DIR="results/sweep/${RUN_NAME}"
mkdir -p "$OUT_DIR"

CKPTS=$(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V)
if [ -z "$CKPTS" ]; then
    echo "No checkpoints found under $CKPT_DIR"; exit 1
fi

source .venv310/bin/activate
N_CKPTS=$(echo "$CKPTS" | wc -l)
echo "Sweeping $N_CKPTS checkpoints across tasks: $TASKS"
echo "Samples per task: $N"
echo

for ckpt in $CKPTS; do
    cid=$(basename "$ckpt")          # e.g. checkpoint-1875
    for task in $TASKS; do
        LOG="${OUT_DIR}/${cid}_${task}.log"
        JSONL="${OUT_DIR}/${cid}_${task}.jsonl"
        echo "=== $cid | $task ==="
        CUDA_VISIBLE_DEVICES=0 python3 "llava/eval/model_${task}.py" \
            --model-path "$BASE_MODEL" \
            --lora-path "$ckpt" \
            --video-folder ./data \
            --embodiedscan-folder data/embodiedscan \
            --n_gpu 1 \
            --frame_sampling_strategy "$SAMPLING" \
            --max_frame_num "$FRAMES" \
            --test_size "$N" \
            --question-file "data/processed/${task}_val_llava_style.json" \
            --conv-mode qwen_1_5 \
            --answer-file "$JSONL" \
            --overwrite_cfg true > "$LOG" 2>&1 || echo "  (model eval failed)"
        python3 "llava/eval/eval_${task}.py" --input-file "$JSONL" \
            >> "$LOG" 2>&1 || echo "  (scoring failed)"
    done
done

echo
echo "================== SUMMARY =================="
printf "%-25s %-10s %-8s %-8s %-8s %-8s\n" "checkpoint" "task" "CIDER" "BLEU-1" "ROUGE" "EM"
for ckpt in $CKPTS; do
    cid=$(basename "$ckpt")
    for task in $TASKS; do
        LOG="${OUT_DIR}/${cid}_${task}.log"
        cider=$(grep -oE "CIDER: [0-9.]+" "$LOG" | head -1 | awk '{print $2}')
        bleu1=$(grep -oE "BLEU: [0-9.]+" "$LOG" | head -1 | awk '{print $2}')
        rouge=$(grep -oE "Rouge: [0-9.]+" "$LOG" | head -1 | awk '{print $2}')
        em=$(grep -oE "EM: [0-9.]+" "$LOG" | head -1 | awk '{print $2}')
        printf "%-25s %-10s %-8s %-8s %-8s %-8s\n" "$cid" "$task" "${cider:-N/A}" "${bleu1:-N/A}" "${rouge:-N/A}" "${em:-N/A}"
    done
done
