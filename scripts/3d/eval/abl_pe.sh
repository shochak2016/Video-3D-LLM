#!/bin/bash
# 3-way PE ablation on scanqa val:
#   avg-discrete-sin3d  |  exp3-discrete-sin3d  |  none
# Patches the checkpoint's world_position_embedding_type in-place before each run,
# renames the output so the three runs don't overwrite each other, and prints
# diagnostic + scoring lines plus a pairwise file-identity matrix at the end.
#
# Edit the four knobs below to retarget.

set -e
cd "$(dirname "$0")/../../.."
source .venv310/bin/activate

# ---- knobs (all env-overridable, e.g. TEMP=0.7 TAG=_t07 bash abl_pe.sh) --
CKPT="${CKPT:-llava_qwen_scanqa_init_res}"
SAMPLING="${SAMPLING:-mc}"      # uniform | mc | mc-ratio90 | mc-ratio95
FRAMES="${FRAMES:-16}"          # max_frame_num
N="${N:-50}"                    # test_size
TEMP="${TEMP:-0}"               # generation temperature (5th arg to eval_scanqa.sh)
TAG="${TAG:-}"                  # appended to output filename, e.g. "_t07"
MODES_STR="${MODES:-avg-discrete-sin3d exp3-discrete-sin3d none}"
read -ra MODES <<< "$MODES_STR"
# --------------------------------------------------------------------------

set_pe () {
  python3 -c "
import json, pathlib
p = pathlib.Path('ckpt/${CKPT}/config.json')
c = json.loads(p.read_text())
c['world_position_embedding_type'] = '$1'
p.write_text(json.dumps(c, indent=2))
print('config.json ->', c['world_position_embedding_type'])
"
}

OUT="results/scanqa/${CKPT}_${SAMPLING}_fsr${FRAMES}.jsonl"

# clear any stale per-mode outputs
rm -f "$OUT"
for mode in "${MODES[@]}"; do
    suffix=$(echo "$mode" | sed 's/-discrete-sin3d//')
    rm -f "results/scanqa/${CKPT}_${SAMPLING}_fsr${FRAMES}_${suffix}${TAG}.jsonl"
done

# run each mode
for mode in "${MODES[@]}"; do
    set_pe "$mode"
    suffix=$(echo "$mode" | sed 's/-discrete-sin3d//')
    LOG="results/scanqa/abl_${mode}${TAG}.log"
    bash scripts/3d/eval/eval_scanqa.sh "$CKPT" "$SAMPLING" "$FRAMES" "$N" "$TEMP" > "$LOG" 2>&1
    mv "$OUT" "results/scanqa/${CKPT}_${SAMPLING}_fsr${FRAMES}_${suffix}${TAG}.jsonl" 2>/dev/null || true
done

# summary
echo
echo "================================= RESULTS ================================="
for mode in "${MODES[@]}"; do
    suffix=$(echo "$mode" | sed 's/-discrete-sin3d//')
    echo
    echo "=== $suffix ==="
    grep -E "\[pe_check\]|\[exp3\]|\[dispatch\]|^CIDER|^BLEU|^METEOR|^Rouge|^EM" \
        "results/scanqa/abl_${mode}${TAG}.log"
done

echo
echo "================================= FILE IDENTITY ================================="
suffixes=()
for mode in "${MODES[@]}"; do
    suffixes+=("$(echo "$mode" | sed 's/-discrete-sin3d//')")
done
for ((i=0; i<${#suffixes[@]}; i++)); do
    for ((j=i+1; j<${#suffixes[@]}; j++)); do
        a="${suffixes[$i]}"; b="${suffixes[$j]}"
        fa="results/scanqa/${CKPT}_${SAMPLING}_fsr${FRAMES}_${a}${TAG}.jsonl"
        fb="results/scanqa/${CKPT}_${SAMPLING}_fsr${FRAMES}_${b}${TAG}.jsonl"
        cmp "$fa" "$fb" >/dev/null 2>&1 && echo "$a == $b (identical)" || echo "$a != $b (differ)"
    done
done
