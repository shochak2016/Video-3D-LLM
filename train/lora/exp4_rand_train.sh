#!/bin/bash
# Exp 4 LoRA fine-tune with randomized augmentation, tuned for a single B200
# (Blackwell sm_100) + the modernized torch 2.9.1 / nvblox_torch stack.
#
# What this wrapper sets that the generic train_lora.sh does not:
#   - LD_LIBRARY_PATH : the conda env's nvidia/*/lib dirs, so nvblox can dlopen
#                       libnppc.so.12 etc. (the cu12 runtime torch bundles).
#   - PE_TYPE         : exp4-discrete-sin3d -> turns on the voxel-cluster crop
#                       token path (VideoProcessor builds crops; llava_arch
#                       augments the uniform frames with them).
#   - LORA_RAND_AUG=1 : per-sample 50/50 randomization of
#                         * frame sampling : uniform <-> mc
#                         * frames_upbound : 16      <-> 32
#                         * PE reduction   : avg     <-> minmax   (encoder = sin3d)
#   - ATTN_IMPL=sdpa  : skip flash-attn (no prebuilt sm_100 wheel; sdpa is fine).
#   - DS_CONFIG       : zero2 with a CLIENT optimizer (torch AdamW) so DeepSpeed
#                       doesn't JIT-compile FusedAdam against a mismatched nvcc.
#   - EXP4_* knobs    : clustering + crop-budget params (see VideoProcessor).
#
# Smoke test (2 steps, no checkpoint):
#   EXTRA_ARGS="--max_steps 2 --save_strategy no" bash train/lora/exp4_rand_train.sh
#
# Full run: just `bash train/lora/exp4_rand_train.sh`.

set -e
cd "$(dirname "$0")/../.."

# --- nvblox runtime libs (cu12 NPP etc. live under the env's site-packages) ---
SITE_NV="$(python -c 'import site,os;print(os.path.join(site.getsitepackages()[0],"nvidia"))' 2>/dev/null)"
if [ -d "$SITE_NV" ]; then
    EXTRA_LD="$(ls -d "$SITE_NV"/*/lib 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${EXTRA_LD}${LD_LIBRARY_PATH}"
fi

# --- Exp 4 / randomization config ---
export PE_TYPE="${PE_TYPE:-exp4-discrete-sin3d}"
export RUN_NAME="${RUN_NAME:-exp4-lora-rand}"
export DATA_YAML="${DATA_YAML:-train/lora/exp4.yaml}"
export LORA_RAND_AUG="${LORA_RAND_AUG:-1}"
export SAMPLING="${SAMPLING:-mc}"          # base strategy (loads mc tables); randomized per-sample
export NUM_FRAMES="${NUM_FRAMES:-32}"      # upper bound; randomized 16/32 per-sample when RAND_AUG=1

# LoRA rank 8 for fastest (alpha=16 keeps alpha/r=2). Decoder LoRA is ~free here:
# the full 7B backward is already paid to train mm_projector at the input, so the
# adapter only adds tiny rank-8 matmuls. Drop it (LORA_R=0 / freeze) only for a
# deliberate projector-only ablation, not for speed.
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"

# --- Quiet the noisy-but-benign logs ---
# nvblox uses glog; minloglevel=2 mutes its per-scene INFO/WARNING chatter
# (LayerCake / GPUHashImpl / "Allocating a new cached image" ...).
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"   # torch use_reentrant / requires_grad / transformers FutureWarnings

# --- B200 / modern-stack infra ---
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export DS_CONFIG="${DS_CONFIG:-scripts/zero2_clientoptim.json}"
export REPORT_TO="${REPORT_TO:-none}"
# With the geometry cache pre-built (run train/lora/exp4_precache.sh ONCE first),
# the data path only reads cache + rebuilds crops on CPU -- no CUDA -- so workers>0
# is safe and overlaps data prep with training. If a needed combo is NOT cached,
# the worker raises a clear error (rather than hanging) telling you to pre-cache.
# Set DATALOADER_WORKERS=0 to allow on-the-fly clustering without pre-caching.
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# --- Exp 4 clustering / crop budget (VideoProcessor reads these) ---
export EXP4_VOXEL_SIZE="${EXP4_VOXEL_SIZE:-0.1}"
export EXP4_EPS="${EXP4_EPS:-0.15}"
export EXP4_MIN_SAMPLES="${EXP4_MIN_SAMPLES:-5}"
export EXP4_TOP_K_PER_FRAME="${EXP4_TOP_K_PER_FRAME:-2}"
export EXP4_MAX_CROPS="${EXP4_MAX_CROPS:-24}"
# Crop selection: "area" (per-frame top-k) | "angular" (per-cluster azimuth-diverse views)
export EXP4_VIEW_SELECT="${EXP4_VIEW_SELECT:-area}"
export EXP4_N_VIEWS="${EXP4_N_VIEWS:-3}"
export EXP4_CACHE_DIR="${EXP4_CACHE_DIR:-data/exp4_cache}"

echo "[exp4_rand] PE_TYPE=$PE_TYPE RAND_AUG=$LORA_RAND_AUG ATTN=$ATTN_IMPL DS=$DS_CONFIG"
echo "[exp4_rand] LD_LIBRARY_PATH has $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -c nvidia) nvidia dirs"

exec bash "$(dirname "$0")/train_lora.sh"
