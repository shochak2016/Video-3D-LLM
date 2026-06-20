#!/bin/bash
# Offline Exp 4 clustering pre-cache (single process, GPU). Run this ONCE before
# training so the per-step nvblox clustering bottleneck is gone and you can train
# with DATALOADER_WORKERS>0.
#
# It clusters every training scene x {uniform,mc} x {16,32 frames} and writes the
# small geometry cache to EXP4_CACHE_DIR (~50 KB/scene; ~200 MB total). The crop
# pixels are rebuilt cheaply on CPU at train time, so they are NOT cached.
#
# Usage (from repo root):
#   bash train/lora/exp4_precache.sh
# Honors the same EXP4_* env knobs as training (defaults below match the wrapper).

set -e
cd "$(dirname "$0")/../.."

# nvblox runtime libs + quiet logs (same as the train wrapper).
SITE_NV="$(python -c 'import site,os;print(os.path.join(site.getsitepackages()[0],"nvidia"))' 2>/dev/null)"
if [ -d "$SITE_NV" ]; then
    export LD_LIBRARY_PATH="$(ls -d "$SITE_NV"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH}"
fi
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

# Clustering / crop-budget knobs (MUST match training for cache keys to hit).
export EXP4_VOXEL_SIZE="${EXP4_VOXEL_SIZE:-0.1}"
export EXP4_EPS="${EXP4_EPS:-0.15}"
export EXP4_MIN_SAMPLES="${EXP4_MIN_SAMPLES:-5}"
export EXP4_TOP_K_PER_FRAME="${EXP4_TOP_K_PER_FRAME:-2}"
export EXP4_MAX_CROPS="${EXP4_MAX_CROPS:-24}"
export EXP4_VIEW_SELECT="${EXP4_VIEW_SELECT:-area}"
export EXP4_N_VIEWS="${EXP4_N_VIEWS:-3}"
export EXP4_CACHE_DIR="${EXP4_CACHE_DIR:-data/exp4_cache}"

DATA_YAML="${DATA_YAML:-train/lora/exp4.yaml}"
echo "[exp4_precache] data_yaml=$DATA_YAML cache=$EXP4_CACHE_DIR max_crops=$EXP4_MAX_CROPS"

exec python scripts/3d/preprocessing/exp4_precache.py \
    --data_yaml "$DATA_YAML" \
    --samplings "${PRECACHE_SAMPLINGS:-uniform,mc}" \
    --frames "${PRECACHE_FRAMES:-16,32}"
