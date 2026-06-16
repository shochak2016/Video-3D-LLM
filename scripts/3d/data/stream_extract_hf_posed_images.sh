#!/usr/bin/env bash
# Re-build data/scannet/posed_images from Hugging Face split archives without needing
# a second full-sized copy on disk:
#   (download one ~10GiB shard → cat stdout → discard shard) × 5  →  gzip -dc → tar xf
#
# Prerequisites: huggingface-cli (`pip install huggingface_hub[cli]` or repo venv).
# Dataset: https://huggingface.co/datasets/zd11024/Video-3D-LLM_data
#
# Usage (from repo root):
#   bash scripts/3d/data/stream_extract_hf_posed_images.sh [DEST_PARENT]
#
# DEST_PARENT defaults to "$(pwd)/data/scannet" so extracted paths are DEST_PARENT/posed_images/...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DEST="${1:-${ROOT}/data/scannet}"
HF_REPO="zd11024/Video-3D-LLM_data"
PARTS=(
  posed_images_part_aa posed_images_part_ab posed_images_part_ac
  posed_images_part_ad posed_images_part_ae
)

cd "$ROOT"
if [[ ! -x .venv310/bin/hf ]] && [[ ! -x .venv/bin/hf ]]; then
  HF_BIN="$(command -v hf || true)"
else
  HF_BIN="${ROOT}/.venv310/bin/hf"
  [[ -x "$HF_BIN" ]] || HF_BIN="${ROOT}/.venv/bin/hf"
fi
[[ -n "${HF_BIN}" && -x "$HF_BIN" ]] || { echo "Install hf CLI or use repo .venv310"; exit 1; }

mkdir -p "$DEST"
shard_dir="/tmp/pose_shard_dl_$$"
trap 'rm -rf "$shard_dir"' EXIT

echo "Streaming ${#PARTS[@]} shards → gzip → tar into ${DEST}" >&2
(
  set -o pipefail
  for part in "${PARTS[@]}"; do
    rm -rf "$shard_dir"
    mkdir -p "$shard_dir"
    echo "Downloading ${part}" >&2
    HF_HUB_DISABLE_PROGRESS_BARS=1 "${HF_BIN}" download "$HF_REPO" "$part" \
      --repo-type dataset --local-dir "$shard_dir" >/dev/null
    f="${shard_dir}/${part}"
    [[ -s "$f" ]] || { echo "Missing or empty shard: $f" >&2; exit 12; }
    cat "$f"
    rm -f "$f"
  done
  echo "All shards streamed; extracting tar archive..." >&2
) | gzip -dc | tar xf - -C "$DEST"

echo "Done. posed_images tree: ${DEST}/posed_images" >&2
