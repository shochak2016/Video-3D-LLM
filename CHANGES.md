# Recent changes

This note summarizes edits made during the ScanQA / EmbodiedScan setup work session. For **git-tracked** code edits, see the sections below; for environment and data placement, see **Operational notes**.

## New helper script (tracked)

### `scripts/3d/data/stream_extract_hf_posed_images.sh`

- Downloads each HF **`posed_images_part_*`** shard into **`/tmp`**, **`cat`**s shard bytes **only on stdout**, pipes **`gzip -dc | tar xf -`** into **`data/scannet/`** (or a custom destination). This avoids **`cat`**ing all shards into an extra **`posed_images.tar.gz`** on disk (saves tens of gigabytes vs. concatenate-then-decompress workflows).
- **Important**: Anything printed to stdout from the downloader (or stray `echo` without **`>&2`**) will corrupt gzip and silently break **`tar`**; **`hf download … >/dev/null`** is required while **`cat`** is the sole stdout producer.

## Code changes (tracked)

### `llava/eval/model_scanqa.py`

- **Question subsampling**: After loading `--question-file`, if `--test_size` is smaller than the number of questions, the loader keeps a random subset of that size (**seed fixed to `42`**) so runs are repeatable.
- **Existing flag wired up**: `--test_size` was already defined (default `10000000`) but unused; it now controls subsampling before Ray work is scheduled.

### `scripts/3d/eval/eval_scanqa.sh`

- **Fourth script argument**: `TEST_SIZE=${4:-10000000}` is passed through as `--test_size` so you can run, for example:

  ```bash
  bash scripts/3d/eval/eval_scanqa.sh <checkpoint_dir_name> uniform 32 100
  ```

  Omit the fourth argument to evaluate on **all** questions in the ScanQA JSON (subject to `--test_size` default).

- **Single-GPU default**: The launcher uses `CUDA_VISIBLE_DEVICES=0` and `--n_gpu 1`. To use eight GPUs again, restore the previous `CUDA_VISIBLE_DEVICES` / `--n_gpu 8` block from git history.

## Operational notes (not enforced by git)

These steps were performed on the workspace to unblock evaluation; they are **local data layout** and tooling, not source patches.

### EmbodiedScan metadata

- **`embodiedscan_infos_{train,val,test}.pkl`** must live under **`data/embodiedscan/`** when using the default `--embodiedscan-folder data/embodiedscan` so `VideoProcessor` can open:

  `{annotation_dir}/embodiedscan_infos_{split}.pkl`

### ScanNet posed images

- Inference loads RGB (`.jpg`), depth (`.png`), and pose (`.txt`) under **`data/scannet/posed_images/<scene_id>/`** (paths come from EmbodiedScan metadata `img_path` joined with `--video-folder`, typically `./data`).
- The **HF dataset** [`zd11024/Video-3D-LLM_data`](https://huggingface.co/datasets/zd11024/Video-3D-LLM_data) provides split gzip members **`posed_images_part_aa`** … **`posed_images_part_ae`**. Concatenating those bytes yields a gzip stream compatible with **`gzip -dc | tar xf -`**.
- **Recommended**: `bash scripts/3d/data/stream_extract_hf_posed_images.sh`

#### Corrupt / zero-byte depth or RGB (“cannot identify image file … .png”)

- If **`tar`** was interrupted (**disk full**) or **`stdout`** picked up downloader/progress noise, **`posed_images/[scene]`** may contain **`0`**-byte **`.jpg` / `.png`**. Pillow then raises **`UnidentifiedImageError`**.
- **Fix**: Remove the bad tree and re-extract cleanly, e.g. `rm -rf data/scannet/posed_images` then run **`stream_extract_hf_posed_images.sh`**. Sanity check:

  ```bash
  ls -la data/scannet/posed_images/scene0081_00/00000.png   # non-zero size
  ```

### `VideoProcessor` ancillary files

- On init, **`data/metadata/scannet_{train,val}_{gt,pred}_box.json`** are loaded for object boxes (`scan2obj`). Ensure these files exist from the repo’s preprocessing or HF metadata bundle before running inference.

### Troubleshooting checklist

- If **`eval_scanqa.py`** fails in CIDER with **`max() arg is an empty sequence`**, **`model_scanqa.py`** often exited early (e.g. missing data) and left **empty** **`results/scanqa/<name>.jsonl`**. Remove that file **or** use a new `--answer-file` path; **`model_scanqa.py` refuses to overwrite** an existing answer file.

## How to revert

```bash
git restore llava/eval/model_scanqa.py scripts/3d/eval/eval_scanqa.sh
```

Remove or relocate local `data/` contents as needed; they are normally untracked.
