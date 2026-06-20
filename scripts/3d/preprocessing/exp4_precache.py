"""Offline Exp 4 clustering pre-cache.

Clusters every scene in the training mix once, for every (sampling, frames)
combo the randomized recipe can pick, and writes the lightweight geometry cache
to EXP4_CACHE_DIR. After this runs, training can use DATALOADER_WORKERS>0 (the
data path only reads cache + rebuilds crops on CPU -- no CUDA in workers).

Usage (via train/lora/exp4_precache.sh, which sets LD_LIBRARY_PATH / GLOG etc.):
    python scripts/3d/preprocessing/exp4_precache.py --data_yaml train/lora/exp4.yaml

Honors the same EXP4_* / sampling env vars as training so the cache keys match.
"""
import argparse
import json
import os
import re

import yaml
from tqdm import tqdm

from llava.video_utils import VideoProcessor


def collect_scene_ids(data_yaml):
    """Unique 'video' ids (scannet/sceneXXXX_YY) across the data-mix JSON files."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    scenes = set()
    for entry in cfg["datasets"]:
        with open(entry["json_path"]) as jf:
            for item in json.load(jf):
                vid = item.get("video")
                if vid and vid.startswith("scannet"):
                    scenes.add(vid)
    return sorted(scenes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", default="train/lora/exp4.yaml")
    ap.add_argument("--samplings", default="uniform,mc", help="comma list, matches LORA_RAND_AUG choices")
    ap.add_argument("--frames", default="16,32", help="comma list of frames_upbound")
    args = ap.parse_args()

    samplings = args.samplings.split(",")
    frames_list = [int(x) for x in args.frames.split(",")]

    # mc base strategy so the max-coverage tables load; exp4 PE type turns on
    # the clustering params. force_sample matches training (uniform path).
    vp = VideoProcessor(
        video_folder="data",
        frame_sampling_strategy="mc",
        world_position_embedding_type="exp4-discrete-sin3d",
    )

    scenes = collect_scene_ids(args.data_yaml)
    print(f"[precache] {len(scenes)} scenes x {len(samplings)} sampling x {len(frames_list)} frames "
          f"= {len(scenes) * len(samplings) * len(frames_list)} combos -> {vp.exp4_cache_dir}", flush=True)

    built = skipped = failed = 0
    pbar = tqdm(scenes, desc="exp4 precache")
    for vid in pbar:
        if vid not in vp.scene:
            failed += 1
            continue
        for samp in samplings:
            for fr in frames_list:
                try:
                    if "mc" in samp:
                        frame_files = vp.sample_frame_files_mc(vid, frames_upbound=fr)
                    else:
                        frame_files = vp.sample_frame_files(vid, force_sample=True, frames_upbound=fr)
                    if vp.ensure_exp4_cache(vid, frame_files, samp):
                        built += 1
                    else:
                        skipped += 1
                except Exception as e:  # noqa: BLE001 -- keep going, report at end
                    failed += 1
                    tqdm.write(f"[precache] FAIL {vid} {samp}/{fr}: {type(e).__name__}: {e}")
        pbar.set_postfix(built=built, skipped=skipped, failed=failed)

    print(f"[precache] done: built={built} skipped(existing)={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
