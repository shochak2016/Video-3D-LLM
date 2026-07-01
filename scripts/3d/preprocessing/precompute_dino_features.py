"""Precompute frozen DINOv3 patch features for Experiment 7.

Runs in the DEDICATED DINO env (/home/ubuntu/.venvs/dino: torch>=2.4 + transformers 5.x),
NOT the training env (which is pinned to torch 2.1.2 and cannot load dinov3 via HF).

For each ScanNet scene we take the SAME max-coverage frames the training pipeline samples,
apply the SAME 384px center-crop (via VideoProcessor.preprocess -> the exact frames SigLIP
sees), then run DINOv3 and keep the 196 patch tokens (14x14, row-major) at 768-dim.

Output: one file per scene at <out_dir>/<scene_id>.pt =
    {"frame_ids": [int,...], "feats": fp16 tensor (V, 196, 768), "meta": {...}}
Cached per (scene, frame_id): the crop is deterministic per frame, so the cache is valid
for any frames_upbound (fsr16 is a subset of fsr32).

Alignment: DINOv3-vitb16 @224 -> 14x14=196 patch tokens; SigLIP so400m@384 -> 27x27 ->
get_2dPool(stride2) -> 14x14=196. Both row-major over the same cropped view -> 1:1 add.
"""
import argparse
import importlib.util
import os
import traceback

import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def load_video_processor(path="llava/video_utils.py"):
    # Load video_utils.py standalone: importing the llava package would pull in the base
    # model (needs the 2.1.2 env). video_utils only depends on cv2/numpy/torch/PIL/transformers.
    spec = importlib.util.spec_from_file_location("v3d_video_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VideoProcessor


class StubImageProcessor:
    # VideoProcessor.preprocess only reads image_processor.crop_size["width"].
    crop_size = {"width": 384, "height": 384}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--frames_upbound", type=int, default=32)
    ap.add_argument("--frame_sampling_strategy", default="mc")
    ap.add_argument("--strategy", default="center_crop")
    ap.add_argument("--model_id", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--out_dir", default="/mnt/local/dino_features/dinov3-vitb16")
    ap.add_argument("--video_folder", default="data")
    ap.add_argument("--embodiedscan_folder", default="data/embodiedscan/")
    ap.add_argument("--voxel_size", type=float, default=0.1)
    ap.add_argument("--scene_list", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap #scenes (for testing)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    VideoProcessor = load_video_processor()
    vp = VideoProcessor(
        args.video_folder,
        annotation_dir=args.embodiedscan_folder,
        voxel_size=args.voxel_size,
        frame_sampling_strategy=args.frame_sampling_strategy,
    )
    ip = StubImageProcessor()

    scene_list = args.scene_list or f"scripts/3d/preprocessing/scannet_metadata/scannetv2_{args.split}.txt"
    with open(scene_list) as f:
        scenes = [f"scannet/{ln.strip()}" for ln in f if ln.strip()]
    if args.limit:
        scenes = scenes[: args.limit]

    dev = "cuda"
    model = AutoModel.from_pretrained(args.model_id, dtype=torch.float32).eval().to(dev)
    nreg = model.config.num_register_tokens
    proc = AutoImageProcessor.from_pretrained(args.model_id)

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "model": args.model_id,
        "strategy": args.strategy,
        "frames_upbound": args.frames_upbound,
        "frame_sampling_strategy": args.frame_sampling_strategy,
        "n_register_tokens": nreg,
    }

    n_ok, n_skip, n_fail = 0, 0, 0
    for vid in tqdm(scenes, desc=f"dino/{args.split}"):
        scene_id = vid.split("/")[-1]
        out_path = os.path.join(args.out_dir, f"{scene_id}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue
        if vid not in vp.scene:
            print(f"[skip] no scene metadata: {vid}")
            n_skip += 1
            continue
        try:
            frame_files = vp.sample_frame_files_mc(vid, frames_upbound=args.frames_upbound)
            pre = vp.preprocess(vid, ip, force_sample=True,
                                frames_upbound=args.frames_upbound, strategy=args.strategy)
            images = pre["images"]  # list of 384 PIL, same deterministic order as frame_files
            assert len(images) == len(frame_files), (len(images), len(frame_files))
            frame_ids = [int(os.path.basename(fp).split(".")[0]) for fp in frame_files]

            px = proc(images=images, return_tensors="pt")["pixel_values"].to(dev)
            chunks = []
            with torch.no_grad():
                for i in range(0, px.shape[0], args.batch_size):
                    lhs = model(pixel_values=px[i:i + args.batch_size]).last_hidden_state
                    chunks.append(lhs[:, 1 + nreg:, :].to(torch.float16).cpu())  # drop CLS + regs
            feats = torch.cat(chunks, 0)  # (V, 196, 768)
            assert feats.shape[1:] == (196, 768), feats.shape
            torch.save({"frame_ids": frame_ids, "feats": feats, "meta": meta}, out_path)
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f"[FAIL] {vid}")
            traceback.print_exc()

    print(f"\ndone: ok={n_ok} skip={n_skip} fail={n_fail} -> {args.out_dir}")


if __name__ == "__main__":
    main()
