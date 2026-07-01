"""Push ONLY the final shareable weights of an exp7 run to the Hugging Face Hub.

Uploads the LoRA adapter + the fusion MLP (non_lora_trainables.bin) + configs.
Explicitly SKIPS the multi-GB DeepSpeed optimizer states (global_step*/, optimizer*,
zero_to_fp32.py, rng/scheduler state) so the repo stays small.

Usage:
  python scripts/3d/push_final_to_hub.py \
      --ckpt_dir ckpt/exp7-scanqa-dinov3-fuse \
      --repo_id sho16/exp7-scanqa-dinov3-fuse \
      [--private]
"""
import argparse
import ast
import os
import re

from huggingface_hub import HfApi, create_repo


def final_train_loss(log_path):
    """Pull the final train_loss from the tee'd training log, if present."""
    if not log_path or not os.path.exists(log_path):
        return None
    val = None
    for line in open(log_path, errors="ignore"):
        m = re.search(r"\{.*'train_loss':.*\}", line)
        if m:
            try:
                val = float(ast.literal_eval(m.group(0))["train_loss"])
            except Exception:
                pass
    return val


def build_model_card(repo_id, log_path):
    loss = final_train_loss(log_path)
    loss_line = f"- Final train loss: **{loss:.4f}**\n" if loss is not None else ""
    return f"""---
base_model: lmms-lab/LLaVA-Video-7B-Qwen2
library_name: peft
tags:
- video-3d-llm
- scanqa
- dinov3
- lora
---

# {repo_id.split('/')[-1]}

Experiment 7 — **DINOv3 spatial features replace the 3D positional encoding** in
Video-3D-LLM. Frozen per-token DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`,
196 patch tokens) are fused into the pooled SigLIP grid at the old PE site via a
zero-init residual MLP, so the model starts exactly at "base minus PE".

## Contents
- `adapter_model.bin` + `adapter_config.json` — LoRA (r=128, α=256) on the Qwen2-7B LLM.
- `non_lora_trainables.bin` — the full-Adam `dino_fusion` MLP.

## Training
- Base: `lmms-lab/LLaVA-Video-7B-Qwen2`; SigLIP + projector frozen, LLM via LoRA.
- Data: ScanQA (26,515 samples), 3 epochs, lr 2e-4 cosine, DeepSpeed ZeRO-2, 8×H200.
{loss_line}
## Load
```python
from peft import PeftModel
# base = load LLaVA-Video-7B-Qwen2 with spatial_feature_type='dinov3-concat'
model = PeftModel.from_pretrained(base, "{repo_id}")
# also load non_lora_trainables.bin into the model's dino_fusion module
```

## Eval
ScanQA (EM, CIDEr) primary; SQA3D (test) secondary — see `scripts/3d/eval/`.
Metrics: _TODO — fill after running eval._
"""


def write_card(ckpt_dir, repo_id, log_path):
    card = build_model_card(repo_id, log_path)
    path = os.path.join(ckpt_dir, "README.md")
    with open(path, "w") as f:
        f.write(card)
    print(f"Wrote model card -> {path}")

# Whitelist: only the small artifacts needed to reload the model for eval.
ALLOW = [
    "README.md",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "non_lora_trainables.bin",   # the dino_fusion MLP (eval loads this by name)
    "config.json",
    "generation_config.json",
    "*.json",                    # tokenizer/config jsons
    "tokenizer.model",
    "*.txt",
]
# Belt-and-suspenders: never upload optimizer/checkpoint bulk even if it matches above.
# checkpoint-*/** ignores the intermediate-checkpoint subdirs wholesale (the "*.json"/"*.txt"
# ALLOW globs would otherwise recurse into them and duplicate tokenizers).
IGNORE = [
    "checkpoint-*/**",
    "global_step*/**",
    "checkpoint-*/global_step*/**",
    "*optimizer*",
    "*scheduler*",
    "rng_state*",
    "zero_to_fp32.py",
    "latest",
    "*.pt",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="ckpt/exp7-scanqa-dinov3-fuse")
    ap.add_argument("--repo_id", required=True, help="e.g. sho16/exp7-scanqa-dinov3-fuse")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--log", default="ckpt/exp7-scanqa-dinov3-fuse.log",
                    help="training log used to fill the model card's final loss")
    ap.add_argument("--no_card", action="store_true", help="skip regenerating README.md")
    args = ap.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        raise SystemExit(f"ckpt_dir not found: {args.ckpt_dir}")

    if not args.no_card:
        write_card(args.ckpt_dir, args.repo_id, args.log)

    # Show what will actually be uploaded before doing it.
    api = HfApi()
    print(f"Creating/using repo: {args.repo_id} (private={args.private})")
    create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")

    print(f"Uploading small weights from {args.ckpt_dir} ...")
    api.upload_folder(
        folder_path=args.ckpt_dir,
        repo_id=args.repo_id,
        repo_type="model",
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        commit_message="exp7 final LoRA + dino_fusion weights",
    )
    print(f"Done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
