# exp7-b — 8×B200 training port (torch 2.7 / CUDA 12.8 / Blackwell)

Leaf branch off `exp7` (carries all exp7 DINO-fusion code). Goal: run exp7 (and full-LM
fine-tuning) on 8×B200 80/192GB. Motivation: B200 cheaper + faster for long runs, and its
192GB makes **full 7B fine-tuning** comfortable (vs A100 80GB → forced ZeRO-3 over PCIe).

DO NOT try to build this stack on the current Mithril A100 dev box (root is ~98% full,
GPUs are A100). Execute on the B200 box.

## The forcing chain
B200 = `sm_100` → **CUDA 12.8 + PyTorch ≥2.7** → **transformers ≥~4.50** (torch 2.7 won't
run on the pinned 4.40) → port the fork's Qwen2 overrides + Blackwell flash-attn + bump
peft/deepspeed/accelerate. No B200-capable torch exists near 2.1 — the jump is unavoidable.

## Base environment (do this FIRST — kills most of the pain)
Use an **NVIDIA NGC PyTorch container** (ships torch 2.7+/CUDA 12.8/flash-attn prebuilt for
Blackwell): `nvcr.io/nvidia/pytorch:25.xx-py3` (pick the tag whose torch ≥2.7, cu128).
Building flash-attn for Blackwell by hand is the worst part; the container avoids it.

## Pin changes (pyproject `train` extra)
| pkg | pinned (A100) | B200 target |
|---|---|---|
| torch / torchvision | 2.1.2 / 0.16.2 | ≥2.7 (from container) |
| transformers | 4.40 (git 1c39974a) | ~4.50–5.x (whichever runs the fork; 5.x also gives native DINOv3) |
| flash-attn | 2.5.6 | ≥2.7 / FA3 (from container) |
| deepspeed | 0.14.4 | ≥0.15 |
| peft | 0.4.0 | ≥0.11 |
| accelerate | 0.33.0 | ≥0.34 |
| tokenizers | ~0.15.2 | (transformers-coupled) |

## Code port — file by file
- `llava/model/language_model/llava_qwen.py` — **main work.** Update `forward`,
  `generate`, `prepare_inputs_for_generation` to new Qwen2 signatures: `cache_position`,
  `DynamicCache` (legacy tuple cache removed), possibly `logits_to_keep`. Re-verify the
  `inputs_embeds` injection path (multimodal splice) still lines up with the new generation
  loop + cache.
- `llava/train/llama_flash_attn_monkey_patch.py` — **likely delete / no-op.** It patches
  *LLaMA* attention with old `flash_attn_unpadded_qkvpacked_func`; Qwen2 uses native FA2 via
  `attn_implementation="flash_attention_2"`. Confirm nothing calls
  `replace_llama_attn_with_flash_attn()` in the Qwen path (check `train_mem.py`).
- `llava/model/multimodal_encoder/siglip_encoder.py` — self-contained custom impl; low risk.
  Check `transformers.image_utils` / `modeling_outputs` imports still resolve.
- `llava/model/llava_arch.py` — exp7 fusion is version-agnostic (nn.Linear/torch.cat); should
  need no changes. Verify `prepare_inputs_labels_for_multimodal` cache handling.
- `llava/train/train_3d.py` — peft LoRA save/load API moved (0.4→0.11); check
  `get_peft_state_maybe_zero_3` / adapter save. deepspeed config unchanged (json).

## exp7 DINO simplification on this stack (big win)
transformers 5.x loads **DINOv3 natively via AutoModel** on torch ≥2.7 → the whole
**two-env + precompute + cache** machinery becomes UNNECESSARY. Options:
- (simplest) Add a frozen `DINOv3VisionTower` and compute features **live** in the training
  loop; drop `--dino_feature_dir` + `load_dino_feats`. One env, no cache, no ephemeral-disk
  worry. Fusion module (`dino_fusion`) is unchanged.
- (or) keep the precomputed-cache path — still works, just unnecessary.

## B200 config choices (train script)
- **Full FT**: `--mm_tunable_parts mm_language_model[,mm_mlp_adapter]` (drop `--lora_enable`);
  keep `dino_fusion` trainable (already unfrozen when `spatial_feature_type` set).
- **ZeRO-2 suffices** on 192GB (7B full FT: ~23GB/GPU sharded optim+grad + activations).
  Likely no ZeRO-3 needed → avoids param all-gather. Can raise `per_device_train_batch_size`
  and drop `--gradient_checkpointing` for speed given the memory headroom.
- LoRA path still available (same as A100 script) for cheap ablations.

## Acceptance test (gate before trusting exp7 numbers)
Reproduce **base Video-3D-LLM ScanQA** (avg-discrete-sin3d PE, no DINO) EM/CIDEr within noise
on the new stack. Attention-mask / rope / cache changes can silently regress numerics — this
is the guardrail. Only then run exp7 arms.

## Rough effort
~2–4 focused days (LLaVA+HF-savvy, using the NGC container). ~1 day if the Qwen2 overrides
port cleanly; a week+ if generation/cache changes or numerics fight back. Medium difficulty,
moderate risk — not a config tweak.
