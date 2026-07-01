# Experiment 7 — DINOv3 spatial features in place of 3D positional encoding

**Branch:** `exp7` (fresh from `main` @ 3ff70e8; none of the exp3/exp4 code).
**Status:** design locked, not yet implemented.

## Hypothesis
The base Video-3D-LLM 3D positional encoding (PE) contributes very little (small EM /
CIDEr gain). The metric-3D PE may be a weak way to inject spatial structure. DINOv3
(the backbone behind Depth Anything) produces dense features that encode spatial /
geometric relationships between objects. Replacing PE with per-token DINOv3 features
should convey spatial/positional information more effectively.

This tests: **are learned 2D relational features a stronger spatial signal than explicit
metric-3D PE?**

## Scientific caveat (keep in mind when reading results)
DINOv3 features are relational/semantic *2D* features — they are NOT metric 3D
coordinates. The base PE injects actual world coords (depth + camera pose). Removing PE
removes the only ground-truth metric grounding. So a flat/negative result could mean
"DINO doesn't help" OR "metric grounding mattered after all." Planned follow-up arm:
**PE + DINO** together, to separate the two.

## Design (locked)
- **Remove PE.** Delete the PE add at `llava/model/llava_arch.py:515`
  (`image_feat = image_feat + world_position_embedding(coords)`), gated behind the new
  experiment flag so the base path is untouched.
- **Keep SigLIP** vision tower (`google/siglip-so400m-patch14-384`) unchanged, frozen.
- **Add DINOv3** as a second, frozen encoder: `facebook/dinov3-vitb16-pretrain-lvd1689m`
  (already cached). Hidden dim **768**, patch 16.
- **Per-token dense fusion.** Run DINOv3 at **224px → 14×14 = 196 patch tokens**, which
  exactly matches the pooled SigLIP grid (SigLIP 27×27=729 → `get_2dPool` stride-2 →
  14×14=196). Token *i* of DINO aligns to token *i* of SigLIP — no interpolation needed.
- **Fusion location: post-projector (LLM dim 3584).** Drop-in at the PE site:
  ```
  fused = siglip_tokens(196×3584) + fusion_mlp(dino_tokens(196×768))   # fusion_mlp: 768→3584
  ```
  Base `mm_projector` (mlp2x_gelu, 1152→3584→3584) is untouched.
- **Fusion MLP** last layer **zero-init** → at step 0 the DINO term is exactly 0, so the
  model == "base minus PE". Satisfies the degenerate-at-init playbook.

### Data flow (one frame)
```
frame ─SigLIP(frozen)─▶ 729×1152 ─mm_projector─▶ 729×3584 ─2dPool─▶ 196×3584 ┐
                                                                             (+)──▶ 196×3584 ─▶ LLM
frame ─DINOv3@224(frozen)─▶ 196×768 ─fusion_mlp(768→3584, zero-init)─▶ 196×3584 ┘
```

## Training plan
- **LoRA** on the LLM transformer weights.
- **Full Adam** on the fusion MLP (zero-init on the DINO branch).
- SigLIP **and** DINOv3 both frozen. Base mm_projector frozen.
- Filter with 1% val mini-runs before full training (per playbook).
- Compare modes on gain ÷ retraining cost.

## Eval
- Datasets: ScanQA (EM, CIDEr) primary; SQA3D as secondary.
- **Baseline = published `avg` sin3d PE** (the `config.json.bak.avg` config), NOT the
  exp3 variant currently in `ckpt/llava_qwen_scanqa_init_res/config.json`.

## Environment resolution (DONE — step 1)
**Root cause:** DINOv3 via HF `transformers` needs transformers ≥5.x → **torch ≥2.4**, but
the base Video-3D-LLM fork is pinned to **torch 2.1.2** (flash-attn 2.5.6 / deepspeed 0.14.4 /
llava fork). No single env runs both. `.venv310` had been upgraded to transformers 5.12.1,
which silently broke the base model (`torch.distributed.device_mesh` missing, `is_torch_available=False`).

**Decision: two envs + offline precompute of frozen DINOv3 features.**
- **Training/eval env** = `.venv310`, reverted to the pinned **transformers 4.40.0.dev0
  (commit 1c39974a) + tokenizers 0.15.2 + torch 2.1.2**. Base model imports & runs again. ✅
- **DINOv3 env** = `/home/ubuntu/.venvs/dino` (Python 3.10), **torch 2.4.1+cu121 +
  transformers 5.12.1**. Loads `facebook/dinov3-vitb16-pretrain-lvd1689m` via `AutoModel`
  natively (HF token present; fbaipublicfiles direct URL is 403-gated, HF weights are not). ✅

**Verified DINOv3 I/O (step-1 de-risk):**
- Input: 224×224, ImageNet norm `mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225)`,
  bilinear resample, do_rescale+do_normalize (from its `AutoImageProcessor`).
- `last_hidden_state = (B, 201, 768)` = 1 CLS + 4 register + **196 patch tokens**.
- Patch grid = `lhs[:, 1+num_register_tokens:, :]` = `lhs[:, 5:, :]` →
  `(B, 196, 768)` → `reshape(B, 14, 14, 768)`. **Exact match to pooled SigLIP grid.** ✅

## Integration code (DONE — validated on this machine)
SigLIP stays **live** in training (decided: keep flexibility; it has no env conflict, only
DINO must be precomputed). Vision-free (precompute SigLIP too) rejected for now.
- `scripts/3d/preprocessing/precompute_dino_features.py` — runs in `.venvs/dino`; per scene
  emits `(V, 196, 768)` fp16 keyed by frame_id + meta. Validated (2 scenes). NOT run in full
  (training is on another machine). ~294 KB/frame, ~14 GB train+val, ~40 min.
- `llava/video_utils.py`: `preprocess()` now returns `frame_files`; `load_dino_feats(dir,
  scene_id, frame_files)` → `(V,196,768)` aligned to frame order; `merge_video_dict` stacks
  `dino_feats`. (validated: shape + order-preserving)
- `llava/train/train_3d.py`: `ModelArguments.spatial_feature_type` / `dino_feature_dim`,
  `DataArguments.dino_feature_dir`; `__getitem__` attaches `dino_feats`; `get_model` writes
  `spatial_feature_type`/`dino_feature_dim` into model config.
- `llava/model/llava_arch.py`: builds `self.dino_fusion` (Linear→GELU→Linear, **final layer
  zero-init**) when `spatial_feature_type` contains `dinov3`; applies
  `image_feat += dino_fusion(dino_feats[idx])` at the PE site (196×3584 grid).
  (validated: zero-init ⇒ `img + fusion(dino) == img` at step 0)
- `get_2dPool` bilinear stride-2 on 27×27 → `ceil(27/2)=14` → 196 tokens. 1:1 with DINO. ✅

## Fusion variants (A/B) + signal controls
Flag `spatial_feature_type` selects the fusion. BOTH are **residual** with the MLP's final
layer zero-init → at step 0 the added term is exactly 0 (fused == siglip == base minus PE).
Both are 2-layer MLPs (Linear→GELU→Linear→3584); only the MLP INPUT differs:
- `dinov3-concat` (DEFAULT / first run): `fused = siglip + MLP([siglip ; dino])`,
  in_dim 3584+768=4352, ~28.5M params. MLP sees both modalities → can modulate siglip by
  dino. Validated: fused==siglip at init.
- `dinov3-add` (A/B): `fused = siglip + MLP(dino)`, in_dim 768, ~15.6M params. MLP sees only
  dino → cleaner causal attribution (siglip path enters only via the residual). Validated.

**Signal controls (does DINO content actually matter?) — train-time is the valid form:**
- `dino-shuffle` (BEST control): feed REAL dino vectors from the WRONG frames/scenes →
  preserves feature statistics, destroys spatial-semantic correspondence. Accuracy vs real
  isolates whether the *correspondence* carries signal.
- `dino-random`: gaussian noise in place of dino → tests "is it DINO content or just extra
  params?" (if it matches real, content doesn't help).
- `dino-zero`: == base-minus-PE lower bound.
- FREE built-in diagnostic (concat): after training, measure norm of the learned dino columns
  `dino_fusion.weight[:, 3584:]`; if ~0, the model learned to ignore DINO. No extra run needed.
- NOTE: swapping to random/shuffle only at INFERENCE is confounded (OOD to the trained fusion);
  prefer train-time controls.

## Training setup (DONE — validated)
- `scripts/3d/train/scanqa.yaml` — ScanQA-only (26,515 samples, 562 scenes).
- `scripts/3d/train/train_scanqa_dino.sh` — ZeRO-2, LoRA(r128,a256) on LLM, full-Adam
  `dino_fusion`, PE off, `--spatial_feature_type dinov3-fuse`, `--dino_feature_dir`,
  `--frame_sampling_strategy mc --frames_upbound 32`, `--dataloader_num_workers 16`
  (+persistent +prefetch 4), 3 epochs, lr 2e-4.
- `train_3d.py`: `find_all_linear_names` excludes `dino_fusion` (verified: LoRA targets =
  {q_proj,k_proj,...} only); `dino_fusion` explicitly unfrozen (full Adam) in both lora/
  non-lora modes. SigLIP + projector + base LLM frozen (LLM via LoRA).

**INVARIANTS (must hold or cache misses / KeyError):**
- Training `frame_sampling_strategy` + `frames_upbound` MUST equal the precompute's
  (both `mc` / 32) — else sampled frame_ids won't be in the DINO cache.
- Cache is per (scene, frame_id); precompute train AND val (val needed for eval).

**KNOWN CAVEATS / follow-ups:**
- Single LR (2e-4) covers both LoRA and the full-Adam fusion MLP. If the fusion MLP wants a
  different LR, add a `dino_fusion_lr` optimizer group in `llava_trainer` (like mm_projector_lr).
- Hardware note: 8×A100 80GB **PCIe** → ZeRO-2 chosen over ZeRO-3 to avoid per-layer
  param all-gather over PCIe.

## Open / TODO (remaining, on the training machine)
- [ ] Run the full precompute on the training box (`.venvs/dino`, output to `/mnt/local`,
      `--split train` and `--split val`).
- [ ] Eval scripts (`llava/eval/model_scanqa.py` etc.): mirror the `__getitem__` cache load
      (pop `frame_files`, attach `dino_feats`) — currently only the train dataset is wired.
- [ ] 1% val mini-run as go/no-go filter (fusion zero-init ⇒ starts == base-minus-PE).
- [ ] (Optional) Skip the world_coords unproject in `preprocess()` when PE off + no grounding
      — ~halves dataloader CPU, removes 32 depth-file reads/sample. Low priority once GPU-bound.
- [ ] Later arm: PE + DINO (to separate "DINO helps" from "PE was weak").
