### Experiment 1

1. Create voxel map
2. Compute average coord of each voxel cluster (or voxel)
3. Compute positional embedding for each coord above
4. Add (or concat) into final embedding for each patch
5. ("Optional") Use NN/encoder to map

### Experiment 2

1. Create voxel map
2. Create patches the same size as one voxel
3. Compute positional embeddings by using the voxel as X, Y, Z coords
4. Grid search to find best voxel/patch size

### Experiment 3

1. Compute average depth per patch
2. Go pixel by pixel in current frame and assign it to a patch. If pixel depth is close to average depth of the patch its in, place it there. If it is above some threshold (maybe just do KNN here), then assign to different patch
3. Compute positional embeddings

### Experiment 4

Create nonuniform patches by clustering regions of the same depth/object (RGB), and respecitive positional embeddings. Is not compatible with ViT used here (Sigflip v1).

Because SigLIP v1 needs fixed 384² inputs, we don't feed nonuniform patches to the
ViT directly. Instead each voxel cluster is projected into the frames it's visible
in, the cluster's 2D bbox is cropped + resized to 384², and run through SigLIP as a
normal crop. So a cluster becomes a 14×14 token grid whose **per-patch 3D positional
embedding** comes from the cluster's own voxels (boundary/background pixels masked
out, empty patches fall back to the cluster centroid).

**Pipeline** (`llava/exp4.py`):
1. `cluster_scene` — nvblox TSDF over the scene's depth frames → surface voxels → DBSCAN → `VoxelClusters`.
2. `compute_crop_geometry` — per (cluster, frame): 2D bbox + per-patch masked world coords (`(14,14,3)`), capped to a crop budget.
3. `assemble_crop_pixels` — crop + resize RGB at each bbox → SigLIP-normalized `(N,3,384,384)`.
4. Model (`llava/model/llava_arch.py`, `is_exp4` branch) — encode crops → 2D-pool → add sin3d PE from the per-patch coords → **append as extra "frames"** to the uniform video tokens (augment, not replace).

**How Exp 4 should be configured / run**

It runs on the modernized Blackwell (B200) stack: torch 2.9.1+cu128, `nvblox_torch`
wheel `v0.0.10` + `nvidia-npp-cu12`, conda env `video3d`. Two scripts:

```bash
# 1. ONE-TIME: pre-cache the clustering geometry (GPU, single process, ~1h for the
#    562 ScanQA+Scan2Cap scenes x {uniform,mc} x {16,32 frames}). Writes the small
#    geometry cache (~50 KB/scene; crop pixels are rebuilt cheaply on CPU at train time).
bash train/lora/exp4_precache.sh

# 2. TRAIN: randomized-augmentation LoRA, layered on the Exp 4 crop tokens.
bash train/lora/exp4_rand_train.sh
#    (tqdm prints to terminal + tees to ckpt/exp4-lora-rand.log; checkpoints every SAVE_STEPS)
```

Key choices baked into the wrappers (all env-overridable):
- **PE**: `world_position_embedding_type=exp4-discrete-sin3d`. Encoder is **sinusoidal** (`sin3d`). The sin3d module is built with `n_points=2`: uniform-frame PE feeds 2 points (`minmax`→`[min,max]`, `avg`→`[c,c]`), crop PE feeds `[c,c]`.
- **Randomized augmentation** (`LORA_RAND_AUG=1`): per sample, independent 50/50 coin flips on
  - frame sampling `uniform ↔ mc`,
  - frames `16 ↔ 32`,
  - PE reduction `avg ↔ minmax` (uniform frames only — crops always use their per-patch coord).
- **Crop budget**: `EXP4_TOP_K_PER_FRAME=2`, `EXP4_MAX_CROPS=24` (most-covered crops kept). Clustering: `EXP4_VOXEL_SIZE=0.1`, `EXP4_EPS=0.15`, `EXP4_MIN_SAMPLES=5`.
- **Trainable parts**: full `mm_projector` + LoRA on the decoder (`LORA_R=8`, `LORA_ALPHA=16`), vision tower frozen, sin3d PE is parameter-free. Decoder LoRA is ~free (the full backward is already paid to train the projector at the input) and lets the LLM learn to use the new crop/PE tokens — keep it unless running a deliberate projector-only ablation.
- **Infra**: `ATTN_IMPL=sdpa` (no flash-attn wheel for sm_100), `DS_CONFIG=scripts/zero2_clientoptim.json` (client torch AdamW, avoids FusedAdam JIT), `LD_LIBRARY_PATH` includes the env's `nvidia/*/lib` (nvblox needs `libnppc.so.12`), `GLOG_minloglevel=2` + `PYTHONWARNINGS=ignore` to mute nvblox/torch log spam.
- **DataLoader**: after pre-caching, `DATALOADER_WORKERS=4` (crops rebuilt on CPU, no CUDA in workers). Without a pre-cache, nvblox clustering can't run in a forked worker — set `DATALOADER_WORKERS=0` to cluster on the fly in the main process (much slower), or the worker raises a clear "pre-cache first" error.

Cache keys include `(scene, sampling, n_frames, voxel/eps/min_samples, top_k, max_crops)`,
so changing any clustering/budget knob requires re-running the pre-cache (or it
rebuilds on the fly).

**Measured — crop diagnostics (81,779 cached crops):**
- **Aspect distortion is mostly mild.** Median stretch (max/min side ratio) = **1.33**, mean 1.48;
  **85% of crops < 1.5**, only **7.6% exceed 2:1**, 2.6% > 3:1. This sits right where NaViT found the
  aspect-preservation benefit ≈ 0.3% — so **NaFlex is weakly justified** (weeks of work to fix a 7.6%
  tail). Verdict: skip the vendored NaFlex encoder; at most run the free letterbox A/B to confirm.
- **43% of crops are the FULL frame (640×480).** Nearly half the crop-token budget is **whole-frame
  duplicates of the uniform video frames** — zero localization value. Cause: geometry-only DBSCAN
  merges floor+walls+ceiling (all spatially contiguous at the junctions) into one **room-shell
  mega-cluster** that spans every frame. This is the **highest-ROI lever** found so far, and it makes
  angular/coverage selection meaningful (today angular is choosing among views where ~half are
  whole-frame).

**Planned ablations (Exp 4):**
- **DBSCAN parameter sweep — DONE (eps=0.10 fixes the 43% redundancy).** 25-scene cluster-stats
  sweep (voxel=0.1, top_k=2, max_crops=24), measuring full-frame-crop fraction + useful crops/scene:

  | eps | min_samples | full-frame % | useful crops/scene |
  |----|----|----|----|
  | 0.15 | 5 (current) | **42.7%** | 13.0 |
  | 0.12 | 5 | 20.2% | 19.2 |
  | **0.10** | **5** | **0.0%** | 19.9 |
  | 0.15 | 10 | 9.0% | 21.8 |
  | 0.15 | 15 | 0.0% | 19.3 |

  **The room-shell *is* splittable** (my earlier "contiguous, can't split" caveat was wrong): the
  floor↔wall junction is a **perpendicular/diagonal adjacency**, so `eps=0.10` (= voxel; face-neighbors
  only, 0.10 m) **severs the 0.14–0.17 m corner** and breaks the shell into floor/wall/ceiling.
  `min_samples` works differently (sparse junction voxels drop below density → noise → disconnect).
  Both **eliminate the redundancy and raise useful object crops ~13→~20/scene.** Adopt **`eps=0.10`**
  (clean mechanism: keeps same-surface contiguity, breaks perpendicular merges). Nuance: face-only
  connectivity *could* split an object's perpendicular sub-parts (chair seat vs back), but evidence is
  mild (crops/scene rose to ~20 not ~100, median obj distortion stable ~1.35). Degenerate combos
  (eps≤voxel, or min_samples>6) yield 0 clusters — the 6-face-neighbor grid limit. *Next: re-precache
  at eps=0.10 + retrain; surface-normals / RANSAC plane-removal now unnecessary unless part-splitting
  hurts.*
- **Augment vs. replace (full frames + crops → crops-only).** *Current* Exp 4 **augments**: the
  LLM sees the uniform video frames (`V×196` tokens, `V=16/32`) **plus** the cluster crops
  (`N×196`, `N≤max_crops`), concatenated (`llava_arch.py:668`, `torch.cat([uf, cf])`). The ablation
  drops the uniform frames entirely → a **fully object-centric / "clusterized"** representation
  where the scene *is* the set of cluster crops. Tradeoffs: fewer tokens + cleaner single path, but
  **loses scene context** — background, floor/walls, empty space, and the layout *between* objects
  vanish (no whole-frame fallback). That makes the **per-cluster 3D PE load-bearing** (it becomes the
  *only* carrier of spatial layout) and **view coverage a correctness requirement**, not an
  optimization. Mitigations to test: a low-res "context cluster" / a few whole-scene frames, vs. a
  pure object-bag betting that the 3D coords reconstruct layout. This is also the regime where a
  **NaFlex encoder** (no-stretch, native-aspect crops) would be the *sole* encoder path — see Exp 5.
  Wire behind `EXP4_DROP_UNIFORM=1`.
- **Crop fit: stretch vs. letterbox vs. NaFlex** (`EXP4_CROP_FIT = stretch | pad | naflex`). Today
  crops are **anisotropically stretched** to 384² (aspect distorted). `pad` = letterbox to 384² on
  the *existing* SigLIP-v1 + mask padded patches (cheap, no new encoder) — isolates "does removing
  the stretch help?". `naflex` = vendored SigLIP2-NaFlex, native aspect, variable `H_p×W_p` patches
  (the principled fix; see Exp 5). Lit (NaViT +2.8 ImageNet square→aspect; SigLIP2-NaFlex) says the
  payoff scales with how non-square the crops are — and the **measured distribution above (85% < 1.5,
  only 7.6% > 2:1) makes NaFlex weakly justified**. So: run `pad` (free) if curious, but **deprioritize
  the NaFlex encoder** until/unless the crop distribution changes (e.g. after the DBSCAN sweep splits
  the room-shell into many small, more aspect-varied object crops).
- **Angular view-coverage selection** (`EXP4_VIEW_SELECT=angular`, `EXP4_N_VIEWS`) — per-cluster
  azimuth-diverse crop views instead of per-frame top-k by bbox area. **Inference A/B (r16,
  uniform/16, 200 seed-42): angular 96.9 CIDEr vs area 101.2 — angular ~4 worse.** Caveats: model
  *trained* on area (OOD selection swap), and eps=0.15 means angular shuffles views where ~half are
  whole-frame. **BUT the A/B is INVALID — it never tested multi-view.** Empirically the current
  angular selection at `max_crops=12` returns **12 objects × exactly 1 view each, zero multi-angle**:
  the global `(rank, area)` sort is breadth-first ("every cluster gets its primary before any gets a
  2nd/3rd"), so with ≥12 clusters/scene the budget is fully consumed by *primary* views and the
  azimuth-diverse rank-1/2 views never get selected. So "angular" degenerated to "one clearest view
  of 12 objects" — the opposite of the hypothesis. **Re-open, don't bury.**
  **Fix (depth-favoring, = the intended algorithm):** order candidates by bbox-area desc; keep an
  object's clearest view first, then keep additional crops of it *only when a frame sees it from a
  genuinely new azimuth* (`_ang_dist > THRESH`, ~30°), skipping redundant angles, until `max_crops`.
  Self-balances breadth vs depth. Cluster IDs are global (scene-level clustering), so per-cluster
  angular-coverage tracking is trivial. Retest the FIXED selection on eps=0.10 clustering, trained,
  on ScanQA + **SQA3D** (multi-view should help situated/3D reasoning most).
- **32mc + angular** — coverage-best frames (the ablation showed frame count dominates, 32≈+15 pts
  over 16) combined with angular cluster views. Pre-cached; ~28 h on 1 GPU (ckpt-on).
- **RGB-as-tiebreaker clustering** (`EXP4_USE_RGB=1`, sweep `rgb_weight`). Today DBSCAN clusters
  surface voxels on **XYZ only**; adding per-voxel mean RGB should split spatially-touching but
  differently-colored objects (book on table, poster on wall). Caveat: appending scaled RGB to the
  DBSCAN distance is a *local* metric, so it can't distinguish a within-object texture edge from a
  true object boundary (locally identical: small `dXYZ`, large `dRGB`). A weak `rgb_weight` (~0.01)
  helps on mostly-uniform surfaces but **over-segments textured/patterned objects** (posters,
  patterned furniture), and no weight fixes both. Principled version: split a geometric cluster only
  on a *coherent structural* color boundary (two-stage — geometry first, conservative color split,
  `EXP4_RGB_MODE=tiebreak`), or cluster on **lifted semantic features** (DINO/SAM voxels) instead of
  raw color. Cheap first pass: `EXP4_USE_RGB=1`, `rgb_weight≈0.01`.

### Experiment 5

Positional embeddings of pixel distributions within a patch from centroid, via some combination of NN or sinusoidal

**Possibility — VAE distribution embedding.** Instead of collapsing a patch/cluster to
one coord (current Exp 4 = masked-mean → sin3d), feed the patch's set of 3D points into
a small VAE; the latent `z` becomes (part of) the PE, capturing the *shape/spread/orientation*
of the surface region rather than just its center.

- **Input to the VAE**: the per-patch cluster points (XYZ) — encoder is PointNet-style
  (per-point MLP + symmetric pool, permutation-invariant); or a local occupancy grid
  (3D-conv VAE); or just mean+covariance (tiny MLP VAE, Gaussian-only).
- **PE = `sin3d(centroid) ⊕ VAE_z(distribution)`** — keep the positional anchor (the VAE
  encodes *shape*, not absolute position), concat the learned distribution latent.
- **Loss** (doubles as the "new loss function" direction): joint
  `L = L_LM + λ(L_recon + β·L_KL)` — decoder reconstructs the point set / occupancy from `z`.
- **Granularity**: per-patch (196 latents/crop, fine-grained) vs. one `z` per cluster
  (a learned object-shape embedding shared across its tokens).
- **VAE vs deterministic set-encoder**: the variational latent buys a smooth/regularized
  space + the ability to *sample* `z` (stochastic PE = built-in augmentation/uncertainty).
  If sampling isn't needed, a plain set-encoder is simpler and may match it.
- The data is already there: `_per_patch_coord` already masks each patch to its cluster
  pixels — that mask yields the point set to summarize. Wire behind `EXP4_PE_MODE` (e.g.
  `sin3d` | `gaussian` | `vae`) so it A/Bs cleanly against the current sin3d baseline.

Note: unlike the current parameter-free sin3d, this PE is *learned*, so it trains
(no extra capacity frozen).

### Current built-in modes (`world_position_embedding_type`)

Composite string, 3 composable axes (llava_arch.py:384-517):

- **Coord reduction** (pick one): `avg` mean of patch | `sample1/5/9` N sampled pts | `minmax` min&max corner.
- **PE encoder** (pick one): `sin3d` fixed sinusoidal (concat pts into slots) | `mlp` learned 3→512→hidden.
- **Modifiers** (optional): `discrete` quantize coords to voxel grid first | `mrope` use discrete coords as multimodal RoPE.

PE is added to the patch feature. e.g. `sample9_sin3d`, `avg_mlp_discrete`.

### Benchmarks

#### ScanQA val — Exp 4 (`world_position_embedding_type = exp4-discrete-sin3d`)

Per-patch cluster-masked coord → sin3d PE, crop tokens augment uniform frames.
Eval: uniform sampling, 16 frames, `EXP4_MAX_CROPS=12`, `top_k=1`, loaded as a merged
full checkpoint with `sdpa`. Train mix: ScanQA + Scan2Cap (63,180 samples / epoch).

| Run | LoRA | Train data | Eval set | EM | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE |
|---|---|---|---|---|---|---|---|---|---|
| `exp4-lora-fast` | r8 (α16) | ~0.38 ep (3000 steps, 24k) | val 50 (seed 42) | 0.28 | 100.9 | 44.9 | ~0\* | 20.8 | 44.5 |
| `exp4-lora-fast` | r8 (α16) | ~0.38 ep (3000 steps, 24k) | val full (4675) | 0.3005 | 101.35 | 46.80 | 16.07 | 19.90 | 48.96 |
| `exp4-lora-r16-ep1` | r16 (α32) | 1.0 epoch (7897 steps, 63k) | val full (4675) | 0.3106 | 105.10 | 47.81 | 17.64 | 20.40 | 50.20 |

\* BLEU-4 ≈ 0 on the 50-sample subset is a small-sample artifact (short answers form few 4-grams); on full val BLEU-4 = 16.07. Full-val BLEU-2/3 = 31.29 / 22.71. Inference ~1.07 it/s (~73 min for 4675).

Takeaway: ~0.38 epoch (38% of data) already reaches ~paper-level CIDEr — strong base
model + LoRA saturates fast, so the full-epoch r16 run is expected to add only a few
points. Likely larger levers toward/above paper: frames 16→32, eval at paper sampling,
or the 5-task mix.

#### SQA3D test — Exp 4 (zero-shot transfer)

The `exp4-lora-r16-ep1` model (trained on **ScanQA + Scan2Cap only**, SQA3D held out)
evaluated on the full SQA3D **test** split (3,519 questions) — i.e. **zero-shot** situated
QA, no SQA3D in the train mix. Same eval config as ScanQA (uniform/16, `EXP4_MAX_CROPS=12`,
`top_k=1`, merged checkpoint, `sdpa`). Metric is exact-match answer accuracy per question type.

| Run | LoRA | Setting | all | what | is | how | can | which | others |
|---|---|---|---|---|---|---|---|---|---|
| `exp4-lora-r16-ep1` | r16 (α32) | zero-shot | **50.87** | 44.46 | 59.66 | 50.11 | 65.09 | 47.58 | 47.88 |

Takeaway: ~50.9% zero-shot on SQA3D from a model that never trained on it is strong
(situated reasoning transfers from ScanQA+Scan2Cap). `can`/`is` (yes-no-ish) are highest;
`what` (open-vocab) lowest, as expected. Training *on* SQA3D (the paper's 5-task protocol)
is the apples-to-apples next step and should lift this meaningfully.
