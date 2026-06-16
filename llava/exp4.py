# llava/exp4.py
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor
from sklearn.cluster import DBSCAN
import torch, numpy as np

from typing import List, Optional, Tuple

# SigLIP geometry. Hardcoded because Exp 4 commits to one encoder.
SIGLIP_INPUT = 384                                   # SigLIP-so400m input edge
PATCH_PIX = 27                                       # source pixels per pooled-token edge
GRID = 14                                            # 14x14 tokens per crop (post 2D pool)
DROP_PX = SIGLIP_INPUT - GRID * PATCH_PIX            # = 6, trailing pixels that don't fit a patch

# ── data container ────────────────────────────────────────────────
class VoxelClusters:
    # (the same class from blocks 6-12: __init__, cluster_ids, _mask,
    #  cluster_voxels, _linearize, cluster_voxel_keys, cluster_bbox_2d)
        """
    Result of clustering one scene's surface voxels. Read-only after construction.
    Exposes only what the downstream Exp 4 pipeline needs:
      - cluster_ids        : list of cluster ids (excludes noise -1)
      - cluster_voxels     : per-cluster centers + integer ijk
      - cluster_voxel_keys : linearized int64 keys for torch.isin() membership masks
      - cluster_bbox_2d    : project a cluster's voxels into a camera, return clipped bbox
    """

    def __init__(
        self,
        voxel_centers: torch.Tensor,        # (M, 3) float32, CUDA recommended (used in matmul, isin downstream)
        voxel_ijk: torch.Tensor,            # (M, 3) int64,   CUDA, same device as voxel_centers
        voxel_rgb: Optional[torch.Tensor],  # (M, 3) uint8 or None, same device
        labels: torch.Tensor,               # (M,)   int64,   same device (convert from sklearn numpy first)
        voxel_size: float,                  # Python scalar
        origin: torch.Tensor,               # (3,)   float32, same device
    ):
        self.voxel_centers = voxel_centers           # (M, 3) world coords
        self.voxel_ijk = voxel_ijk                   # (M, 3) integer voxel indices
        self.voxel_rgb = voxel_rgb                   # (M, 3) uint8 or None
        self.labels = labels                         # (M,) int, -1 = noise
        self.voxel_size = float(voxel_size)
        self.origin = origin                         # (3,) grid origin in world coords
        self._mask_cache: dict = {}

    @property
    def cluster_ids(self) -> list:
        return sorted(int(i) for i in self.labels.unique().tolist() if i >= 0)

    def _mask(self, cid: int) -> torch.Tensor:
        if cid not in self._mask_cache:
            self._mask_cache[cid] = (self.labels == cid)
        return self._mask_cache[cid]

    def cluster_voxels(self, cid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self._mask(cid)
        return self.voxel_centers[m], self.voxel_ijk[m]

    @staticmethod
    def _linearize(ijk: torch.Tensor) -> torch.Tensor:
        # Pack (..., 3) ints into one int64 key per voxel for fast `torch.isin` lookups.
        # Bit layout: [i+OFFSET][j+OFFSET][k+OFFSET], 21 bits each.
        # OFFSET shifts signed coords into a positive range; the field is 21 bits
        # so any |i,j,k| < 2**20 fits (a 1M-voxel-wide grid per axis -- plenty).
        OFFSET = 1 << 20
        i = ijk[..., 0].to(torch.int64) + OFFSET
        j = ijk[..., 1].to(torch.int64) + OFFSET
        k = ijk[..., 2].to(torch.int64) + OFFSET
        return (i << 42) | (j << 21) | k

    def cluster_voxel_keys(self, cid: int) -> torch.Tensor:
        _, ijk = self.cluster_voxels(cid)
        return self._linearize(ijk)

    def cluster_bbox_2d(
        self,
        cid: int,
        pose_c2w: torch.Tensor,            # (4, 4)
        intrinsics: torch.Tensor,          # (3, 3) or (4, 4)
        img_shape: Tuple[int, int],        # (H, W)
        pad: int = 0,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Project cluster_id's voxel centers into the given camera and return the
        tight 2D bbox in pixel coords, clipped to the image.

        Camera math (rigid-inverse closed form, no linalg.inv):
            P_cam = R^T (P_w - t)     where pose_c2w = [[R, t], [0, 1]]
        And `(centers - t) @ R` computes that for batched row vectors.

        Returns (x0, y0, x1, y1) ints, x1/y1 EXCLUSIVE (slice-style),
        or None if no voxel projects in front of the camera or inside the image.
        """
        centers, _ = self.cluster_voxels(cid)
        if centers.numel() == 0:
            return None

        dev, dt = centers.device, centers.dtype
        R = pose_c2w[:3, :3].to(dev, dt)
        t = pose_c2w[:3,  3].to(dev, dt)
        cam = (centers - t) @ R                                              # (K, 3)
        z = cam[:, 2]
        front = z > 1e-6
        if not front.any():
            return None
        cam, z = cam[front], z[front]

        K = intrinsics.to(dev, dt)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = (cam[:, 0] * fx / z + cx).round().long()
        v = (cam[:, 1] * fy / z + cy).round().long()

        H, W = int(img_shape[0]), int(img_shape[1])
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not in_img.any():
            return None
        u, v = u[in_img], v[in_img]

        x0 = max(0, int(u.min().item()) - pad)
        y0 = max(0, int(v.min().item()) - pad)
        x1 = min(W, int(u.max().item()) + 1 + pad)
        y1 = min(H, int(v.max().item()) + 1 + pad)
        return None if (x1 <= x0 or y1 <= y0) else (x0, y0, x1, y1)

# ── one-shot pipeline ─────────────────────────────────────────────
def cluster_scene(
    depths, poses_c2w, intrinsics, world_coords,
    rgb_frames=None,
    voxel_size=0.1, surface_thresh=None,
    eps=0.15, min_samples=5, use_rgb=False, rgb_weight=0.05,
) -> VoxelClusters:
    """
    Build per-scene nvblox TSDF, snap candidates to voxel grid, TSDF-filter,
    DBSCAN. Returns VoxelClusters.
    """
    mapper = Mapper(voxel_size, ProjectiveIntegratorType.TSDF)
    H, W = depths.shape[-2:]
    sensor = _make_sensor(intrinsics, W, H)
    for v in range(depths.shape[0]):
        t_w_c = poses_c2w[v].cpu().float().contiguous()
        mapper.add_depth_frame(depths[v].cuda().float().contiguous(), t_w_c, sensor)
        if rgb_frames is not None:
            mapper.add_color_frame(rgb_frames[v].cuda().contiguous(), t_w_c, sensor)
    return _cluster_via_mapper(mapper, world_coords, rgb_frames, voxel_size,
                                surface_thresh, eps, min_samples, use_rgb, rgb_weight)

# (plus _make_sensor and _cluster_via_mapper as small free helpers)
def _make_sensor(intr, width: int, height: int) -> Sensor:
    """Build an nvblox Sensor from (fx,fy,cx,cy) tuple, 3x3, or 4x4 intrinsics."""
    if isinstance(intr, (tuple, list)) and len(intr) == 4:
        fx, fy, cx, cy = (float(x) for x in intr)
    else:
        K = torch.as_tensor(intr, dtype=torch.float32)
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
    return Sensor.from_camera(fu=fx, fv=fy, cu=cx, cv=cy,
                              width=int(width), height=int(height))


@torch.no_grad()
def cluster_scene(
    depths_m: torch.Tensor,                        # (V, H, W) float METRES, 0 = invalid
    poses_c2w: torch.Tensor,                       # (V, 4, 4) camera -> world
    intrinsics: torch.Tensor,                      # (3, 3) or (4, 4) for the depth camera
    world_coords: torch.Tensor,                    # (V, H, W, 3) unproject(depths_m)
    rgb_frames: Optional[torch.Tensor] = None,     # (V, H, W, 3) uint8
    voxel_size: float = 0.1,                       # matches existing dispatch's voxel_size
    surface_thresh: Optional[float] = None,        # |tsdf| <= thresh => surface; default voxel_size
    eps: float = 0.15,                             # DBSCAN neighbor radius in METRES
    min_samples: int = 5,
    use_rgb: bool = False,                         # add scaled RGB to DBSCAN features
    rgb_weight: float = 0.05,                      # metres per unit RGB (so eps stays in metres)
) -> VoxelClusters:
    """
    Integrate every frame's depth (+optional color) into a per-scene TSDF, then
    extract surface voxels from the candidate point cloud and DBSCAN them.

    Why TSDF-filter the candidates: nvblox fuses depth across all frames, so noisy
    single-frame voxels get filtered out. Using the multi-view TSDF as a gate
    gives much cleaner clusters than raw per-frame voxel hashing.
    """
    V, H, W = depths_m.shape
    mapper = Mapper(voxel_size, ProjectiveIntegratorType.TSDF)
    sensor = _make_sensor(intrinsics, W, H)

    # 1. Integrate frames. nvblox requires pose on CPU + depth on CUDA float32.
    for v in range(V):
        t_w_c = poses_c2w[v].cpu().float().contiguous()
        depth = depths_m[v].cuda().float().contiguous()
        mapper.add_depth_frame(depth, t_w_c, sensor)
        if rgb_frames is not None:
            mapper.add_color_frame(rgb_frames[v].cuda().contiguous(), t_w_c, sensor)

    # 2. Collect candidate world points + matching RGB; drop depth==0 invalid pixels.
    pts = world_coords.reshape(-1, 3).cuda().float()
    valid = (depths_m.reshape(-1) > 0)
    pts = pts[valid]
    rgb_pts = None
    if rgb_frames is not None and use_rgb:
        rgb_pts = rgb_frames.reshape(-1, 3).cuda()[valid]

    # 3. Snap to voxel grid. `inverse` lets us scatter per-point data (RGB) back to per-voxel means.
    origin = pts.min(dim=0).values
    ijk_all = torch.floor((pts - origin) / voxel_size).long()
    voxel_ijk, inverse, counts = torch.unique(
        ijk_all, dim=0, return_inverse=True, return_counts=True
    )
    voxel_centers = ((voxel_ijk.float() + 0.5) * voxel_size + origin).contiguous()

    # 4. Per-voxel mean RGB from candidates (used both for DBSCAN and for downstream debug).
    voxel_rgb = None
    if rgb_pts is not None:
        sums = torch.zeros(voxel_ijk.size(0), 3, dtype=torch.float32, device=pts.device)
        sums.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), rgb_pts.float())
        voxel_rgb = (sums / counts.unsqueeze(1)).round().clamp(0, 255).to(torch.uint8)

    # 5. TSDF surface filter. (M, 2) -> [signed_distance, weight];
    #    weight > 0 means observed in at least one frame; |distance| <= thresh
    #    means within the truncation band of a real surface.
    thresh = surface_thresh if surface_thresh is not None else voxel_size
    qout = mapper.query_layer(QueryType.TSDF, voxel_centers)
    dist, weight = qout[:, 0], qout[:, 1]
    keep = (weight > 0) & (dist.abs() <= thresh)
    voxel_centers, voxel_ijk = voxel_centers[keep], voxel_ijk[keep]
    if voxel_rgb is not None:
        voxel_rgb = voxel_rgb[keep]

    # 6. DBSCAN. RGB axes are scaled by rgb_weight so `eps` stays interpretable in metres
    #    when RGB features are appended.
    if voxel_centers.numel() == 0:
        labels = torch.empty(0, dtype=torch.int64, device=voxel_centers.device)
    else:
        feats = voxel_centers.detach().cpu().numpy()
        if use_rgb and voxel_rgb is not None:
            feats = np.concatenate(
                [feats, voxel_rgb.cpu().float().numpy() * rgb_weight], axis=1
            )
        labels_np = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)
        labels = torch.from_numpy(labels_np).to(voxel_centers.device)

    return VoxelClusters(voxel_centers, voxel_ijk, voxel_rgb, labels, voxel_size, origin)



def _crop_and_resize_coords(world_coords_frame: torch.Tensor,
                            bbox: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Slice the per-pixel world_coords at the bbox and resize to 384x384.

    CRITICAL: use NEAREST, not bilinear. Bilinear-interpolating 3D coords creates
    fake "in-between" points that don't correspond to any real surface -- exactly
    the phantom-coord problem Exp 4 is designed to escape. Nearest preserves
    identity: every resized pixel inherits a real source pixel's world XYZ.

    Returns (384, 384, 3).
    """
    x0, y0, x1, y1 = bbox
    crop = world_coords_frame[y0:y1, x0:x1]                                # (h_c, w_c, 3)
    crop = F.interpolate(
        crop.permute(2, 0, 1).unsqueeze(0).float(),
        size=(SIGLIP_INPUT, SIGLIP_INPUT),
        mode='nearest',
    ).squeeze(0).permute(1, 2, 0)
    return crop                                                            # (384, 384, 3)


def _crop_and_resize_rgb(rgb_frame: torch.Tensor,
                          bbox: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Slice + resize the RGB frame to 384x384 using BILINEAR.

    Bilinear here is correct (unlike coords): RGB interpolation between pixels
    yields a smooth, vision-plausible image -- the input distribution SigLIP
    was trained on. Returns float32 in [0, 1] (SigLIP normalization applied later).
    """
    x0, y0, x1, y1 = bbox
    crop = rgb_frame[y0:y1, x0:x1].float() / 255.0                         # (h_c, w_c, 3)
    crop = F.interpolate(
        crop.permute(2, 0, 1).unsqueeze(0),
        size=(SIGLIP_INPUT, SIGLIP_INPUT),
        mode='bilinear', align_corners=False,
    ).squeeze(0)                                                            # (3, 384, 384)
    return crop


def _per_patch_coord(
    crop_coords: torch.Tensor,             # (384, 384, 3) world XYZ per resized pixel
    cluster_voxel_keys: torch.Tensor,      # (K,) int64 linearized cluster voxel keys
    voxel_size: float,
    origin: torch.Tensor,                  # (3,) grid origin
    cluster_centroid: torch.Tensor,        # (3,) used as fallback for empty patches
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute one 3D world coord per token in the 14x14 grid, using ONLY pixels
    whose voxel belongs to the cluster (boundary background pixels excluded).
    Empty patches fall back to the cluster centroid (option (b)).

    Returns:
      patch_coord : (14, 14, 3)
      patch_valid : (14, 14)  True if >=1 cluster pixel landed in this patch
    """
    # 1. Drop trailing 6 px so 14x14 patches of 27x27 fit cleanly (matches the
    #    existing 384 -> 378 -> 14x14 SigLIP grid alignment in llava_arch).
    crop_coords = crop_coords[:-DROP_PX, :-DROP_PX, :]                     # (378, 378, 3)

    # 2. Voxel-membership mask: hash each pixel's coord to a voxel key, check isin().
    #    Must match VoxelClusters._linearize exactly (same OFFSET + bit layout).
    ijk = torch.floor((crop_coords - origin) / voxel_size).long()
    OFFSET = 1 << 20
    keys = (((ijk[..., 0] + OFFSET).to(torch.int64) << 42)
            | ((ijk[..., 1] + OFFSET).to(torch.int64) << 21)
            |  (ijk[..., 2] + OFFSET).to(torch.int64))                     # (378, 378)
    mask = torch.isin(keys, cluster_voxel_keys)                            # (378, 378) bool

    # 3. Per-patch masked mean over 27x27 blocks.
    cb = crop_coords.view(GRID, PATCH_PIX, GRID, PATCH_PIX, 3) \
                    .permute(0, 2, 1, 3, 4)                                # (14, 14, 27, 27, 3)
    mb = mask.view(GRID, PATCH_PIX, GRID, PATCH_PIX) \
             .permute(0, 2, 1, 3)                                          # (14, 14, 27, 27)
    weighted = (cb * mb.float().unsqueeze(-1)).sum(dim=(2, 3))             # (14, 14, 3)
    counts = mb.sum(dim=(2, 3))                                            # (14, 14)
    patch_coord = weighted / counts.clamp_min(1).unsqueeze(-1)             # (14, 14, 3)

    # 4. Empty-patch fallback to cluster centroid (option (b) from the design).
    valid = counts > 0
    centroid = cluster_centroid.to(patch_coord.device).expand(GRID, GRID, 3)
    patch_coord = torch.where(valid.unsqueeze(-1), patch_coord, centroid)
    return patch_coord, valid


# ── main pipeline (rest of Exp 4: crop → SigLIP → PE) ─────────────
@torch.no_grad()
def build_crop_tokens(
    world_coords: torch.Tensor,                    # (V, H, W, 3)
    rgb_frames: torch.Tensor,                      # (V, H, W, 3) uint8
    poses_c2w: torch.Tensor,                       # (V, 4, 4)
    intrinsics: torch.Tensor,                      # (3, 3) or (4, 4) matching (H, W)
    clusters: VoxelClusters,
    pad: int = 4,                                  # bbox expansion in pixels
    bbox_min_side: int = 16,                       # skip too-tiny crops
    top_k_per_frame: Optional[int] = None,         # keep K largest crops per frame; None = all
    siglip_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    siglip_std:  Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict:
    """
    Build per-crop SigLIP inputs + per-patch 3D PE coords for one scene.

    For every (cluster, frame) pair where the cluster projects in-frame:
      - crop the RGB frame at the cluster's 2D bbox, resize to 384x384,
      - slice world_coords identically (nearest resize) so SigLIP RGB tokens
        and PE coord-tokens stay pixel-aligned,
      - compute per-patch cluster-masked coords (boundary BG pixels excluded).

    Returns dict of stacked tensors (N = number of valid crops in the scene):
      pixel_values : (N, 3, 384, 384) float32, SigLIP-normalized
      patch_coords : (N, 14, 14, 3)   per-token world coord for PE
      patch_valid  : (N, 14, 14) bool  True if patch had cluster pixels
      cluster_id   : (N,) long         bookkeeping (which cluster)
      frame_id     : (N,) long         bookkeeping (which frame)
      centroid     : (N, 3) float      per-crop cluster centroid

    N varies per scene; the downstream LLM input path needs to handle variable
    token counts (pad to a per-batch max, or pool down to a fixed budget).
    """
    V, H, W, _ = world_coords.shape
    device = world_coords.device

    mean = torch.tensor(siglip_mean, device=device).view(3, 1, 1)
    std  = torch.tensor(siglip_std,  device=device).view(3, 1, 1)

    # Pre-fetch per-cluster centroids + voxel-key sets (avoid recomputing per frame).
    cluster_ids = clusters.cluster_ids
    centroids = {cid: clusters.cluster_voxels(cid)[0].mean(dim=0).to(device)
                  for cid in cluster_ids}
    voxel_keys = {cid: clusters.cluster_voxel_keys(cid).to(device)
                  for cid in cluster_ids}
    voxel_size = clusters.voxel_size
    origin = clusters.origin.to(device)

    pix_vals, p_coords, p_valid, c_ids, f_ids, c_cents = [], [], [], [], [], []

    for v in range(V):
        # 1. Compute bboxes for every cluster in this frame; remember areas for top-K.
        frame_crops = []
        for cid in cluster_ids:
            bbox = clusters.cluster_bbox_2d(cid, poses_c2w[v], intrinsics, (H, W), pad=pad)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            if (x1 - x0) < bbox_min_side or (y1 - y0) < bbox_min_side:
                continue
            frame_crops.append((cid, bbox, (x1 - x0) * (y1 - y0)))

        # 2. Optional budget cap: keep the K largest crops per frame.
        if top_k_per_frame is not None:
            frame_crops.sort(key=lambda x: -x[2])
            frame_crops = frame_crops[:top_k_per_frame]

        # 3. Build a crop entry per surviving (cluster, frame).
        for cid, bbox, _ in frame_crops:
            rgb_crop = _crop_and_resize_rgb(rgb_frames[v], bbox)            # (3, 384, 384)
            rgb_crop = (rgb_crop - mean) / std                              # SigLIP-normalized

            wc_crop = _crop_and_resize_coords(world_coords[v], bbox)        # (384, 384, 3)
            pc, pv = _per_patch_coord(wc_crop, voxel_keys[cid], voxel_size,
                                       origin, centroids[cid])

            pix_vals.append(rgb_crop)
            p_coords.append(pc)
            p_valid.append(pv)
            c_ids.append(cid)
            f_ids.append(v)
            c_cents.append(centroids[cid])

    # 4. Stack into batched tensors. Empty-result branch returns properly-shaped
    #    empty tensors so the caller doesn't need special-case logic.
    if not pix_vals:
        return {
            "pixel_values": torch.empty(0, 3, SIGLIP_INPUT, SIGLIP_INPUT, device=device),
            "patch_coords": torch.empty(0, GRID, GRID, 3, device=device),
            "patch_valid":  torch.empty(0, GRID, GRID, dtype=torch.bool, device=device),
            "cluster_id":   torch.empty(0, dtype=torch.long, device=device),
            "frame_id":     torch.empty(0, dtype=torch.long, device=device),
            "centroid":     torch.empty(0, 3, device=device),
        }
    return {
        "pixel_values": torch.stack(pix_vals, dim=0),                       # (N, 3, 384, 384)
        "patch_coords": torch.stack(p_coords, dim=0),                       # (N, 14, 14, 3)
        "patch_valid":  torch.stack(p_valid,  dim=0),                       # (N, 14, 14)
        "cluster_id":   torch.tensor(c_ids, dtype=torch.long, device=device),
        "frame_id":     torch.tensor(f_ids, dtype=torch.long, device=device),
        "centroid":     torch.stack(c_cents, dim=0),                        # (N, 3)
    }

