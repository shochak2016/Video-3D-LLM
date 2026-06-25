import os
import json
import torch
import pickle
import cv2
import numpy as np
from PIL import Image
from transformers.image_utils import to_numpy_array
import json
from tqdm import tqdm
import random
import copy
# from llava.voxelize import *   # disabled: nvblox_torch not installed; Exp 3 doesn't need voxelize

def convert_from_uvd(u, v, d, intr, pose):
    # extr = np.linalg.inv(pose)
    
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000
    
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    world = (pose @ np.array([x, y, z, 1]))
    return world[:3] / world[3]
    
def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


'''
Takes 3d point clouds from camera frame to world frame
(3d point cloud is crated from cam intrinsics)
'''
def unproject(intrinsics, poses, depths):
    """
        intrinsics: (V, 4, 4)
        poses: (V, 4, 4)
        depths: (V, H, W)
    """
    V, H, W = depths.shape
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

    z = depths.view(V, H*W) / 1000       # (V, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)      # (V, H*W, 4)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)   # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords

'''Convert world coords into voxels'''
def create_voxels():
    pass

'''Average depth (made for a patch)'''
def calc_av_depth(depths):
    return np.mean(depths)


class VideoProcessor:
    def __init__(
        self,
        video_folder="data",
        annotation_dir="data/embodiedscan/",
        voxel_size=None,
        min_xyz_range=None,
        max_xyz_range=None,
        frame_sampling_strategy='uniform',
        val_box_type='pred',
        world_position_embedding_type=None,
    ):
        self.video_folder = video_folder
        self.voxel_size = voxel_size
        self.min_xyz_range = torch.tensor(min_xyz_range) if min_xyz_range is not None else None
        self.max_xyz_range = torch.tensor(max_xyz_range) if max_xyz_range is not None else None
        self.frame_sampling_strategy = frame_sampling_strategy
        # Exp 4: nvblox voxel-cluster crop tokens. Enabled when the PE type string
        # contains 'exp4'. Params are env-overridable so the launcher can tune the
        # crop budget / clustering without code edits.
        self.exp4_enabled = bool(world_position_embedding_type) and ('exp4' in world_position_embedding_type)
        if self.exp4_enabled:
            self.exp4_voxel_size = float(os.environ.get("EXP4_VOXEL_SIZE", 0.1))
            self.exp4_eps = float(os.environ.get("EXP4_EPS", 0.15))
            self.exp4_min_samples = int(os.environ.get("EXP4_MIN_SAMPLES", 5))
            self.exp4_top_k_per_frame = int(os.environ.get("EXP4_TOP_K_PER_FRAME", 2))
            self.exp4_max_crops = int(os.environ.get("EXP4_MAX_CROPS", 24))
            self.exp4_use_rgb = os.environ.get("EXP4_USE_RGB", "0") == "1"
            # Crop selection: "area" (per-frame top-k by bbox) or "angular" (per-cluster
            # azimuth-diverse views -- cuts multi-view redundancy).
            self.exp4_view_select = os.environ.get("EXP4_VIEW_SELECT", "area")
            self.exp4_n_views = int(os.environ.get("EXP4_N_VIEWS", 3))
            self.exp4_ang_thresh = float(os.environ.get("EXP4_ANG_THRESH", 0.52))  # ~30 deg
            self.exp4_cache_dir = os.environ.get("EXP4_CACHE_DIR", "data/exp4_cache")
            os.makedirs(self.exp4_cache_dir, exist_ok=True)
            print(f"[exp4] enabled: voxel={self.exp4_voxel_size} eps={self.exp4_eps} "
                  f"top_k/frame={self.exp4_top_k_per_frame} max_crops={self.exp4_max_crops} "
                  f"view_select={self.exp4_view_select} n_views={self.exp4_n_views} "
                  f"use_rgb={self.exp4_use_rgb} cache={self.exp4_cache_dir}", flush=True)
        self.scene = {}
        print('============frame sampling strategy: {}============='.format(self.frame_sampling_strategy))

        for split in ["train", "val", "test"]:
            with open(os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl"), "rb") as f:
                data = pickle.load(f)["data_list"]
                for item in data:
                    # item["sample_idx"]: "scannet/scene0415_00"
                    if item["sample_idx"].startswith("scannet"):
                        self.scene[item["sample_idx"]] = item

        self.scan2obj = {}

        for split in ['train', 'val']:
            box_type = "gt" if split == "train" else val_box_type
            filename = os.path.join("data", "metadata", f"scannet_{split}_{box_type}_box.json")
            with open(filename) as f:
                data = json.load(f)
                self.scan2obj.update(data)


        # Load max-coverage sampling tables when mc is the base strategy OR when the
        # randomized LoRA recipe may pick 'mc' per sample (so both paths are available).
        if 'mc' in self.frame_sampling_strategy or os.environ.get("LORA_RAND_AUG", "0") == "1":
            sampling_file = "data/metadata/scannet_select_frames.json"
            self.mc_sampling_files = {}
            with open(sampling_file) as f:
                data = json.load(f)
                for dd in data:
                    self.mc_sampling_files[dd['video_id']] = dd

            with open('data/metadata/pcd_discrete_0.1.pkl', 'rb') as f:
                pc_data = pickle.load(f)
            self.pc_min = {}
            self.pc_max = {}
            for scene_id in pc_data:
                pc_points = pc_data[scene_id]
                min_xyz = [1000, 1000, 1000]
                max_xyz = [-1000, -1000, -1000]
                for data in pc_points:
                    min_xyz = [min(v1, v2) for v1, v2 in zip(min_xyz, data)]
                    max_xyz = [max(v1, v2) for v1, v2 in zip(max_xyz, data)]
                self.pc_min[scene_id] = torch.Tensor(min_xyz) / 10
                self.pc_max[scene_id] = torch.Tensor(max_xyz) / 10


    def sample_frame_files_mc(self, video_id: str, frames_upbound: int = 32, do_shift=False):
        mc_files = self.mc_sampling_files[video_id]
        frame_files = mc_files['frame_files'][:frames_upbound]
        voxel_nums = mc_files['voxel_nums'][:frames_upbound]

        ratio = 1.0
        if 'ratio95' in self.frame_sampling_strategy:
            ratio = 0.95
        elif 'ratio90' in self.frame_sampling_strategy:
            ratio = 0.9

        if ratio != 1.0:
            num_all_voxels = mc_files['num_all_voxels']
            out = []
            cc = 0
            for frame_file, voxel_num in zip(frame_files, voxel_nums):
                out.append(frame_file)
                cc += voxel_num
                if cc >= num_all_voxels * ratio:
                    break
            frame_files = out

        frame_files.sort(key=lambda file: int(file.split('/')[-1].split('.')[0]))
        # if do_shift:
        #     ori_len = len(frame_files)
        #     i = random.randint(0, len(frame_files)-1)
        #     frame_files = frame_files[-i:] + frame_files[:-i]
        #     assert len(frame_files) == ori_len
        return frame_files  


    def sample_frame_files(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):
        # video_file: scannet/scene00000_01

        # since the color images have the suffix .jpg
        # frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f)) and os.path.join(video_file, f).endswith(".jpg")]
        # frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
        meta_info = self.scene[video_id]
        frame_files = [os.path.join(self.video_folder, img["img_path"]) for img in meta_info["images"]]

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        if force_sample:
            num_frames_to_sample = frames_upbound
        else:
            num_frames_to_sample = 10

        # For scannet, the RGB camera data is temporally synchronized with the depth sensor via hardware, providing synchronized depth and color capture at 30Hz
        # We follow embodiedscan by sampling one out of every ten images.
        avg_fps = 3
        
        total_frames = len(frame_files)
        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # frame_time = [i/3 for i in sampled_indices]
        # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

        # video_time = total_frames / avg_fps

        return [frame_files[i] for i in sampled_indices]

    def calculate_world_coords(
        self,
        video_id: str, 
        frame_files,
        do_normalize=False,
    ):
        meta_info = self.scene[video_id]
        scene_id = video_id.split('/')[-1]

        axis_align_matrix = torch.from_numpy(np.array(meta_info['axis_align_matrix']))
        depth_intrinsic = torch.from_numpy(np.array(meta_info["depth_cam2img"]))

        depths = []
        poses = []
 
        # Read and store the sampled frames
        for frame_path in frame_files:

            # depth image
            depth_path = frame_path.replace(".jpg", ".png")
            with Image.open(depth_path) as depth_img:
                depth = np.array(depth_img).astype(np.int32)
                depths.append(torch.from_numpy(depth))

            # pose
            pose_file = frame_path.replace("jpg", "txt")
            pose = np.loadtxt(pose_file)
            poses.append(torch.from_numpy(pose))


        depths = torch.stack(depths)   # (V, H, W)
        poses = torch.stack([axis_align_matrix @ pose for pose in poses])     # (V, 4, 4)
        depth_intrinsic = depth_intrinsic.unsqueeze(0).repeat(len(frame_files), 1, 1)
        
        world_coords = unproject(depth_intrinsic.float(), poses.float(), depths.float())    # (V, H, W, 3)

        if do_normalize:
            world_coords = torch.maximum(world_coords, self.pc_min[scene_id].to(world_coords.device))
            world_coords = torch.minimum(world_coords, self.pc_max[scene_id].to(world_coords.device))
        
        return {
            "world_coords": world_coords,
            "poses": poses,
        }

    def load_exp4_inputs(self, video_id, frame_files):
        """Load full-res depth(m)/RGB/intrinsics/poses for Exp 4 clustering.

        RGB jpgs and depth pngs are different resolutions in ScanNet posed_images
        (1296x968 vs 640x480), so we resize RGB down to the depth resolution and
        use depth_cam2img -- this keeps RGB pixels, world_coords, and the bbox
        projection all in one consistent (depth) image frame.
        """
        meta_info = self.scene[video_id]
        axis_align_matrix = torch.from_numpy(np.array(meta_info['axis_align_matrix'])).float()
        depth_intrinsic = torch.from_numpy(np.array(meta_info["depth_cam2img"])).float()  # (4,4)

        depths, poses, rgbs = [], [], []
        for frame_path in frame_files:
            depth_path = frame_path.replace(".jpg", ".png")
            with Image.open(depth_path) as depth_img:
                depth = np.array(depth_img).astype(np.float32)        # (Hd,Wd) mm
                Hd, Wd = depth.shape
                depths.append(torch.from_numpy(depth))
            pose = np.loadtxt(frame_path.replace("jpg", "txt"))
            poses.append(torch.from_numpy(pose).float())
            with Image.open(frame_path) as img:
                rgb = img.convert("RGB").resize((Wd, Hd))             # -> depth res
                rgbs.append(torch.from_numpy(np.array(rgb)))          # (Hd,Wd,3) uint8

        depths = torch.stack(depths)                                  # (V,Hd,Wd) mm
        poses = torch.stack([axis_align_matrix @ p for p in poses])   # (V,4,4) axis-aligned
        rgbs = torch.stack(rgbs)                                      # (V,Hd,Wd,3) uint8
        intr_rep = depth_intrinsic.unsqueeze(0).repeat(len(frame_files), 1, 1)
        world_coords = unproject(intr_rep, poses, depths)             # (V,Hd,Wd,3); divides mm by 1000
        return depths / 1000.0, poses, depth_intrinsic[:3, :3], world_coords, rgbs

    def load_exp4_rgb(self, frame_files):
        """Load just the sampled RGB frames at depth resolution (CPU, worker-safe).

        Used to rebuild crop pixels from cached geometry without touching CUDA.
        """
        rgbs = []
        Hd = Wd = None
        for frame_path in frame_files:
            if Hd is None:
                with Image.open(frame_path.replace(".jpg", ".png")) as d:
                    Wd, Hd = d.size                       # PIL size is (W, H)
            with Image.open(frame_path) as img:
                rgb = img.convert("RGB").resize((Wd, Hd))  # -> depth res
                rgbs.append(torch.from_numpy(np.array(rgb)))
        return torch.stack(rgbs)                           # (V,Hd,Wd,3) uint8

    def _exp4_cache_path(self, video_id, frame_files, eff_sampling):
        import hashlib
        key_src = (f"{video_id}|{len(frame_files)}|{eff_sampling}"
                   f"|v{self.exp4_voxel_size}|e{self.exp4_eps}|m{self.exp4_min_samples}"
                   f"|k{self.exp4_top_k_per_frame}|c{self.exp4_max_crops}|rgb{int(self.exp4_use_rgb)}"
                   f"|vs{self.exp4_view_select}|nv{self.exp4_n_views}"
                   f"{('|at'+str(self.exp4_ang_thresh)) if self.exp4_view_select=='angular' else ''}")
        key = hashlib.md5(key_src.encode()).hexdigest()[:12]
        # _geo suffix marks the lightweight geometry cache (bboxes + per-patch
        # coords); pixels are rebuilt on load, so this stays ~50 KB/scene.
        return os.path.join(self.exp4_cache_dir, f"{video_id.replace('/', '_')}_{key}_geo.pt")

    def ensure_exp4_cache(self, video_id, frame_files, eff_sampling):
        """Build + save the geometry cache for one (scene, sampling, frames) combo
        if missing. GPU/main-process only; used by the offline pre-cache pass.
        Returns True if it built, False if already cached."""
        cache_path = self._exp4_cache_path(video_id, frame_files, eff_sampling)
        if os.path.exists(cache_path):
            return False
        from llava.exp4 import build_scene_geometry
        depths_m, poses, intr, world_coords, _ = self.load_exp4_inputs(video_id, frame_files)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        geo = build_scene_geometry(
            depths_m.to(dev), poses, intr, world_coords.to(dev),
            voxel_size=self.exp4_voxel_size, eps=self.exp4_eps, min_samples=self.exp4_min_samples,
            use_rgb=self.exp4_use_rgb, top_k_per_frame=self.exp4_top_k_per_frame,
            max_crops=self.exp4_max_crops,
            view_select=self.exp4_view_select, n_views_per_cluster=self.exp4_n_views,
            ang_thresh=self.exp4_ang_thresh,
        )
        geo = {k: geo[k].detach().cpu() for k in ("bboxes", "patch_coords", "patch_valid", "frame_id", "img_shape")}
        torch.save(geo, cache_path)
        return True

    def process_exp4(self, video_id, frame_files, image_processor, eff_sampling=None):
        """Exp 4 crop tokens for one scene.

        Caches only the clustering GEOMETRY (the expensive nvblox+DBSCAN step) and
        rebuilds the RGB crop pixels on every call (cheap, CPU, DataLoader-worker
        safe). On a cache miss we run nvblox on CUDA -- which CANNOT happen inside a
        forked worker, so that path raises a clear error telling you to pre-cache.
        """
        from torch.utils.data import get_worker_info
        from llava.exp4 import build_scene_geometry, assemble_crop_pixels
        eff_sampling = eff_sampling if eff_sampling is not None else self.frame_sampling_strategy
        cache_path = self._exp4_cache_path(video_id, frame_files, eff_sampling)

        if os.path.exists(cache_path):
            geo = torch.load(cache_path, map_location='cpu')
        else:
            if get_worker_info() is not None:
                raise RuntimeError(
                    f"[exp4] cache miss for {video_id} ({eff_sampling}, {len(frame_files)} frames) "
                    f"inside a DataLoader worker. nvblox needs CUDA, which can't run in a forked "
                    f"worker. Pre-cache first (train/lora/exp4_precache.sh) or set DATALOADER_WORKERS=0.")
            depths_m, poses, intr, world_coords, _ = self.load_exp4_inputs(video_id, frame_files)
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            geo = build_scene_geometry(
                depths_m.to(dev), poses, intr, world_coords.to(dev),
                voxel_size=self.exp4_voxel_size, eps=self.exp4_eps, min_samples=self.exp4_min_samples,
                use_rgb=self.exp4_use_rgb, top_k_per_frame=self.exp4_top_k_per_frame,
                max_crops=self.exp4_max_crops,
                view_select=self.exp4_view_select, n_views_per_cluster=self.exp4_n_views,
                ang_thresh=self.exp4_ang_thresh,
            )
            geo = {k: geo[k].detach().cpu() for k in ("bboxes", "patch_coords", "patch_valid", "frame_id", "img_shape")}
            torch.save(geo, cache_path)

        # Rebuild SigLIP-normalized crop pixels from cached geometry (CPU).
        rgbs = self.load_exp4_rgb(frame_files)
        pixel_values = assemble_crop_pixels(
            rgbs, geo["frame_id"], geo["bboxes"],
            siglip_mean=tuple(image_processor.image_mean), siglip_std=tuple(image_processor.image_std),
        )
        return {"pixel_values": pixel_values,
                "patch_coords": geo["patch_coords"],
                "patch_valid": geo["patch_valid"]}

            

    def preprocess(
        self,
        video_id: str,
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
        sampling_override: str = None,
    ):

        # Per-call frame-sampling override (used by the randomized LoRA recipe).
        # Falls back to the processor's configured strategy when not given.
        eff_sampling = sampling_override if sampling_override is not None else self.frame_sampling_strategy
        if 'mc' in eff_sampling:
            frame_files = self.sample_frame_files_mc(
                video_id,
                frames_upbound=frames_upbound,
                do_shift=('shift' in self.frame_sampling_strategy),
            )
        else:
            frame_files = self.sample_frame_files(
                video_id,
                force_sample=force_sample,
                frames_upbound=frames_upbound,
            )

        video_dict = self.calculate_world_coords(
            video_id,
            frame_files,
            do_normalize=('norm' in self.frame_sampling_strategy),
        )
        world_coords = video_dict["world_coords"]
        V, H, W, _ = world_coords.shape
        
        # boundry
        world_coords_flat = world_coords.reshape(-1, 3)
        x_min, x_max = world_coords_flat[:, 0].min().item(), world_coords_flat[:, 0].max().item()
        y_min, y_max = world_coords_flat[:, 1].min().item(), world_coords_flat[:, 1].max().item()
        z_min, z_max = world_coords_flat[:, 2].min().item(), world_coords_flat[:, 2].max().item()
        boundry = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

        # x_max = min(world_coords_flat[:, 0].min().abs().item(), world_coords_flat[:, 0].max().item())
        # x_min = - x_max
        # y_max = min(world_coords_flat[:, 1].min().abs().item(), world_coords_flat[:, 1].max().item())
        # y_min = - y_max
        # z_min, z_max = world_coords_flat[:, 2].min().item(), world_coords_flat[:, 2].max().item()
        # boundry = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

        images = []
        for frame_file in frame_files:
            with Image.open(frame_file) as img:
                frame = img.convert("RGB")
                images.append(frame)

        crop_size = image_processor.crop_size["width"]
        if strategy == "resize":
            images = [frame.resize((crop_size, crop_size)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (384, 384), interpolation=cv2.INTER_NEAREST) for coords in world_coords] 
        elif strategy == "center_crop":
            new_height = crop_size
            new_width = int(W * (crop_size / H))
            images = [frame.resize((new_width, new_height)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (new_width, new_height), interpolation=cv2.INTER_NEAREST) for coords in world_coords]
            # Calculate the position and perform the center crop
            left = (new_width - crop_size) // 2
            right = left + crop_size
            top = (new_height - crop_size) // 2
            bottom = top + crop_size
            images = [frame.crop((left, top, right, bottom)) for frame in images]

            resized_coords = [coords[top:bottom, left:right, :] for coords in resized_coords]
        
        # resized_coords_norm = []
        # for coords in resized_coords:
        #     new_coords = coords.copy()
        #     new_coords[...,0] = (new_coords[...,0] - x_min) / (x_max - x_min)
        #     new_coords[...,1] = (new_coords[...,1] - y_min) / (y_max - y_min)
        #     new_coords[...,2] = (new_coords[...,2] - z_min) / (z_max - z_min)
        #     resized_coords_norm.append(new_coords)

        # resized_coords_norm = torch.from_numpy(np.stack(resized_coords_norm))
        return {
            "images": images,
            "world_coords": torch.from_numpy(np.stack(resized_coords)),
            "poses": video_dict["poses"],
            "video_size": len(images),
            "boundry": boundry,
            "objects": torch.tensor(self.scan2obj[video_id]),
            "frame_files": frame_files,
            # "world_coords_norm": resized_coords_norm
        }


    def process_3d_video(
        self,
        video_id: str,
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
        sampling_override: str = None,
    ):
        video_dict = self.preprocess(
            video_id,
            image_processor,
            force_sample,
            frames_upbound,
            strategy,
            sampling_override=sampling_override,
        )
        video_dict["images"] = image_processor.preprocess(video_dict["images"], return_tensors="pt")["pixel_values"]
        if self.exp4_enabled:
            eff_sampling = sampling_override if sampling_override is not None else self.frame_sampling_strategy
            crops = self.process_exp4(video_id, video_dict["frame_files"], image_processor, eff_sampling)
            video_dict["exp4_pixel_values"] = crops["pixel_values"]   # (N,3,384,384)
            video_dict["exp4_patch_coords"] = crops["patch_coords"]   # (N,14,14,3)
            video_dict["exp4_patch_valid"] = crops["patch_valid"]     # (N,14,14)
        video_dict.pop("frame_files", None)
        return video_dict

    
    def discrete_point(self, xyz):
        xyz = torch.tensor(xyz)
        if self.min_xyz_range is not None:
            xyz = torch.maximum(xyz, self.min_xyz_range.to(xyz.device))
        if self.max_xyz_range is not None:
            xyz = torch.minimum(xyz, self.max_xyz_range.to(xyz.device))
        if self.min_xyz_range is not None:
            xyz = (xyz - self.min_xyz_range.to(xyz.device)) 
            
        xyz = xyz / self.voxel_size
        return xyz.round().int().tolist()
    

def merge_video_dict(video_dict_list):
    new_video_dict = {}
    # Per-sample (variable-shape) keys are kept as lists aligned with the batch so
    # batch_size > 1 works (object counts and crop counts differ across scenes, and
    # box_input must stay aligned to its own sample's coord tokens).
    list_keys = ['objects', 'exp4_pixel_values', 'exp4_patch_coords', 'exp4_patch_valid', 'pe_reduction']
    for k in video_dict_list[0]:
        if k in ["world_coords", 'images', 'poses']:
            new_video_dict[k] = torch.stack([vd[k] for vd in video_dict_list])
        elif k in list_keys:
            new_video_dict[k] = [vd[k] for vd in video_dict_list]
        elif k == 'box_input':
            # One entry per sample: (1,3) tensor or None (None = no coord token).
            new_video_dict['box_input'] = [
                (torch.as_tensor(vd['box_input'], dtype=torch.float32).view(1, 3)
                 if vd.get('box_input') is not None else None)
                for vd in video_dict_list
            ]
    if 'box_input' not in new_video_dict:
        new_video_dict['box_input'] = [None] * len(video_dict_list)
    return new_video_dict
