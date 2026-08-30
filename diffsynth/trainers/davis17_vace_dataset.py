from PIL import Image
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import glob
import json
from tqdm import tqdm  # type: ignore
import torch
import torch.nn.functional as F
import time
import fcntl
import random
import copy
from diffsynth.trainers.mesh_util import MeshLoader, RenderConfig
from diffsynth.trainers.aug_utils import (
    apply_horizontal_flip,
    apply_vertical_flip,
    apply_crop,
    apply_gaussian_blur,
    apply_fft_high_freq
)
DEBUG_TIME = False

def _compute_bbox_ratio(mask: np.ndarray) -> float:
    bbox = np.where(mask > 0)
    if bbox[0].size > 0:    
        x1, y1, x2, y2 = bbox[0].min(), bbox[1].min(), bbox[0].max(), bbox[1].max()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        return (w * h / (mask.shape[1] * mask.shape[0])).tolist()
    else:
        return 0.0


split_names = ["train", "val","val_no_filter"]


class DAVIS17_VACE_Dataset(torch.utils.data.Dataset):
    """
    适配 Wan2.1-VACE-1.3B 训练的 DAVIS17 数据集加载器
    
    返回格式：
    {
        "video": [PIL.Image, ...],  # Original video-frame list.
        "prompt": str,  # Prompt.
        "vace_video": [PIL.Image, ...],  # VACE video-frame list (original or mask-processed images).
        "vace_reference_image": [PIL.Image]  # VACE reference image (the first frame).
    }
    """
    def __init__(
        self, 
        dataset_path: str = os.getenv("DAVIS17_DATASET_PATH"),
        split_name: str = "train", 
        length: int = 81,
        split_year: int = 2017,
        target_resolution: Tuple[int, int] = (480, 832),  # Matches the training script's height=480 and width=832.
        min_instance_ratio: float = 0.00,
        min_bbox_ratio: float = 0.00,
        # prompt_template: Optional[str] = None,
        use_masked_vace_video: bool = True,  # If True, use mask-processed images for vace_video; otherwise, use originals.
        dataset_repeat: int = 1,
        max_samples: Optional[int] = None,
        mask_foreground: bool = True,
        mask_strategy: str = "bbox_cover_traj",  # select from ["fine", "bbox_with_traj", "bbox_cover_traj"]
        frame_selecting_strategy: str = "farthest",  # select from ["nearest", "farthest"]
        bbox_scale: float = 1.0,  # bbox_scale times the size of the bbox
        trajectory_type: str = "mask",  # select from ["mask", "box", "sparse_box"]
        sparse_box_interval: int = 10,  # interval for sparse_box (e.g., 5 means keep box every 5 frames)
        enable_data_augmentation: bool = True,  # Whether to enable data augmentation.
        augmentation_no_aug_prob: float = 0.2,  # Probability of no augmentation (20%).
        augmentation_blur_prob: float = 0.5,  # Probability of applying blur during augmentation.
        augmentation_device: str = 'cuda',  # Device used for FFT.
        generate_high_freq_mask: bool = False,  # Whether to generate a high-frequency mask. ### TODO
        downsample_10hz: bool = False, 
        **kwargs,  # Additional args
    ):
        self.dataset_path = dataset_path
        self.split_name = split_name
        assert split_name in split_names, "split_name must be in " + str(split_names)
        self.length = length
        self.target_resolution = target_resolution
        self.min_instance_ratio = min_instance_ratio
        self.min_bbox_ratio = min_bbox_ratio
        self.use_masked_vace_video = use_masked_vace_video
        self.dataset_repeat = dataset_repeat
        # self.prompt_template = prompt_template if prompt_template else "a video of {class_name}"
        self.max_samples = max_samples
        self.mask_foreground = mask_foreground
        self.mask_strategy = mask_strategy
        self.frame_selecting_strategy = frame_selecting_strategy
        self.bbox_scale = bbox_scale
        self.trajectory_type = trajectory_type
        assert trajectory_type in ["mask", "box", "sparse_box"], f"trajectory_type must be one of ['mask', 'box', 'sparse_box'], got {trajectory_type}"
        self.sparse_box_interval = sparse_box_interval
        self.enable_data_augmentation = enable_data_augmentation
        self.augmentation_no_aug_prob = augmentation_no_aug_prob
        self.augmentation_blur_prob = augmentation_blur_prob
        self.augmentation_device = augmentation_device if torch.cuda.is_available() and augmentation_device == 'cuda' else 'cpu'
        self.generate_high_freq_mask = generate_high_freq_mask
        self.downsample_10hz = downsample_10hz

        
        assert os.getenv("DAVIS17_CAPTION_PATH") is not None, "DAVIS17_CAPTION_PATH is not set, please source path_setup.sh first!"
        self.caption_path = os.getenv("DAVIS17_CAPTION_PATH")
        # Load mesh with optional PTv3 feature extraction
        # self.enable_ptv3 = kwargs.get("enable_ptv3", False)
        assert os.getenv("DAVIS17_V2M4_RESULTS_PATH") is not None, "DAVIS17_V2M4_RESULTS_PATH is not set, please source path_setup.sh first!"
        self.mesh_loader = MeshLoader(
            dataset="DAVIS17",
            data_root=os.getenv("DAVIS17_V2M4_RESULTS_PATH"),
            # enable_ptv3=self.enable_ptv3,
        )
        self.render_configs = [RenderConfig(rotation_angle_deg=0, texture=True, align_to_x_axis=True,
                                  light_type='none'),
                                  RenderConfig(rotation_angle_deg=90, texture=True, align_to_x_axis=True,
                                  light_type='none'),
                                  RenderConfig(rotation_angle_deg=180, texture=True, align_to_x_axis=True,
                                  light_type='none'),
                                  RenderConfig(rotation_angle_deg=270, texture=True, align_to_x_axis=True,
                                  light_type='none')
                                  ]
        # Add the load_from_cache attribute for compatibility with launch_training_task.
        self.load_from_cache = False
        
        self.CLASSES = glob.glob(os.path.join(dataset_path, "Annotations", "Full-Resolution", "*"))
        self.CLASSES = [os.path.basename(cls) for cls in self.CLASSES]
        assert os.getenv("DAVIS17_DEPTH_PATH") is not None, "DAVIS17_DEPTH_PATH is not set, please source path_setup.sh first!"
        self.depth_data_path = os.getenv("DAVIS17_DEPTH_PATH")
        assert os.getenv("DAVIS17_DATASET_JSON_PATH") is not None, "DAVIS17_DATASET_JSON_PATH is not set, please source path_setup.sh first!"
        data_json_path = os.getenv("DAVIS17_DATASET_JSON_PATH")
        # Ensure directory exists
        # print("-----------",data_json_path)
        os.makedirs(data_json_path, exist_ok=True)
        
        # Load from json file if exists
        json_filename = os.path.join(
            data_json_path, 
            "dataset_list_" + self.split_name + "_" + str(self.length) + "_" + 
            str(round(self.min_instance_ratio*100, 2)) + "_" + str(round(self.min_bbox_ratio*100, 2)) + ".json"
        )
        lock_filename = json_filename + ".lock"
        
        # print(json_filename)
        # Try to load from existing JSON file
        if os.path.exists(json_filename):
            # print("Exists-----------",json_filename)
            try:
                self.dataset_list = json.load(open(json_filename, "r"))
                # target_classes = [
                #     "bike-packing",
                #     "camel",
                #     "car-roundabout",
                #     "cows",
                #     "dog",
                #     "drift-straight",
                #     "lab-coat",
                #     "loading",
                #     "motocross-jump",
                #     "pigs",
                #     "shooting"
                # ]
                # # keep only target class for debugging
                # self.dataset_list = [item for item in self.dataset_list if item['major_class'] in target_classes]
                print("Data loaded from json file")
                # print("Total dataset length: ", len(self.dataset_list))
                self._apply_max_samples()
                return
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading JSON file: {e}, will regenerate...")
        
        # Use file lock to ensure only one process prepares data
        max_wait_time = 3600  # Maximum wait time: 1 hour
        wait_interval = 2  # Check every 2 seconds
        start_time = time.time()
        
        lock_file = None
        while True:
            try:
                # Try to acquire exclusive lock
                lock_file = open(lock_filename, "w")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # We got the lock, check if file was created by another process
                if os.path.exists(json_filename):
                    try:
                        self.dataset_list = json.load(open(json_filename, "r"))
                        print("Data loaded from json file (created by another process)")
                        print("Total dataset length: ", len(self.dataset_list))
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        lock_file.close()
                        self._apply_max_samples()
                        return
                    except (json.JSONDecodeError, IOError):
                        # File exists but is corrupted, regenerate
                        pass
                
                # We have the lock and file doesn't exist, prepare data
                print("No json file found, preparing data...")
                self._prepare_dataset()
                print("Data prepared")
                print("Total dataset length: ", len(self.dataset_list))
                
                # Save to temporary file first, then rename (atomic operation)
                temp_filename = json_filename + ".tmp"
                with open(temp_filename, "w") as f:
                    json.dump(self.dataset_list, f)
                os.rename(temp_filename, json_filename)
                print(f"Data saved to json file: {json_filename}")
                
                # Release lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                self._apply_max_samples()
                return
                
            except BlockingIOError:
                # Lock is held by another process, wait and retry
                if lock_file is not None:
                    lock_file.close()
                    lock_file = None
                
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    raise RuntimeError(f"Timeout waiting for data preparation. Lock file: {lock_filename}")
                
                # Print waiting message
                # wait_count = int(elapsed_time / wait_interval) + 1
                # if wait_count % 5 == 0:  # Print every 10 seconds (5 * 2s)
                #     print(f"[Rank {os.environ.get('RANK', '?')}] Waiting for data preparation... ({int(elapsed_time)}s elapsed)", flush=True)
                
                # Wait and check if file was created
                time.sleep(wait_interval)
                if os.path.exists(json_filename):
                    try:
                        self.dataset_list = json.load(open(json_filename, "r"))
                        print("Data loaded from json file (created by another process)")
                        print("Total dataset length: ", len(self.dataset_list))
                        self._apply_max_samples()
                        return
                    except (json.JSONDecodeError, IOError):
                        # File exists but is corrupted, continue waiting
                        continue
            except Exception as e:
                if lock_file is not None:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        lock_file.close()
                    except:
                        pass
                raise

    def _prepare_dataset(self):
        # Load split folder names
        split_file_path = os.path.join(self.dataset_path, "ImageSets", "2017", self.split_name + ".txt")
        if self.split_name=='val_no_filter':
            split_file_path = os.path.join(self.dataset_path, "ImageSets", "2017", 'val' + ".txt")
        with open(split_file_path, "r") as f:
            self.split_folder_names = f.read().splitlines()
        assert len(self.split_folder_names) > 0, "split_folder_names is empty"

        seq_iter = []
        for split_folder_name in self.split_folder_names:
            seq_name = split_folder_name.split("/")[-1]
            mask_files = glob.glob(os.path.join(self.dataset_path, "Annotations", "Full-Resolution", seq_name, "*.png"))
            img_files = glob.glob(os.path.join(self.dataset_path, "JPEGImages", "Full-Resolution", seq_name, "*.jpg"))
            mask_files.sort()
            img_files.sort()
            seq_iter.append((seq_name, mask_files, img_files))
        
        seq_iter = tqdm(seq_iter, desc="Preparing data", unit="seq")
        self.dataset_list = []
        
        for seq_name, mask_files, img_files in seq_iter:
            # Get image dimensions
            with Image.open(mask_files[0]) as m0:
                img_w, img_h = m0.size

            instance_ratios_dict: Dict[int, List[float]] = {}
            bbox_ratios_dict: Dict[int, List[float]] = {}
            frame_iter = tqdm(mask_files, desc=f'{seq_name} masks', unit='frame', leave=False)

            for mf in frame_iter:
                with Image.open(mf) as m:
                    arr = np.array(m)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    # Find all instance ids (>0)
                    ids = np.unique(arr)
                    ids = ids[ids > 0]
                    if ids.size == 0:
                        continue

                    if img_w <= 0 or img_h <= 0:
                        h, w = arr.shape[:2]
                        denom = float(w * h)
                    else:
                        denom = float(img_w * img_h)
                    if denom <= 0:
                        continue

                    for inst_id in ids.tolist():
                        mask = (arr == inst_id)
                        area = int(mask.sum())
                        if area <= 0:
                            continue
                        ratio = area / denom

                        if inst_id not in instance_ratios_dict:
                            instance_ratios_dict[inst_id] = []
                        instance_ratios_dict[inst_id].append(ratio)

                        # Calculate bbox ratio
                        bbox_ratio = _compute_bbox_ratio(mask)
                        if inst_id not in bbox_ratios_dict:
                            bbox_ratios_dict[inst_id] = []
                        bbox_ratios_dict[inst_id].append(bbox_ratio)
                        assert bbox_ratio >= ratio

            # Calculate statistics for each instance
            for inst_id in instance_ratios_dict.keys():
                inst_ratios = instance_ratios_dict[inst_id]
                bbox_ratios = bbox_ratios_dict[inst_id]
                if inst_ratios:
                    inst_mean = float(np.mean(inst_ratios))
                    bbox_mean = float(np.mean(bbox_ratios))
                    if inst_mean < self.min_instance_ratio or bbox_mean < self.min_bbox_ratio:
                        continue
                    if self.split_name == 'val_no_filter':
                        with Image.open(mask_files[0]) as m:
                            arr = np.array(m)
                            if arr.ndim == 3:
                                arr = arr[..., 0]
                            if inst_id not in np.unique(arr):
                                continue
                        mask_files_slice = mask_files
                        img_files_slice = img_files
                        self.dataset_list.append({
                            'instance_id': inst_id,
                            'major_class': seq_name,
                            'img_w': int(img_w),
                            'img_h': int(img_h),
                            'mask_files': mask_files_slice,
                            'img_files': img_files_slice
                        })
                    else:
                        for i in range(len(mask_files) - self.length + 1):
                            with Image.open(mask_files[i]) as m:
                                arr = np.array(m)
                                if arr.ndim == 3:
                                    arr = arr[..., 0]
                                if inst_id not in np.unique(arr):
                                    continue
                            mask_files_slice = mask_files[i:i+self.length]
                            img_files_slice = img_files[i:i+self.length]
                            self.dataset_list.append({
                                'instance_id': inst_id,
                                'major_class': seq_name,
                                'img_w': int(img_w),
                                'img_h': int(img_h),
                                'mask_files': mask_files_slice,
                                'img_files': img_files_slice
                            })

    def _apply_max_samples(self):
        if self.max_samples is None:
            return
        if self.max_samples <= 0:
            return
        original_length = len(self.dataset_list)
        if original_length <= self.max_samples:
            return
        self.dataset_list = self.dataset_list[:self.max_samples]
        print(f"[DEBUG] Limiting dataset from {original_length} to {len(self.dataset_list)} samples.")

    def _generate_trajectory_maps(self, mask_arrays: List[np.ndarray], linewidth: int = 10) -> List[Image.Image]:
        """
        Generate trajectory maps based on trajectory_type.
        
        Args:
            mask_arrays: List of binary mask arrays (H, W) with values 0 or 1
            
        Returns:
            List of PIL.Image (RGB) representing trajectory maps
        """
        H, W = mask_arrays[0].shape
        trajectory_maps = []
        
        if self.trajectory_type == "mask":
            # Convert fine masks directly to RGB images (white for mask region)
            # mask_arrays are fine masks (precise segmentation masks without bbox processing)
            for mask_arr in mask_arrays:
                # Convert float32 [0,1] fine mask to uint8 [0,255] RGB images
                mask_rgb = np.stack([mask_arr.astype(np.uint8) * 255] * 3, axis=-1)
                trajectory_maps.append(Image.fromarray(mask_rgb, mode='RGB'))
                
        elif self.trajectory_type == "box":
            # Draw bounding boxes for each frame
            for mask_arr in mask_arrays:
                # Create blank RGB image
                trajectory_map = np.zeros((H, W, 3), dtype=np.uint8)
                
                # Find bounding box
                ys, xs = np.where(mask_arr > 0)
                if len(ys) > 0 and len(xs) > 0:
                    y1, x1 = int(np.min(ys)), int(np.min(xs))
                    y2, x2 = int(np.max(ys)), int(np.max(xs))
                    
                    # Apply bbox_scale
                    h, w = y2 - y1, x2 - x1
                    y1 = int(y1 - h * (self.bbox_scale - 1) / 2)
                    y2 = int(y2 + h * (self.bbox_scale - 1) / 2)
                    x1 = int(x1 - w * (self.bbox_scale - 1) / 2)
                    x2 = int(x2 + w * (self.bbox_scale - 1) / 2)
                    
                    # Clip to image boundaries
                    y1 = max(0, min(y1, H - linewidth))
                    y2 = max(0, min(y2, H - linewidth))
                    x1 = max(0, min(x1, W - linewidth))
                    x2 = max(0, min(x2, W - linewidth))                    
                    
                    # Draw rectangle (white)
                    trajectory_map[y1:y1+linewidth, x1:x2, :] = 255
                    trajectory_map[y2:y2+linewidth, x1:x2, :] = 255
                    trajectory_map[y1:y2, x1:x1+linewidth, :] = 255
                    trajectory_map[y1:y2, x2:x2+linewidth, :] = 255
                
                trajectory_maps.append(Image.fromarray(trajectory_map, mode='RGB'))
                
        elif self.trajectory_type == "sparse_box":
            # Draw bounding boxes only at specified intervals
            # Calculate which frames to keep (uniformly spaced)
            num_frames = len(mask_arrays)
            # Generate uniformly spaced indices (always includes 0)
            sparse_indices = set(range(0, num_frames, self.sparse_box_interval))
            
            for frame_idx, mask_arr in enumerate(mask_arrays):
                # Create blank RGB image
                trajectory_map = np.zeros((H, W, 3), dtype=np.uint8)
                
                # Only draw box if this frame is in sparse_indices
                if frame_idx in sparse_indices:
                    # Find bounding box
                    ys, xs = np.where(mask_arr > 0)
                    if len(ys) > 0 and len(xs) > 0:
                        y1, x1 = int(np.min(ys)), int(np.min(xs))
                        y2, x2 = int(np.max(ys)), int(np.max(xs))
                        
                        # Apply bbox_scale
                        h, w = y2 - y1, x2 - x1
                        y1 = int(y1 - h * (self.bbox_scale - 1) / 2)
                        y2 = int(y2 + h * (self.bbox_scale - 1) / 2)
                        x1 = int(x1 - w * (self.bbox_scale - 1) / 2)
                        x2 = int(x2 + w * (self.bbox_scale - 1) / 2)
                        
                        # Clip to image boundaries
                        y1 = max(0, min(y1, H - 1))
                        y2 = max(0, min(y2, H - 1))
                        x1 = max(0, min(x1, W - 1))
                        x2 = max(0, min(x2, W - 1))
                        
                        # Draw rectangle (white)
                        # Draw top and bottom edges
                        trajectory_map[y1, x1:x2+1, :] = 255
                        trajectory_map[y2, x1:x2+1, :] = 255
                        # Draw left and right edges
                        trajectory_map[y1:y2+1, x1, :] = 255
                        trajectory_map[y1:y2+1, x2, :] = 255
                
                trajectory_maps.append(Image.fromarray(trajectory_map, mode='RGB'))
        else:
            raise ValueError(f"Unknown trajectory_type: {self.trajectory_type}")
        
        return trajectory_maps

    @staticmethod
    def compute_high_freq_mask_pixel(video_tensor: torch.Tensor) -> torch.Tensor:
        """
        在像素级别计算高频mask（全图）。
        
        Args:
            video_tensor: Tensor of shape [B, C, T, H, W] - 视频帧（RGB，值范围[-1, 1]或[0, 1]）
        
        Returns:
            high_freq_mask: Tensor of shape [B, 1, T, H, W] - 高频mask（0或1）
        """
        B, C, T, H, W = video_tensor.shape
        high_freq_mask = torch.zeros((B, 1, T, H, W), dtype=video_tensor.dtype, device=video_tensor.device)
        
        # Create the Laplacian convolution kernel.
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=video_tensor.dtype, device=video_tensor.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        
        # Iterate over batches and time steps.
        for i in range(B):
            for t in range(T):
                frame = video_tensor[i, :, t, :, :]  # [C, H, W]
                
                # Apply Laplacian filtering to each channel separately.
                laplacian_results = []
                for c in range(C):
                    channel_frame = frame[c:c+1, :, :].unsqueeze(0)  # [1, 1, H, W]
                    laplacian_result = F.conv2d(channel_frame, laplacian_kernel, padding=1)  # [1, 1, H, W]
                    laplacian_results.append(laplacian_result)
                
                # Combine results from all channels and take the absolute value.
                laplacian_combined = torch.cat(laplacian_results, dim=1)  # [1, C, H, W]
                laplacian_abs = torch.abs(laplacian_combined)  # [1, C, H, W]
                
                # Average across all channels.
                laplacian_abs = laplacian_abs.mean(dim=1, keepdim=True)  # [1, 1, H, W]
                
                # Compute the standard deviation and threshold to identify high-frequency regions.
                threshold = laplacian_abs.mean() + 2 * laplacian_abs.std()
                high_freq_mask[i, 0, t, :, :] = (laplacian_abs.squeeze() > threshold).to(dtype=video_tensor.dtype)
        
        return high_freq_mask

    @staticmethod
    def compute_high_freq_mask_in_mask_region(
        video_tensor: torch.Tensor, 
        mask_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        在mask区域内计算高频mask。
        
        Args:
            video_tensor: Tensor of shape [B, C, T, H, W] - 视频帧
            mask_tensor: Tensor of shape [B, 1, T, H, W] or [B, T, H, W] - mask（0或1）
        
        Returns:
            high_freq_mask: Tensor of shape [B, 1, T, H, W] - mask区域内的高频mask
        """
        # Ensure that the mask uses the [B, 1, T, H, W] format.
        if mask_tensor.ndim == 4:
            mask_tensor = mask_tensor.unsqueeze(1)  # [B, T, H, W] -> [B, 1, T, H, W]
        
        # First compute the high-frequency mask for the full image.
        high_freq_mask_full = DAVIS17_VACE_Dataset.compute_high_freq_mask_pixel(video_tensor)
        
        # Retain high-frequency information only within the mask region.
        high_freq_mask_in_mask = high_freq_mask_full * mask_tensor
        
        return high_freq_mask_in_mask

    @staticmethod
    def mask_video_to_latents(
        high_freq_mask: torch.Tensor,  # [B, 1, T, H, W] high-frequency mask at video-frame level
        latent_shape: Tuple[int, int, int, int, int],  # (B, C, T_latent, H_latent, W_latent)
        num_frames: int = 81,  # Number of video frames.
        num_multiview_frames: int = 0,  # Number of multiview_reference_image frames (f); no extra noise for the first f frames.
        downsample_mode: str = "nearest",  # Mask downsampling mode.
        time_reduce: str = "max",  # Temporal downsampling method: "max" or "mean".
    ) -> torch.Tensor:
        """
        将视频帧级别的高频mask映射到latent空间。
        
        Args:
            high_freq_mask: Tensor of shape [B, 1, T, H, W] - 视频帧级别的高频mask
            latent_shape: Tuple (B, C, T_latent, H_latent, W_latent) - latent的shape，其中T_latent = f + T'
            num_frames: 视频帧数（默认81）
            num_multiview_frames: multiview_reference_image 的帧数（f），前f帧不进行额外加噪声
            downsample_mode: 空间下采样模式
            time_reduce: 时间维度降采样方式
        
        Returns:
            latent_mask: Tensor of shape [B, 1, T_latent, H_latent, W_latent] - latent级别的高频mask
                前f帧（multiview_reference_image）设为0（不进行额外加噪声）
                后续帧对应原视频的高频mask
        """
        B, C, T_latent, H_latent, W_latent = latent_shape
        Bm, Cm, T, H, W = high_freq_mask.shape
        
        assert Bm == B and Cm == 1, f"mask should be [B,1,T,H,W], got {high_freq_mask.shape}"
        assert T == num_frames, f"expect video T={num_frames}, got {T}"
        assert T_latent >= num_multiview_frames + 1, f"T_latent ({T_latent}) should be >= num_multiview_frames ({num_multiview_frames}) + 1"
        
        # 1) Downsample the spatial dimensions from (H, W) to (H_latent, W_latent).
        mask_2d = high_freq_mask.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)  # [B*T,1,H,W]
        mask_2d_ds = F.interpolate(mask_2d, size=(H_latent, W_latent), mode=downsample_mode)  # [B*T,1,H_latent,W_latent]
        mask_ds = mask_2d_ds.view(B, T, 1, H_latent, W_latent).permute(0, 2, 1, 3, 4)  # [B,1,T,H_latent,W_latent]
        
        # 2) Construct latent_mask: [B, 1, T_latent, H_latent, W_latent].
        # Set the first f multiview_reference_image frames to 0 (no extra noise).
        latent_mask = torch.zeros((B, 1, T_latent, H_latent, W_latent), device=high_freq_mask.device, dtype=mask_ds.dtype)
        
        # 2.1) Latent t'=f corresponds to the first video frame (t=0).
        # The first dimension (index f) retains the first T entry in high_freq_mask.
        if T_latent > num_multiview_frames:
            latent_mask[:, :, num_multiview_frames, :, :] = mask_ds[:, :, 0, :, :]
        
        # 2.2) Latent t'=f+1..T_latent-1 corresponds to video frames 2..T.
        # Every four frames correspond to one latent.
        # Compute the number of latent frames for the original video: T' = T_latent - num_multiview_frames.
        T_prime = T_latent - num_multiview_frames  # Latent-frame count for the original video (e.g., 25).
        
        # Per the reference implementation, latent t'=f+1 maps to video frame 2 (t=1), with one latent per four frames.
        # Segment k=0..(T_prime-2) maps to video t: 1+4k .. 1+4(k+1)-1.
        num_chunks = min(T_prime - 1, (T - 1 + 3) // 4)  # At most T_prime-1 chunks.
        for k in range(num_chunks):
            start = 1 + 4 * k
            end = min(1 + 4 * (k + 1), T)
            if start >= T:
                break
            chunk = mask_ds[:, :, start:end, :, :]  # [B,1,4,H_latent,W_latent] or fewer frames
            
            if time_reduce == "max":
                pooled = chunk.max(dim=2).values
            elif time_reduce == "mean":
                pooled = chunk.mean(dim=2)
            else:
                raise ValueError(f"Unsupported time_reduce={time_reduce}")
            
            latent_idx = num_multiview_frames + 1 + k
            if latent_idx < T_latent:
                latent_mask[:, :, latent_idx, :, :] = pooled
        
        return latent_mask

    def __len__(self):
        return len(self.dataset_list) * self.dataset_repeat

    def __getitem__(self, index):
        if DEBUG_TIME:
            # Initialize timing
            if not hasattr(self, '_timing_stats'):
                self._timing_stats = {
                    'load_images': [],
                    'resize': [],
                    'data_augmentation': [],
                    'mask_processing': [],
                    'vace_video_prep': [],
                    'multiview_render': [],
                    'trajectory_maps': [],
                    'caption_loading': [],
                    'high_frequency_masks': [],
                    'depth conversion': [],
                    'other': [],
                    'total': []
                }
                self._timing_counter = 0
            
            total_start = time.time()
        
        # Handle repeat
        actual_index = index % len(self.dataset_list)
        
        seq_name = self.dataset_list[actual_index]['major_class']
        instance_id = self.dataset_list[actual_index]['instance_id']
        mask_files = self.dataset_list[actual_index]['mask_files']
        img_files = self.dataset_list[actual_index]['img_files']
        depth_video_files = [img_file.replace(self.dataset_path+"/JPEGImages/Full-Resolution", os.getenv("DAVIS17_DEPTH_PATH")).replace(".jpg", ".png") for img_file in img_files]
        seq_name_instance_id = f"{seq_name}/{instance_id}"
        # Verify mask and img files match
        for i in range(len(mask_files)):
            assert mask_files[i].split("/")[-1].split(".")[0] == img_files[i].split("/")[-1].split(".")[0]
            assert mask_files[i].split("/")[-1].split(".")[0] == depth_video_files[i].split("/")[-1].split(".")[0]
        # Timing: Load images
        if DEBUG_TIME:
            load_start = time.time()
        mask_images = [Image.open(mf) for mf in mask_files]
        img_images = [Image.open(img_file) for img_file in img_files]
        depth_video_images = [Image.open(depth_video_file) for depth_video_file in depth_video_files]
        if DEBUG_TIME:
            load_end = time.time()
            self._timing_stats['load_images'].append(load_end - load_start)
        
        # Timing: Resize
        if DEBUG_TIME:
            resize_start = time.time()
        mask_images = [mf.resize(self.target_resolution, Image.Resampling.NEAREST) for mf in mask_images]
        img_images = [img.resize(self.target_resolution, Image.Resampling.BICUBIC) for img in img_images]
        depth_video_images = [img.resize(self.target_resolution, Image.Resampling.BICUBIC) for img in depth_video_images]
        if DEBUG_TIME:
            resize_end = time.time()
            self._timing_stats['resize'].append(resize_end - resize_start)

        if DEBUG_TIME:
            depth_start = time.time()
        depth_arrays = [np.array(img) for img in depth_video_images]
        if depth_arrays[0].dtype == np.uint16 or depth_arrays[0].max() > 255:
            # Find global min/max across all frames for consistent scaling
            global_min = 0
            global_max = 65535
            # print(f"Depth video image array min: {global_min}, max: {global_max} (across all frames)")
            # Normalize all frames using the same scale
            converted_depth_images = []
            for depth_array in depth_arrays:
                if global_max > global_min:
                    depth_normalized = ((depth_array - global_min) / (global_max - global_min) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_array, dtype=np.uint8)
                # Create RGB image (grayscale: same value in all 3 channels)
                depth_rgb = np.stack([depth_normalized] * 3, axis=-1)
                converted_depth_images.append(Image.fromarray(depth_rgb, mode='RGB'))
            depth_video_images = converted_depth_images
        else:
            # Already 8-bit or less, just convert mode if needed
            depth_video_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in depth_video_images]
        if DEBUG_TIME:
            depth_end = time.time()
            self._timing_stats['depth conversion'].append(depth_end - depth_start)
        # Timing: Data augmentation
        if DEBUG_TIME:
            aug_start = time.time()
        # Apply data augmentation if enabled (apply to all frames consistently)
        if self.enable_data_augmentation and self.split_name == "train":
            # Apply no augmentation 20% of the time and augmentation 80% of the time.
            if random.random() >= self.augmentation_no_aug_prob:
                # Choose a geometric transform from flip, crop, FFT, and scale_down_pad.
                geometric_aug = random.choice(['flip_h', 'flip_v', 'crop', 'fft']) # 'scale_down_pad'
                if geometric_aug == 'scale_down_pad':
                    scale_ratio = random.uniform(0.3, 0.8)
                # Determine whether to apply blur (never with FFT).
                apply_blur = random.random() < self.augmentation_blur_prob if geometric_aug != 'fft' else False
                
                # Apply the same augmentation to all frames.
                augmented_img_images = []
                augmented_mask_images = []
                augmented_depth_video_images = []
                
                for img, mask, depth_video in zip(img_images, mask_images, depth_video_images):
                    # Apply the geometric transform.
                    if geometric_aug == 'flip_h':
                        aug_img, aug_mask = apply_horizontal_flip(img, mask)
                        aug_depth_video, aug_depth_video_mask = apply_horizontal_flip(depth_video, mask)
                    elif geometric_aug == 'flip_v':
                        aug_img, aug_mask = apply_vertical_flip(img, mask)
                        aug_depth_video, aug_depth_video_mask = apply_vertical_flip(depth_video, mask)
                    elif geometric_aug == 'crop':
                        aug_img, aug_mask = apply_crop(img, mask)
                        aug_depth_video, aug_depth_video_mask = apply_crop(depth_video, mask)
                    elif geometric_aug == 'fft':
                        aug_img, aug_mask = apply_fft_high_freq(img, mask, device=self.augmentation_device)
                        aug_depth_video, aug_depth_video_mask = apply_fft_high_freq(depth_video, mask, device=self.augmentation_device)
                    else:
                        aug_img, aug_mask = img, mask
                        aug_depth_video, aug_depth_video_mask = depth_video, mask
                    
                    # Apply blur if needed (never with FFT).
                    if apply_blur:
                        aug_img, aug_mask = apply_gaussian_blur(aug_img, aug_mask)
                        aug_depth_video, aug_depth_video_mask = apply_gaussian_blur(aug_depth_video, aug_depth_video_mask)
                    augmented_img_images.append(aug_img)
                    augmented_mask_images.append(aug_mask)
                    augmented_depth_video_images.append(aug_depth_video)
                img_images = augmented_img_images
                mask_images = augmented_mask_images
                depth_video_images = augmented_depth_video_images
        if DEBUG_TIME:
            aug_end = time.time()
            self._timing_stats['data_augmentation'].append(aug_end - aug_start)
        
        # Timing: Mask processing
        if DEBUG_TIME:
            mask_proc_start = time.time()
        # Convert mask to numpy array and filter by instance_id
        mask_arrays = [np.array(mf) for mf in mask_images]
        # Handle 3D masks (take first channel if needed)
        mask_arrays = [mf if mf.ndim == 2 else mf[..., 0] for mf in mask_arrays]
        # Convert to binary mask (1 for instance_id, 0 otherwise)
        mask_arrays = [(mf == instance_id).astype(np.float32) for mf in mask_arrays]
        
        # Save a copy of fine masks for trajectory_maps generation
        # This is necessary because mask_arrays may be modified when generating vace_video_mask_images
        fine_mask_arrays = [mask_arr.copy() for mask_arr in mask_arrays]
        
        # Convert images to numpy arrays
        img_arrays = [np.array(img) for img in img_images]
        depth_video_arrays = [np.array(img) for img in depth_video_images]
        # Prepare video frames (PIL.Image list)
        video_frames = img_images.copy()  # Already PIL.Image objects
        depth_video_frames = depth_video_images.copy()  # Already PIL.Image objects
        # Prepare vace_video_mask: Convert mask arrays to PIL.Image list (RGB, 0-255)
        # VACE unit expects PIL.Image list that will be preprocessed with min_value=0, max_value=1
        # Convert float32 [0,1] mask to uint8 [0,255] RGB images (3 channels for consistency)
        vace_video_mask_images = []
        if self.mask_strategy == "bbox_cover_traj":
            # find the bbox that can cover all the mask across frame
            y1_min = np.inf
            x1_min = np.inf
            y2_max = -np.inf
            x2_max = -np.inf
            for mask_arr in mask_arrays:
                ys, xs = np.where(mask_arr > 0)
                if len(ys) > 0 and len(xs) > 0:
                    y1, x1 = np.min(ys), np.min(xs)
                    y2, x2 = np.max(ys), np.max(xs)
                    y1 = int(y1 - (y2 - y1) * (self.bbox_scale - 1) / 2)
                    y2 = int(y2 + (y2 - y1) * (self.bbox_scale - 1) / 2)
                    x1 = int(x1 - (x2 - x1) * (self.bbox_scale - 1) / 2)
                    x2 = int(x2 + (x2 - x1) * (self.bbox_scale - 1) / 2)
                    y1_min = min(y1_min, y1)
                    x1_min = min(x1_min, x1)
                    y2_max = max(y2_max, y2)
                    x2_max = max(x2_max, x2)
            vace_video_mask_images = []
            for mask_arr in mask_arrays:
                mask_arr[y1_min:y2_max, x1_min:x2_max] = 1
                mask_rgb = np.stack([mask_arr.astype(np.uint8)*255] * 3, axis=-1)  # Shape: (H, W, 3)
                vace_video_mask_images.append(Image.fromarray(mask_rgb, mode='RGB'))
        else:
            for mask_arr in mask_arrays:
                if self.mask_strategy == "fine": # use the fine mask
                    pass
                elif self.mask_strategy == "bbox_with_traj":
                    # find the bbox of the mask
                    ys, xs = np.where(mask_arr > 0)
                    if len(ys) > 0 and len(xs) > 0:
                        y1, x1 = np.min(ys), np.min(xs)
                        y2, x2 = np.max(ys), np.max(xs)
                        y1 = int(y1 - (y2 - y1) * (self.bbox_scale - 1) / 2)
                        y2 = int(y2 + (y2 - y1) * (self.bbox_scale - 1) / 2)
                        x1 = int(x1 - (x2 - x1) * (self.bbox_scale - 1) / 2)
                        x2 = int(x2 + (x2 - x1) * (self.bbox_scale - 1) / 2)
                        mask_arr[y1:y2, x1:x2] = 1
                mask_rgb = np.stack([mask_arr.astype(np.uint8)*255] * 3, axis=-1)  # Shape: (H, W, 3)
                # Create RGB PIL.Image
                vace_video_mask_images.append(Image.fromarray(mask_rgb, mode='RGB'))
        if DEBUG_TIME:
            mask_proc_end = time.time()
            self._timing_stats['mask_processing'].append(mask_proc_end - mask_proc_start)
        
        # Timing: VACE video preparation
        if DEBUG_TIME:
            vace_prep_start = time.time()
        # Prepare vace_video
        if self.use_masked_vace_video:
            # Apply mask to images for vace_video
            vace_video_frames = []
            for img_arr, mask_arr in zip(img_arrays, mask_arrays):
                # Apply mask: keep masked area, set unmasked area to 0
                masked_img = img_arr.copy().astype(np.float32)
                # Expand mask to 3 channels
                mask_3d = np.stack([mask_arr] * 3, axis=-1)
                if self.mask_foreground:
                    mask_3d = 1 - mask_3d
                masked_img = masked_img * mask_3d
                masked_img = np.clip(masked_img, 0, 255).astype(np.uint8)
                vace_video_frames.append(Image.fromarray(masked_img))
        else:
            # Use original images for vace_video
            vace_video_frames = img_images.copy()
        
        # Prepare vace_reference_image (first frame)
        vace_reference_image = [img_images[0].copy()]  # Return as list for compatibility
        if DEBUG_TIME:
            vace_prep_end = time.time()
            self._timing_stats['vace_video_prep'].append(vace_prep_end - vace_prep_start)

        # Timing: Multiview rendering
        if DEBUG_TIME:
            multiview_start = time.time()
        # Prepare multi-view images using mesh render
        frame_index = os.path.splitext(os.path.basename(img_files[0]))[0]
        multiview_reference_images = self.mesh_loader.load_mesh_and_render(
            seq_name_instance_id, frame_index, frame_selecting_strategy=self.frame_selecting_strategy, render_config=self.render_configs, verbose=False, height=self.target_resolution[1], width=self.target_resolution[0]
        )
        # Post-process all images in multiview_reference_images
        processed_multiview_images = []
        for rendered in multiview_reference_images:
            # If image is torch tensor, convert to numpy
            if isinstance(rendered, torch.Tensor):
                rendered = rendered.detach().cpu().numpy()
            # If Image, convert to numpy
            if isinstance(rendered, Image.Image):
                rendered = np.array(rendered)
            # If RGBA, remove alpha channel
            if rendered.shape[-1] == 4:
                rendered = rendered[..., :3]
            # If float, normalize to uint8
            if np.issubdtype(rendered.dtype, np.floating):
                rendered = np.clip(rendered, 0.0, 1.0) * 255.0
            rendered = np.clip(rendered, 0, 255).astype(np.uint8)
            render_image = Image.fromarray(rendered)
            render_image = render_image.resize(self.target_resolution, Image.Resampling.BICUBIC)
            processed_multiview_images.append(render_image)
        multiview_reference_images = processed_multiview_images
        if DEBUG_TIME:
            multiview_end = time.time()
            self._timing_stats['multiview_render'].append(multiview_end - multiview_start)
        
        # # Render normal maps and position maps from multiple viewpoints
        # frame_index = self.mesh_loader.process_frame_index(seq_name_instance_id, frame_index, self.frame_selecting_strategy)
        # # Load mesh and camera parameters
        # mesh = self.mesh_loader.load_mesh(seq_name_instance_id, frame_index)
        # params = self.mesh_loader.load_camera_parameters(seq_name_instance_id, frame_index)
    
        normal_maps = []
        position_maps = []
        # for i, config in enumerate(self.render_configs):
        #     test_mesh = copy.deepcopy(mesh)
        #     if config.align_to_x_axis:
        #         test_mesh = self.mesh_loader.align_mesh_bbox_to_x_axis(test_mesh)
        #     normal_map = self.mesh_loader.render_normal_map(
        #         test_mesh, 
        #         params, 
        #         rotation_angle_deg=config.rotation_angle_deg,
        #         use_abs_coor=True,
        #         height=self.target_resolution[1],
        #         width=self.target_resolution[0]
        #     )
        #     normal_maps.append(normal_map)
        #     position_map = self.mesh_loader.render_position_map(
        #         test_mesh, 
        #         params, 
        #         rotation_angle_deg=config.rotation_angle_deg,
        #         height=self.target_resolution[1],
        #         width=self.target_resolution[0]
        #     )
        #     position_maps.append(position_map)

        # # Extract and render PTv3 features if enabled
        # ptv3_feature_maps = []
        # if self.enable_ptv3 and self.mesh_loader.enable_ptv3:
        #     for i, config in enumerate(self.render_configs):
        #         test_mesh = copy.deepcopy(mesh)
        #         if config.align_to_x_axis:
        #             test_mesh = self.mesh_loader.align_mesh_bbox_to_x_axis(test_mesh)
        #         ptv3_map = self.mesh_loader.render_ptv3_features(
        #             test_mesh,
        #             params,
        #             rotation_angle_deg=config.rotation_angle_deg,
        #             height=self.target_resolution[1],
        #             width=self.target_resolution[0]
        #         )
        #         if ptv3_map is not None:
        #             ptv3_feature_maps.append(ptv3_map)

        # Timing: Caption loading
        if DEBUG_TIME:
            caption_start = time.time()
        # Load prompt from caption file
        # Caption files are named: {major_class_name}_{instance_id}_{start_frame_idx:05d}_{end_frame_idx:05d}.txt
        prompt = None
        if self.caption_path is not None:
            instance_id = self.dataset_list[actual_index]['instance_id']
            
            # Get frame range from img_files
            # Extract frame indices from filenames (e.g., "00000.jpg" -> 0)
            first_frame_idx = int(os.path.splitext(os.path.basename(img_files[0]))[0])
            last_frame_idx = int(os.path.splitext(os.path.basename(img_files[-1]))[0])
            
            # Search for matching caption files
            # Pattern: {seq_name}_{instance_id}_*.txt
            caption_pattern = f"{seq_name}_{instance_id}_*.txt"
            matching_files = glob.glob(os.path.join(self.caption_path, caption_pattern))
            
            if matching_files:
                # Find the caption file with largest time overlap
                best_file = None
                best_overlap = -1
                
                for caption_file in matching_files:
                    # Extract frame range from filename
                    # Format: {seq_name}_{instance_id}_{start_frame_idx:05d}_{end_frame_idx:05d}.txt
                    # Parse from the end since seq_name might contain underscores
                    basename = os.path.basename(caption_file)
                    basename_no_ext = basename.replace(".txt", "")
                    
                    # Find the last two underscore-separated parts (should be frame numbers)
                    parts = basename_no_ext.split("_")
                    if len(parts) >= 2:
                        seg_end = int(parts[-1])
                        seg_start = int(parts[-2])
                        # Calculate overlap
                        overlap_start = max(first_frame_idx, seg_start)
                        overlap_end = min(last_frame_idx, seg_end)
                        overlap = max(0, overlap_end - overlap_start + 1)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_file = caption_file
                if best_file is not None:
                    with open(best_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Captions are separated by \n\n (2 long + 2 short)
                    captions = [c.strip() for c in content.split("\n\n") if c.strip()]
                    if captions:
                        if self.split_name == "val" or self.split_name == "val_no_filter":
                            prompt = captions[0]  # Use first caption for validation
                        else:
                            prompt = random.choice(captions)  # Randomly select one caption
                    else:
                        print(f"Warning: empty caption file {best_file}, use fallback prompt")
                        prompt = f"a video of {seq_name}"  # Fallback if file is empty

                else:
                    print(f"Warning: No caption file with valid overlap found for {seq_name} instance {instance_id}, use fallback prompt")
                    prompt = f"a video of {seq_name}"  # Fallback if no overlap found
            else:
                print(f"Warning: No caption files found matching pattern {caption_pattern}, use fallback prompt")
                prompt = f"a video of {seq_name}"  # Fallback if no files found
        else:
            prompt = f"a video of {seq_name}"  # Fallback if caption_path is not set
        if DEBUG_TIME:
            caption_end = time.time()
            self._timing_stats['caption_loading'].append(caption_end - caption_start)
        
        # # Generate prompt
        # prompt = self.prompt_template.format(class_name=seq_name)
        
        # Timing: Trajectory maps generation
        if DEBUG_TIME:
            traj_start = time.time()
        # Generate trajectory_maps based on trajectory_type
        # Use fine_mask_arrays to ensure fine mask is used even if mask_arrays were modified
        trajectory_maps = self._generate_trajectory_maps(fine_mask_arrays, min(self.target_resolution) // 100)
        if DEBUG_TIME:
            traj_end = time.time()
            self._timing_stats['trajectory_maps'].append(traj_end - traj_start)
        
        # Generate high frequency masks if enabled
        if DEBUG_TIME:
            high_freq_start = time.time()
        high_freq_mask_full = None
        high_freq_mask_in_mask = None
        if self.generate_high_freq_mask:
            # Convert video frames to tensor format [B, C, T, H, W]
            # video_frames are PIL.Image list, convert to tensor
            video_tensor_list = []
            for img in video_frames:
                img_array = np.array(img).astype(np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0  # Normalize to [0, 1]
                img_array = img_array * 2.0 - 1.0  # Convert to [-1, 1]
                video_tensor_list.append(torch.from_numpy(img_array).permute(2, 0, 1))  # [C, H, W]
            video_tensor = torch.stack(video_tensor_list, dim=2)  # [C, T, H, W]
            video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
            
            # Convert mask arrays to tensor
            mask_tensor_list = []
            for mask_arr in mask_arrays:
                mask_tensor_list.append(torch.from_numpy(mask_arr).unsqueeze(0))  # [1, H, W]
            mask_tensor = torch.stack(mask_tensor_list, dim=2)  # [1, T, H, W]
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, T, H, W]
            
            # Compute high frequency masks
            high_freq_mask_full = self.compute_high_freq_mask_pixel(video_tensor)  # [1, 1, T, H, W]
            high_freq_mask_in_mask = self.compute_high_freq_mask_in_mask_region(video_tensor, mask_tensor)  # [1, 1, T, H, W]
            
            # Convert back to numpy for storage (optional, can also keep as tensor)
            high_freq_mask_full = high_freq_mask_full.squeeze(0).cpu().numpy()  # [1, T, H, W]
            high_freq_mask_in_mask = high_freq_mask_in_mask.squeeze(0).cpu().numpy()  # [1, T, H, W]
        if DEBUG_TIME:
            high_freq_end = time.time()
            self._timing_stats['high_frequency_masks'].append(high_freq_end - high_freq_start)

        if DEBUG_TIME:
            # Calculate total time
            total_end = time.time()
            self._timing_stats['total'].append(total_end - total_start)
            
            # Print timing stats every [count] samples
            self._timing_counter += 1
            count = 1
            if self._timing_counter % count == 0:
                print(f"[Dataset Timing] Sample {self._timing_counter} - ", end="", flush=True)
                for key, times in self._timing_stats.items():
                    if times:
                        avg_time = np.mean(times[-count:]) * 1000  # Last [count] samples, convert to ms
                        print(f"{key}: {avg_time:.2f}ms | ", end="", flush=True)
                print("", flush=True)

        if self.downsample_10hz:
            video_frames = video_frames[::2] # downsample by 2
            depth_video_frames = depth_video_frames[::2] # downsample by 2
            vace_video_frames = vace_video_frames[::2] # downsample by 2
            vace_video_mask_images = vace_video_mask_images[::2] # downsample by 2
            trajectory_maps = trajectory_maps[::2] # downsample by 2
            if self.generate_high_freq_mask:
                high_freq_mask_full = high_freq_mask_full[::2] # downsample by 2
                high_freq_mask_in_mask = high_freq_mask_in_mask[::2] # downsample by 2

            pading_length = self.length - len(video_frames)
            # pad_with black frames
            if pading_length > 0:
                video_frames = video_frames + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                depth_video_frames = depth_video_frames + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                vace_video_frames = vace_video_frames + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                vace_video_mask_images = vace_video_mask_images + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                trajectory_maps = trajectory_maps + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                if self.generate_high_freq_mask:
                    high_freq_mask_full = high_freq_mask_full + [np.zeros((self.target_resolution[1], self.target_resolution[0]))] * pading_length
                    high_freq_mask_in_mask = high_freq_mask_in_mask + [np.zeros((self.target_resolution[1], self.target_resolution[0]))] * pading_length
        
        result = {
            "seq_name": seq_name,
            "instance_id": instance_id,
            "video": video_frames,  # List of PIL.Image
            "depth_video": depth_video_frames,  # List of PIL.Image
            "prompt": prompt,  # String
            "vace_video": vace_video_frames,  # List of PIL.Image
            "vace_video_mask": vace_video_mask_images,  # List of PIL.Image (grayscale, L mode)
            "vace_reference_image": vace_reference_image,  # List of PIL.Image (single element)
            "multiview_reference_image": multiview_reference_images,  # List of PIL.Image (len = num of view)
            "normal_maps": normal_maps,  # List of PIL.Image (len = num of view)
            "position_maps": position_maps,  # List of PIL.Image (len = num of view)
            "trajectory_maps": trajectory_maps,  # List of PIL.Image (RGB, same length as video frames)
        }
        
        # Add high frequency masks if generated
        if self.generate_high_freq_mask:
            result["high_freq_mask_full"] = high_freq_mask_full  # numpy array [1, T, H, W]
            result["high_freq_mask_in_mask"] = high_freq_mask_in_mask  # numpy array [1, T, H, W]
            # Save the multiview_reference_image frame count for later mapping to latent space.
            result["num_multiview_frames"] = len(multiview_reference_images) if multiview_reference_images else 0
        
        return result


if __name__ == "__main__":
    # Test the dataset
    # dataset = DAVIS17_VACE_Dataset(
    #     dataset_path=os.getenv("DAVIS17_DATASET_PATH"),
    #     split_name="train",
    #     length=81,
    #     target_resolution=(832, 480),
    #     min_instance_ratio=0.00,
    #     min_bbox_ratio=0.00,
    #     use_masked_vace_video=True,
    #     dataset_repeat=1,
    #     trajectory_type="mask",
    #     sparse_box_interval=5,
    #     bbox_scale=1.0,
    #     frame_selecting_strategy="farthest",
    #     mask_foreground=True,
    #     mask_strategy="bbox_with_traj",
    # )

    # for i in tqdm(range(len(dataset))):
    #     sample = dataset[i]

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    dataset = DAVIS17_VACE_Dataset(
        dataset_path=os.getenv("DAVIS17_DATASET_PATH"),
        split_name="train",
        length=81,
        target_resolution=(832, 480),
        min_instance_ratio=0.00,
        min_bbox_ratio=0.00,
        use_masked_vace_video=True,
        dataset_repeat=1,
        trajectory_type="mask",
        sparse_box_interval=5,
        bbox_scale=1.0,
        frame_selecting_strategy="farthest",
        mask_foreground=True,
        mask_strategy="bbox_with_traj",
        enable_data_augmentation=False,
    )

    # for i in tqdm(range(len(dataset))):
    #     sample = dataset[i]
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test getitem
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Seq name: {sample['seq_name']}")
        print(f"Instance id: {sample['instance_id']}")
        print(f"Video frames: {len(sample['video'])}")
        print(f"Video frame size: {sample['video'][0].size}")
        print(f"Video frame mode: {sample['video'][0].mode}")
        print(f"Vace video frames: {len(sample['vace_video'])}")
        print(f"Vace video frame size: {sample['vace_video'][0].size}")
        print(f"Vace video frame mode: {sample['vace_video'][0].mode}")
        print(f"Depth video frames: {len(sample['depth_video'])}")
        print(f"Depth video frame size: {sample['depth_video'][0].size}")
        print(f"Depth video frame mode: {sample['depth_video'][0].mode}")
        print(f"Prompt: {sample['prompt']}")
        print(f"VACE video frames: {len(sample['vace_video'])}")
        print(f"VACE reference image: {len(sample['vace_reference_image'])}")
        print(f"VACE reference image size: {sample['vace_reference_image'][0].size}")
        print(f"multiview reference: {len(sample['multiview_reference_image'])}")
        print(f"multiview reference size: {sample['multiview_reference_image'][0].size}")
        print(f"trajectory maps: {len(sample['trajectory_maps'])}")
        print(f"trajectory map size: {sample['trajectory_maps'][0].size}")
        print(f"trajectory map mode: {sample['trajectory_maps'][0].mode}")
        print(F"vace video mask: {len(sample['vace_video_mask'])}")
        print(f"vace video mask size: {sample['vace_video_mask'][0].size}")
        print(f"vace video mask mode: {sample['vace_video_mask'][0].mode}")
        # save first 10 frames of each sequence for debug
        if not os.path.exists(f"./debug_train"):
            os.makedirs(f"./debug_train")
        with open(f"./debug_train/prompt.txt", "w", encoding="utf-8") as f:
            f.write(sample['prompt'])
        # Save first 10 video frames
        for i, img in enumerate(sample['video'][:10]):
            img.save(f"./debug_train/video_{i}.png")
        # Save first 10 depth video frames
        for i, img in enumerate(sample['depth_video'][:10]):
            img.save(f"./debug_train/depth_video_{i}.png")
        # Save first 10 vace video frames
        for i, img in enumerate(sample['vace_video'][:10]):
            img.save(f"./debug_train/vace_video_{i}.png")
        # Save reference image
        sample['vace_reference_image'][0].save(f"./debug_train/reference_image.png")
        # Save all multiview references (usually small list)
        for i, img in enumerate(sample['multiview_reference_image']):
            img.save(f"./debug_train/multiview_reference_image_{i}.png")
        # Save first 10 vace video masks
        for i, img in enumerate(sample['vace_video_mask'][:10]):
            img.save(f"./debug_train/mask_{i}.png")
        # Save first 10 normal maps
        for i, img in enumerate(sample['normal_maps'][:10]):
            img.save(f"./debug_train/normal_map_{i}.png")
        # Save first 10 position maps
        for i, img in enumerate(sample['position_maps'][:10]):
            img.save(f"./debug_train/position_map_{i}.png")
        # Save first 10 trajectory maps
        if 'trajectory_maps' in sample:
            for i, img in enumerate(sample['trajectory_maps'][:10]):
                img.save(f"./debug_train/trajectory_map_{i}.png")

        if DEBUG_TIME:
            for key, times in dataset._timing_stats.items():
                if times:
                    avg_time = np.mean(times[-10:]) * 1000  # Last 10 samples, convert to ms
                    print(f"{key}: {avg_time:.2f}ms | ", end="", flush=True)
            print("", flush=True)

        # load_images: 168.45ms | resize: 6389.13ms | data_augmentation: 0.01ms | 
        # mask_processing: 408.01ms | vace_video_prep: 345.87ms | multiview_render: 864.46ms | 
        # trajectory_maps: 147.98ms | caption_loading: 1.09ms | high_frequency_masks: 3370.07ms | 
        # depth conversion: 169.46ms | total: 11864.90ms | 
