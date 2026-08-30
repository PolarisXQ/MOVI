from PIL import Image
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import glob
import json
from tqdm import tqdm  # type: ignore
import torch
import copy
import random
import time
from diffsynth.trainers.mesh_util import MeshLoader, RenderConfig
from diffsynth.trainers.aug_utils import (
    apply_horizontal_flip,
    apply_vertical_flip,
    apply_crop,
    apply_gaussian_blur,
    apply_fft_high_freq,
)
DEBUG_TIME = False

CATEGORIES = ['person', 'car', 'wheeled_machine', 'ship_or_boat', 'truck', 'cat', 'other_animal', 'dog', 'airplane', 'horse', 'bus', 'cattle', 'raft', 'motorcycle', 'bicycle']

def _load_ignore_list(ignore_list_path: str) -> set:
    """Load ignore list from file. Returns set of video_seq/class_name/instance_id strings."""
    ignore_set = set()
    if os.path.exists(ignore_list_path):
        try:
            with open(ignore_list_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        ignore_set.add(line)
        except Exception as e:
            print(f"Warning: Failed to load ignore list from {ignore_list_path}: {e}")
    return ignore_set

def _compute_bbox_ratio(mask: np.ndarray) -> float:
    bbox = np.where(mask > 0)
    if bbox[0].size > 0:    
        x1, y1, x2, y2 = bbox[0].min(), bbox[1].min(), bbox[0].max(), bbox[1].max()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        return (w * h / (mask.shape[1] * mask.shape[0])).tolist()
    else:
        return 0.0

def _load_vipseg_categories(categories_file: str) -> Dict[int, str]:
    try:
        with open(categories_file, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        return {cat['id']: cat['name'] for cat in categories}
    except Exception as e:
        print(f"Failed to load categories file: {e}")
        return {}

split_names = ["train", "val", "all", "val_no_filter"]
class VIPSeg_origin_mix_dataloader:
    def __init__(self, 
    vipseg_dataset_path: str=os.getenv("VIPSEG_DATASET_PATH"), 
    vspw_dataset_path: str=os.getenv("VSPW_DATASET_PATH"), 
    split_name: str="all", 
    length: int=81, 
    target_resolution: tuple[int, int]=(480, 270), 
    min_instance_ratio: float=0.10, min_bbox_ratio: float=0.10,
    frame_selecting_strategy: str = "farthest",  # select from ["nearest", "farthest"]
    use_masked_vace_video: bool = True,  # if True, vace_video uses masked image; otherwise uses original image
    dataset_repeat: int = 1,
    max_samples: Optional[int] = None,
    mask_foreground: bool = True,
    mask_strategy: str = "bbox_with_traj",  # select from ["fine", "bbox_with_traj", "bbox_cover_traj"]
    bbox_scale: float = 1.2,  # bbox_scale times the size of the bbox
    trajectory_type: str = "mask",  # select from ["mask", "box", "sparse_box"]
    sparse_box_interval: int = 10,  # interval for sparse_box (e.g., 5 means keep box every 5 frames)
    enable_data_augmentation: bool = True,  # whether to enable data augmentation
    augmentation_no_aug_prob: float = 0.2,  # probability of not applying augmentation (20%)
    augmentation_blur_prob: float = 0.5,  # probability of applying blur (when augmentation is enabled)
    augmentation_device: str = 'cuda',  # device for FFT
    downsample_10hz: bool = False, 
    **kwargs,  # Additional args
    ):
        self.vipseg_dataset_path = vipseg_dataset_path
        self.vspw_dataset_path = os.path.join(vspw_dataset_path,"data")
        self.mask_path = os.path.join(vspw_dataset_path, "Cutie-720p-30hz-panomasks")
        self.split_name = split_name
        assert split_name in split_names, "split_name must be in " + str(split_names)
        self.length = length
        self.target_resolution = target_resolution
        self.min_instance_ratio = min_instance_ratio
        self.min_bbox_ratio = min_bbox_ratio
        self.frame_selecting_strategy = frame_selecting_strategy
        self.use_masked_vace_video = use_masked_vace_video
        self.dataset_repeat = dataset_repeat
        self.max_samples = max_samples
        self.mask_foreground = mask_foreground
        self.mask_strategy = mask_strategy
        self.bbox_scale = bbox_scale
        self.trajectory_type = trajectory_type
        assert trajectory_type in ["mask", "box", "sparse_box"], f"trajectory_type must be one of ['mask', 'box', 'sparse_box'], got {trajectory_type}"
        self.sparse_box_interval = sparse_box_interval
        self.enable_data_augmentation = enable_data_augmentation
        self.augmentation_no_aug_prob = augmentation_no_aug_prob
        self.augmentation_blur_prob = augmentation_blur_prob
        self.augmentation_device = augmentation_device if torch.cuda.is_available() and augmentation_device == 'cuda' else 'cpu'
        self.downsample_10hz = downsample_10hz


        self.CLASSES = _load_vipseg_categories(os.path.join(vipseg_dataset_path, "VIPSeg_720P", "panoVIPSeg_categories.json"))
        self.CLASSES_INDEX = {v: k for k, v in self.CLASSES.items()}
        
        caption_path = os.getenv("VIPSEG_CAPTION_PATH")
        if os.path.exists(caption_path):
            self.caption_path = caption_path
        else:
            print(f"Warning: Caption path {caption_path} does not exist, captions will use fallback prompts")
            self.caption_path = None
        
        mesh_results_path = os.getenv("VIPSEG_V2M4_RESULTS_PATH")
        if mesh_results_path is None:
            print("Warning: VIPSEG_V2M4_RESULTS_PATH is not set, please source path_setup.sh first!, mesh loading will be disabled")
            self.mesh_loader = None
        else:
            self.mesh_loader = MeshLoader(
                data_root=mesh_results_path,
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
        
        assert os.getenv("VIPSEG_DATASET_IGNORE_LIST_PATH") is not None, "VIPSEG_DATASET_IGNORE_LIST_PATH is not set, please source path_setup.sh first!"
        ignore_list_path = os.getenv("VIPSEG_DATASET_IGNORE_LIST_PATH")
        self.ignore_set = _load_ignore_list(ignore_list_path)
        if self.ignore_set:
            print(f"Loaded {len(self.ignore_set)} entries from ignore list: {ignore_list_path}")

        def _should_ignore_sample(video_seq: str, class_name: str, instance_id: int) -> bool:
            """Check if sample matches ignore list entry (format: video_seq/class_name/instance_id)"""
            ignore_key = f"{video_seq}/{class_name}/{instance_id}"
            return ignore_key in self.ignore_set
        
        assert os.getenv("VIPSEG_DEPTH_PATH") is not None, "VIPSEG_DEPTH_PATH is not set, please source path_setup.sh first!"
        self.depth_data_path = os.getenv("VIPSEG_DEPTH_PATH")
        assert os.getenv("VIPSEG_DATASET_JSON_PATH") is not None, "VIPSEG_DATASET_JSON_PATH is not set, please source path_setup.sh first!"
        json_path_env = os.getenv("VIPSEG_DATASET_JSON_PATH")
        json_file_path = os.path.join(json_path_env, "dataset_list_" + self.split_name + "_" + str(self.length)+ "_" + str(round( self.min_instance_ratio*100, 2)) + "_" + str(round( self.min_bbox_ratio*100, 2)) + ".json")
        if os.path.exists(json_file_path):
            self.dataset_list = json.load(open(json_file_path, "r"))
            original_length = len(self.dataset_list)
            filtered_list = []
            for item in self.dataset_list:
                if item.get('major_class') not in CATEGORIES:
                    continue
                video_seq = item.get('video_seq', '')
                if video_seq == '':
                    video_seq = item['img_files'][0].split('/')[-3]
                class_name = item.get('major_class')
                instance_id = item.get('instance_id')
                if not _should_ignore_sample(video_seq, class_name, instance_id):
                    filtered_list.append(item)
            self.dataset_list = filtered_list
            print("data loaded from json file")
            print(f"total dataset length: {len(self.dataset_list)} (filtered from {original_length} to only include {CATEGORIES} and exclude ignore list)")
            self._apply_max_samples()
            return
        else:
            print("no json file found")


            split_folder_names = glob.glob(os.path.join(self.mask_path, "*"))
            split_folder_names = [os.path.basename(cls) for cls in split_folder_names]
            split_folder_names = [cls for cls in split_folder_names if os.path.exists(os.path.join(self.vspw_dataset_path, cls, "map_dict.json"))]

            if split_name == "val" or split_name == "train" or split_name == "val_no_filter":
                split_file = os.path.join(self.vipseg_dataset_path, "VIPSeg_720P", split_name+".txt")
                if split_name == "val_no_filter":
                    split_file = os.path.join(self.vipseg_dataset_path, "VIPSeg_720P", "val.txt")
                with open(split_file, "r") as f:
                    split_folder_names_sf = f.read().splitlines()
                split_folder_names = [cls for cls in split_folder_names if cls in split_folder_names_sf]




            self.split_folder_names = split_folder_names

            print("preparing data...")
            seq_iter = []
            for split_folder_name in self.split_folder_names:
                seq_name = split_folder_name.split("/")[-1]
                mask_files = glob.glob(os.path.join(self.mask_path, seq_name, "*.png"))
                if not mask_files:
                    continue
                img_files = glob.glob(os.path.join(self.vspw_dataset_path, seq_name,"origin", "*.jpg"))
                mask_files.sort()
                img_files.sort()
                mask_files_index = [int(os.path.basename(mask_file).split(".")[0]) for mask_file in mask_files]
                img_files_index = [int(os.path.basename(img_file).split(".")[0]) for img_file in img_files]
                img_start = img_files_index.index(mask_files_index[0])
                img_end = img_files_index.index(mask_files_index[-1])
                img_files = img_files[img_start:img_end+1]
                
                if len(img_files) < length:
                    continue
                seq_iter.append((seq_name, mask_files, img_files))
            seq_iter =tqdm(seq_iter, desc="preparing data", unit="seq") 
            self.dataset_list = []
            for seq_name, mask_files, img_files in seq_iter:
                with Image.open(mask_files[0]) as m0:
                    img_w, img_h = m0.size

                instance_ratios_dict: Dict[int, List[float]] = {}
                # instance_start_end: Dict[int, Tuple[int, int]] = {}  # inst_id -> (start_frame of the image, end_frame of the image)
                instance_mask_start_end: Dict[int, Tuple[int, int]] = {}  # inst_id -> (start_frame of the mask, end_frame of the mask)
                bbox_ratios_dict: Dict[int, List[float]] = {} 
                frame_iter = tqdm(mask_files, desc=f'{seq_name} masks', unit='frame', leave=False)

                for mf in frame_iter:
                    with Image.open(mf) as m:
                        
                        mask_array = np.array(m)
                
                        # Handle different mask formats, consistently with create_panoptic_video_labels.py.
                        if mask_array.ndim == 3:
                            # RGB mask: convert to instance IDs (consistent with lines 91–92 of create_panoptic_video_labels.py).
                            gt_pan = np.uint32(mask_array)
                            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
                        else:
                            # Grayscale mask: use directly.
                            pan_gt = mask_array.astype(np.uint32)
                        
                        # Find all instance IDs.
                        unique_ids = np.unique(pan_gt)
                        unique_ids = unique_ids[unique_ids > 0]  # Exclude the background.
                        
                        if img_w <= 0 or img_h <= 0:
                            h, w = pan_gt.shape[:2]
                            denom = float(w * h)
                        else:
                            denom = float(img_w * img_h)
                        if denom <= 0:
                            continue
                        
                        for inst_id in unique_ids.tolist():
                            mask = (pan_gt == inst_id)
                            area = int(mask.sum())
                            if area <= 0:
                                continue
                            ratio = area / denom
                            

                            if inst_id not in instance_ratios_dict:
                                instance_ratios_dict[inst_id] = []
                            instance_ratios_dict[inst_id].append(ratio)
                            
                            #cal the bbox ratio of the mask
                            bbox_ratio = _compute_bbox_ratio(mask)
                            if inst_id not in bbox_ratios_dict:
                                bbox_ratios_dict[inst_id] = []
                            # if inst_id not in instance_start_end:
                            #     current_image = map_dict[str(int(os.path.basename(mf).split(".")[0]))]
                            #     instance_start_end[inst_id] = (current_image, current_image)
                            # else:
                            #     current_image = map_dict[str(int(os.path.basename(mf).split(".")[0]))]
                            #     instance_start_end[inst_id] = (min(instance_start_end[inst_id][0], current_image), max(instance_start_end[inst_id][1], current_image))
                            if inst_id not in instance_mask_start_end:
                                instance_mask_start_end[inst_id] = (os.path.basename(mf), os.path.basename(mf))
                            else:
                                instance_mask_start_end[inst_id] = (min(instance_mask_start_end[inst_id][0], os.path.basename(mf)), max(instance_mask_start_end[inst_id][1], os.path.basename(mf)))
                            bbox_ratios_dict[inst_id].append(bbox_ratio)
                            assert bbox_ratio >= ratio

            # Compute statistics for each instance.
                for inst_id in instance_ratios_dict.keys():
                    inst_ratios = instance_ratios_dict[inst_id]
                    bbox_ratios = bbox_ratios_dict[inst_id]
                    if inst_ratios:
                        inst_mean = float(np.mean(inst_ratios))
                        bbox_mean = float(np.mean(bbox_ratios))
                        if inst_mean < self.min_instance_ratio or bbox_mean < self.min_bbox_ratio:
                            continue
                        if self.split_name == 'val_no_filter':
                            if inst_id == 0:
                                continue  # Skip the background.
                            if inst_id < 125:
                                semantic_id = inst_id
                            else:
                                semantic_id = inst_id // 100
                            category_id = semantic_id - 1  # Consistent with line 68 of create_panoptic_video_labels.py.
                            category_name = self.CLASSES[category_id]
                            # Only include samples that belong to CATEGORIES
                            if category_name not in CATEGORIES:
                                continue
                            mask_files_slice = mask_files
                            img_files_slice = img_files
                            seq_start = img_files[0].split("/")[-1].split(".")[0]
                            self.dataset_list.append({
                                'instance_id': inst_id,
                                'major_class': category_name,
                                'video_seq': seq_name,  # Store sequence name for mesh and caption loading
                                'img_w': int(img_w),
                                'img_h': int(img_h),
                                'mask_files': mask_files_slice,
                                'img_files': img_files_slice,
                                'seq_start': int(seq_start)
                            })
                        else:
                            for i in range(len(mask_files)-self.length+1):
                                if inst_id == 0:
                                    continue  # Skip the background.
                                if inst_id < 125:
                                    semantic_id = inst_id
                                else:
                                    semantic_id = inst_id // 100
                                category_id = semantic_id - 1  # Consistent with line 68 of create_panoptic_video_labels.py.
                                category_name = self.CLASSES[category_id]
                                # Only include samples that belong to CATEGORIES
                                if category_name not in CATEGORIES:
                                    continue
                                mask_files_slice = mask_files[i:i+self.length]
                                img_files_slice = img_files[i:i+self.length]
                                seq_start = img_files[0].split("/")[-1].split(".")[0]
                                self.dataset_list.append({
                                    'instance_id': inst_id,
                                    'major_class': category_name,
                                    'video_seq': seq_name,  # Store sequence name for mesh and caption loading
                                    'img_w': int(img_w),
                                    'img_h': int(img_h),
                                    'mask_files': mask_files_slice,
                                    'img_files': img_files_slice,
                                    'seq_start': int(seq_start)
                                })

            print("data prepared")
            print(f"total dataset length: {len(self.dataset_list)} (filtered to only include {CATEGORIES})")
            print("save prepared data to json file")
            json_file_path = os.path.join(os.getenv("VIPSEG_DATASET_JSON_PATH"), "dataset_list_" + self.split_name + "_" + str(self.length)+ "_" + str(round( self.min_instance_ratio*100, 2)) + "_" + str(round( self.min_bbox_ratio*100, 2)) + ".json")
            with open(json_file_path, "w") as f:
                json.dump(self.dataset_list, f)
            print("data saved to json file")
            self._apply_max_samples()

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
                    'other': [],
                    'total': []
                }
                self._timing_counter = 0
        
        if DEBUG_TIME:
            total_start = time.time()
        
        seq_name = self.dataset_list[index]['major_class']
        instance_id = self.dataset_list[index]['instance_id']
        mask_files = self.dataset_list[index]['mask_files']
        img_files = self.dataset_list[index]['img_files']
        seq_start = self.dataset_list[index]['seq_start']
        # Get video_seq from dataset_list if available, otherwise extract from files
        if 'video_seq' in self.dataset_list[index]:
            video_seq = self.dataset_list[index]['video_seq']
        else:
            # Fallback: Extract video_seq from first non-None mask file or first img file
            video_seq = None
            for mf in mask_files:
                if mf is not None:
                    video_seq = mf.split("/")[-2]
                    break
            if video_seq is None:
                video_seq = img_files[0].split("/")[-2]
        
        # Construct depth video file paths
        # depth_video_files = [img_file.replace(self.vspw_dataset_path, self.depth_data_path).replace(".jpg", ".png") for img_file in img_files]
        
        # Verify mask, img, and depth files match
        for i in range(len(mask_files)):
            if mask_files[i] is not None:
                assert mask_files[i].split("/")[-1].split(".")[0] == img_files[i].split("/")[-1].split(".")[0]
            # assert img_files[i].split("/")[-1].split(".")[0] == depth_video_files[i].split("/")[-1].split(".")[0]
        
        # Timing: Load images
        if DEBUG_TIME:
            load_start = time.time()
        mask_images = []
        for i in range(len(mask_files)):
            if mask_files[i] is not None:
                mask_img = Image.open(mask_files[i])
                mask_img = mask_img.resize(self.target_resolution, Image.Resampling.NEAREST)
                mask_images.append(mask_img)
            else:
                mask_images.append(None)
        img_images = [Image.open(img_file) for img_file in img_files]
        # depth_video_images = [Image.open(depth_video_file) for depth_video_file in depth_video_files]
        if DEBUG_TIME:
            load_end = time.time()
            self._timing_stats['load_images'].append(load_end - load_start)
        
        # Timing: Resize
        if DEBUG_TIME:
            resize_start = time.time()
        mask_images = [mf.resize(self.target_resolution, Image.Resampling.NEAREST) if mf is not None else None for mf in mask_images]
        img_images = [img.resize(self.target_resolution, Image.Resampling.BICUBIC) for img in img_images]
        # depth_video_images = [img.resize(self.target_resolution, Image.Resampling.BICUBIC) for img in depth_video_images]
        if DEBUG_TIME:
            resize_end = time.time()
            self._timing_stats['resize'].append(resize_end - resize_start)
        
        # # Process depth arrays (normalize if needed, convert to RGB)
        # depth_arrays = [np.array(img) for img in depth_video_images]
        # if depth_arrays[0].dtype == np.uint16 or depth_arrays[0].max() > 255:
        #     # Find global min/max across all frames for consistent scaling
        #     global_min = 0
        #     global_max = 65535
        #     # Normalize all frames using the same scale
        #     converted_depth_images = []
        #     for depth_array in depth_arrays:
        #         if global_max > global_min:
        #             depth_normalized = ((depth_array - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        #         else:
        #             depth_normalized = np.zeros_like(depth_array, dtype=np.uint8)
        #         # Create RGB image (grayscale: same value in all 3 channels)
        #         depth_rgb = np.stack([depth_normalized] * 3, axis=-1)
        #         converted_depth_images.append(Image.fromarray(depth_rgb, mode='RGB'))
        #     depth_video_images = converted_depth_images
        # else:
        #     # Already 8-bit or less, just convert mode if needed
        #     depth_video_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in depth_video_images]
        
        # Timing: Data augmentation
        if DEBUG_TIME:
            aug_start = time.time()
        # Apply data augmentation if enabled (apply to all frames consistently)
        if self.enable_data_augmentation and self.split_name == "train":
            # Apply no augmentation 20% of the time and augmentation 80% of the time.
            if random.random() >= self.augmentation_no_aug_prob:
                # Choose a geometric transform from flip, crop, and FFT.
                geometric_aug = random.choice(['flip_h', 'flip_v', 'crop', 'fft'])
                # Determine whether to apply blur (never with FFT).
                apply_blur = random.random() < self.augmentation_blur_prob if geometric_aug != 'fft' else False
                
                # Apply the same augmentation to all frames.
                augmented_img_images = []
                augmented_mask_images = []
                # augmented_depth_video_images = []
                
                # for img, mask, depth_video in zip(img_images, mask_images, depth_video_images):
                for img, mask in zip(img_images, mask_images):
                    # Skip masks that are None.
                    if mask is None:
                        augmented_img_images.append(img)
                        augmented_mask_images.append(None)
                        # augmented_depth_video_images.append(depth_video)
                        continue
                    
                    # Apply the geometric transform.
                    if geometric_aug == 'flip_h':
                        aug_img, aug_mask = apply_horizontal_flip(img, mask)
                        # aug_depth_video, aug_depth_video_mask = apply_horizontal_flip(depth_video, mask)
                    elif geometric_aug == 'flip_v':
                        aug_img, aug_mask = apply_vertical_flip(img, mask)
                        # aug_depth_video, aug_depth_video_mask = apply_vertical_flip(depth_video, mask)
                    elif geometric_aug == 'crop':
                        aug_img, aug_mask = apply_crop(img, mask)
                        # aug_depth_video, aug_depth_video_mask = apply_crop(depth_video, mask)
                    elif geometric_aug == 'fft':
                        aug_img, aug_mask = apply_fft_high_freq(img, mask, device=self.augmentation_device)
                        # aug_depth_video, aug_depth_video_mask = apply_fft_high_freq(depth_video, mask, device=self.augmentation_device)
                    else:
                        aug_img, aug_mask = img, mask
                        # aug_depth_video, aug_depth_video_mask = depth_video, mask
                    
                    # Apply blur if needed (never with FFT).
                    if apply_blur:
                        aug_img, aug_mask = apply_gaussian_blur(aug_img, aug_mask)
                        # aug_depth_video, aug_depth_video_mask = apply_gaussian_blur(aug_depth_video, aug_depth_video_mask)
                    
                    augmented_img_images.append(aug_img)
                    augmented_mask_images.append(aug_mask)
                    # augmented_depth_video_images.append(aug_depth_video)
                
                img_images = augmented_img_images
                mask_images = augmented_mask_images
                # depth_video_images = augmented_depth_video_images
        if DEBUG_TIME:
            aug_end = time.time()
            self._timing_stats['data_augmentation'].append(aug_end - aug_start)
        
        # Timing: Mask processing
        if DEBUG_TIME:
            mask_proc_start = time.time()
        # Convert mask to numpy array and filter by instance_id
        mask_arrays = []
        for mf in mask_images:
            if mf is not None:
                mask_arr = np.array(mf)
                # Handle 3D masks (take first channel if needed)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]
                # Convert to binary mask (1 for instance_id, 0 otherwise)
                mask_arr = (mask_arr == instance_id).astype(np.float32)
                mask_arrays.append(mask_arr)
            else:
                # Create empty mask for None entries
                mask_arrays.append(np.zeros(self.target_resolution[::-1], dtype=np.float32))
        
        # Save a copy of fine masks for trajectory_maps generation
        # This is necessary because mask_arrays may be modified when generating vace_video_mask_images
        fine_mask_arrays = [mask_arr.copy() for mask_arr in mask_arrays]
        
        # Convert images to numpy arrays
        img_arrays = [np.array(img) for img in img_images]
        # depth_video_arrays = [np.array(img) for img in depth_video_images]
        
        # Prepare video frames (PIL.Image list)
        video_frames = img_images.copy()  # Already PIL.Image objects
        # depth_video_frames = depth_video_images.copy()  # Already PIL.Image objects
        
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
                    h, w = y2 - y1, x2 - x1
                    y1 = int(y1 - h * (self.bbox_scale - 1) / 2)
                    y2 = int(y2 + h * (self.bbox_scale - 1) / 2)
                    x1 = int(x1 - w * (self.bbox_scale - 1) / 2)
                    x2 = int(x2 + w * (self.bbox_scale - 1) / 2)
                    y1_min = min(y1_min, y1)
                    x1_min = min(x1_min, x1)
                    y2_max = max(y2_max, y2)
                    x2_max = max(x2_max, x2)
            vace_video_mask_images = []
            if y1_min < np.inf:  # Only if we found valid bbox
                # Clip to image boundaries
                H, W = mask_arrays[0].shape
                y1_min = max(0, min(int(y1_min), H - 1))
                y2_max = max(0, min(int(y2_max), H - 1))
                x1_min = max(0, min(int(x1_min), W - 1))
                x2_max = max(0, min(int(x2_max), W - 1))
            for mask_arr in mask_arrays:
                if y1_min < np.inf:  # Only if we found valid bbox
                    mask_arr[y1_min:y2_max+1, x1_min:x2_max+1] = 1
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
                        h, w = y2 - y1, x2 - x1
                        y1 = int(y1 - h * (self.bbox_scale - 1) / 2)
                        y2 = int(y2 + h * (self.bbox_scale - 1) / 2)
                        x1 = int(x1 - w * (self.bbox_scale - 1) / 2)
                        x2 = int(x2 + w * (self.bbox_scale - 1) / 2)
                        # Clip to image boundaries
                        H, W = mask_arr.shape
                        y1 = max(0, min(y1, H - 1))
                        y2 = max(0, min(y2, H - 1))
                        x1 = max(0, min(x1, W - 1))
                        x2 = max(0, min(x2, W - 1))
                        mask_arr[y1:y2+1, x1:x2+1] = 1
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
       
        # Timing: Caption loading
        if DEBUG_TIME:
            caption_start = time.time()
        # Load prompt from caption file
        # Caption files are named: {video_name}_{class_name}_{instance_id}_{seg_start_frame}_{seg_end_frame}.txt
        prompt = None
        if self.caption_path is not None:
            
            # Get frame range from img_files
            # Extract frame indices from filenames (e.g., "00000001.jpg" -> 1)
            first_frame_idx = int(os.path.splitext(os.path.basename(img_files[0]))[0])
            last_frame_idx = int(os.path.splitext(os.path.basename(img_files[-1]))[0])
            
            # Extract video_name from video_seq (same logic as qwenvl-caption-VIPSeg.py line 298)
            video_name = video_seq.split("/")[-1].split(".")[0] if "/" in video_seq else video_seq.split(".")[0]
            
            # Search for matching caption files
            # Pattern: {video_name}_{class_name}_{instance_id}_*.txt
            caption_pattern = f"{video_name}_{seq_name}_{instance_id}_*.txt"
            matching_files = glob.glob(os.path.join(self.caption_path, caption_pattern))
            
            if matching_files:
                # Find the caption file with largest time overlap
                best_file = None
                best_overlap = -1
                
                for caption_file in matching_files:
                    # Extract frame range from filename
                    # Format: {video_name}_{class_name}_{instance_id}_{seg_start_frame}_{seg_end_frame}.txt
                    # Parse from the end since class_name might contain underscores
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
                    # print(f"Using caption file {best_file} for {video_seq} instance {instance_id}, frane start: {first_frame_idx}, frame end: {last_frame_idx}")
                    with open(best_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Captions are separated by \n\n (2 long + 2 short)
                    captions = [c.strip() for c in content.split("\n\n") if c.strip()]
                    if captions:
                        if self.split_name == "val":
                            prompt = captions[0]  # Use first caption for validation
                        else:
                            prompt = random.choice(captions)  # Randomly select one caption
                    else:
                        print(f"Warning: empty caption file {best_file}, use fallback prompt")
                        prompt = f"a video of {seq_name}"  # Fallback if file is empty

                else:
                    print(f"Warning: No caption file with valid overlap found for {video_seq} instance {instance_id}, use fallback prompt")
                    prompt = f"a video of {seq_name}"  # Fallback if no overlap found
            else:
                print(f"Warning: No caption files found matching pattern {self.caption_path}/{caption_pattern}, use fallback prompt")
                prompt = f"a video of {seq_name}"  # Fallback if no files found
        else:
            prompt = f"a video of {seq_name}"  # Fallback if caption_path is not set, please source path_setup.sh first!
        if DEBUG_TIME:
            caption_end = time.time()
            self._timing_stats['caption_loading'].append(caption_end - caption_start)
        
        # Timing: Multiview rendering
        if DEBUG_TIME:
            multiview_start = time.time()
        # Prepare multiview images using mesh render
        multiview_reference_images = []
        normal_maps = []
        position_maps = []
        
        if self.mesh_loader is not None:
            # Get frame index from first image file
            frame_index = os.path.splitext(os.path.basename(img_files[0]))[0]
            mesh_frame_index = int(frame_index) - seq_start
            # print(f"frame_index: {frame_index}, mesh_frame_index: {mesh_frame_index}")
            video_seq_class_name_instance_id = f"{video_seq}/{seq_name}/{instance_id}"
            try:
                multiview_reference_images = self.mesh_loader.load_mesh_and_render(
                    video_seq_class_name_instance_id, mesh_frame_index, frame_selecting_strategy=self.frame_selecting_strategy, 
                    render_config=self.render_configs, verbose=False, 
                    height=self.target_resolution[1], width=self.target_resolution[0]
                )
            except Exception as e:
                print(f"Warning: Failed to load mesh for {video_seq_class_name_instance_id}: {e}")
                # Create empty lists if mesh loading 
                black_image = Image.new("RGB", self.target_resolution, (0, 0, 0))
                multiview_reference_images = [black_image] * len(self.render_configs)
                trajectory_maps = self._generate_trajectory_maps(fine_mask_arrays, min(self.target_resolution) // 100)
                return {
                    'class': seq_name,
                    'class_index': self.CLASSES_INDEX[seq_name],
                    'video_seq': video_seq,
                    'seq_name': seq_name,
                    'instance_id': instance_id,
                    'video': video_frames,  # List of PIL.Image
                    # 'depth_video': depth_video_frames,  # List of PIL.Image
                    'prompt': prompt,  # String
                    'vace_video': vace_video_frames,  # List of PIL.Image
                    'vace_video_mask': vace_video_mask_images,  # List of PIL.Image (RGB mode)
                    'vace_reference_image': vace_reference_image,  # List of PIL.Image (single element)
                    'multiview_reference_image': multiview_reference_images,  # List of PIL.Image (len = num of view)
                    'normal_maps': normal_maps,  # List of PIL.Image (len = num of view)
                    'position_maps': position_maps,  # List of PIL.Image (len = num of view)
                    'trajectory_maps': trajectory_maps,  # List of PIL.Image (RGB, same length as video frames)
                    }
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
            # frame_index_processed = self.mesh_loader.process_frame_index(video_seq_class_name_instance_id, mesh_frame_index, self.frame_selecting_strategy)
            # # Load mesh and camera parameters
            # mesh = self.mesh_loader.load_mesh(video_seq_class_name_instance_id, frame_index_processed)
            # params = self.mesh_loader.load_camera_parameters(video_seq_class_name_instance_id, frame_index_processed)
        
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
            # # except Exception as e:
            # #     print(f"Warning: Failed to load mesh for {video_seq}: {e}")
            # #     # Create empty lists if mesh loading fails
            # #     multiview_reference_images = []
            # #     normal_maps = []
            # #     position_maps = []

        # Timing: Trajectory maps generation
        if DEBUG_TIME:
            traj_start = time.time()
        # Generate trajectory_maps based on trajectory_type
        # Use fine_mask_arrays to ensure fine mask is used even if mask_arrays were modified
        trajectory_maps = self._generate_trajectory_maps(fine_mask_arrays, min(self.target_resolution) // 100)
        if DEBUG_TIME:
            traj_end = time.time()
            self._timing_stats['trajectory_maps'].append(traj_end - traj_start)
    
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
            video_frames = video_frames[::3] # downsample by 3
            # depth_video_frames = depth_video_frames[::3] # downsample by 3
            vace_video_frames = vace_video_frames[::3] # downsample by 3
            vace_video_mask_images = vace_video_mask_images[::3] # downsample by 3
            trajectory_maps = trajectory_maps[::3] # downsample by 3
            
            pading_length = self.length - len(video_frames)
            # pad_with black frames
            if pading_length > 0:
                video_frames = video_frames + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                # depth_video_frames = depth_video_frames + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                vace_video_frames = vace_video_frames + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                vace_video_mask_images = vace_video_mask_images + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
                trajectory_maps = trajectory_maps + [Image.new("RGB", (self.target_resolution[0], self.target_resolution[1]), (0, 0, 0))] * pading_length
        
        return {
            'class': seq_name,
            'class_index': self.CLASSES_INDEX[seq_name],
            'video_seq': video_seq,
            'instance_id': instance_id,
            'video': video_frames,  # List of PIL.Image
            'depth_video': depth_video_frames,  # List of PIL.Image
            'prompt': prompt,  # String
            'vace_video': vace_video_frames,  # List of PIL.Image
            'vace_video_mask': vace_video_mask_images,  # List of PIL.Image (RGB mode)
            'vace_reference_image': vace_reference_image,  # List of PIL.Image (single element)
            'multiview_reference_image': multiview_reference_images,  # List of PIL.Image (len = num of view)
            'normal_maps': normal_maps,  # List of PIL.Image (len = num of view)
            'position_maps': position_maps,  # List of PIL.Image (len = num of view)
            'trajectory_maps': trajectory_maps,  # List of PIL.Image (RGB, same length as video frames)
        }

if __name__ == "__main__":
    # Test the dataset
    dataset_val = VIPSeg_origin_mix_dataloader(
        vipseg_dataset_path=os.getenv("VIPSEG_DATASET_PATH"),
        vspw_dataset_path=os.getenv("VSPW_DATASET_PATH"),
        split_name="val",
        length=81,
        target_resolution=(480, 832),
        min_instance_ratio=0.10,
        min_bbox_ratio=0.10,
        frame_selecting_strategy="nearest",
        use_masked_vace_video=True,
        dataset_repeat=1,
        max_samples=None,
        mask_foreground=True,
        mask_strategy="bbox_with_traj",
        bbox_scale=1.0,
        trajectory_type="mask",
        sparse_box_interval=10,
        enable_data_augmentation=False,
    )
    
    print(f"Dataset length: {len(dataset_val)}")

    # dataset_train = VIPSeg_origin_mix_dataloader(
    #     vipseg_dataset_path=os.getenv("VIPSEG_DATASET_PATH"),
    #     vspw_dataset_path=os.getenv("VSPW_DATASET_PATH"),
    #     split_name="train",
    #     length=81,
    #     target_resolution=(480, 270),
    #     min_instance_ratio=0.10,
    #     min_bbox_ratio=0.10,
    #     frame_selecting_strategy="farthest",
    #     use_masked_vace_video=True,
    #     dataset_repeat=1,
    #     max_samples=None,
    #     mask_foreground=True,
    #     mask_strategy="fine",
    #     bbox_scale=1.0,
    #     trajectory_type="mask",
    #     sparse_box_interval=10,
    #     enable_data_augmentation=True,
    # )

    dataset = dataset_val
    
    # Test getitem
    
    import shutil
    # remove previous debug_val_VIPSeg directory
    if os.path.exists(f"./debug_val_VIPSeg"):
        shutil.rmtree(f"./debug_val_VIPSeg")
    os.makedirs(f"./debug_val_VIPSeg")
    random_indices = random.sample(range(len(dataset)), 10)
    # random_indices = [4100]
    with open(f"./debug_val_VIPSeg/random_indices.txt", "w", encoding="utf-8") as f:
        f.write(str(random_indices))
    for index in random_indices:
        sample = dataset[index]
        print(f"Sample keys: {sample.keys()}")
        print(f"Video seq: {sample['video_seq']}")
        print(f"Instance id: {sample['instance_id']}")
        print(f"Video frames: {len(sample['video'])}")
        print(f"Video frame size: {sample['video'][0].size}")
        print(f"Depth video frames: {len(sample['depth_video'])}")
        print(f"Depth video frame size: {sample['depth_video'][0].size}")
        print(f"Prompt: {sample['prompt']}")
        print(f"VACE video frames: {len(sample['vace_video'])}")
        print(f"VACE reference image: {len(sample['vace_reference_image'])}")
        print(f"VACE reference image size: {sample['vace_reference_image'][0].size}")
        print(f"multiview reference: {len(sample['multiview_reference_image'])}")
        print(f"multiview reference size: {sample['multiview_reference_image'][0].size}")
        print(f"trajectory maps: {len(sample['trajectory_maps'])}")
        print(f"trajectory map size: {sample['trajectory_maps'][0].size}")
        print(f"trajectory map mode: {sample['trajectory_maps'][0].mode}")
        print(f"mask: {len(sample['vace_video_mask'])}")
        print(f"mask size: {sample['vace_video_mask'][0].size}")
        print(f"mask mode: {sample['vace_video_mask'][0].mode}")
        # save first 10 frames of each sequence for debug
        output_dir = f"./debug_val_VIPSeg/instance_{index}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/prompt.txt", "w", encoding="utf-8") as f:
            f.write(sample['prompt'])
        # Save first 10 video frames
        for i, img in enumerate(sample['video'][:10]):
            img.save(f"{output_dir}/video_{i}.png")
        # Save first 10 depth video frames
        for i, img in enumerate(sample['depth_video'][:10]):
            img.save(f"{output_dir}/depth_video_{i}.png")
        # Save first 10 vace video frames
        for i, img in enumerate(sample['vace_video'][:10]):
            img.save(f"{output_dir}/vace_video_{i}.png")
        # Save reference image
        sample['vace_reference_image'][0].save(f"{output_dir}/reference_image.png")
        # Save all multiview references (usually small list)
        for i, img in enumerate(sample['multiview_reference_image']):
            img.save(f"{output_dir}/multiview_reference_image_{i}.png")
        # Save first 10 vace video masks
        for i, img in enumerate(sample['vace_video_mask'][:10]):
            img.save(f"{output_dir}/mask_{i}.png")
        # Save first 10 normal maps
        for i, img in enumerate(sample['normal_maps'][:10]):
            img.save(f"{output_dir}/normal_map_{i}.png")
        # Save first 10 position maps
        for i, img in enumerate(sample['position_maps'][:10]):
            img.save(f"{output_dir}/position_map_{i}.png")
        # Save first 10 trajectory maps
        if 'trajectory_maps' in sample:
            for i, img in enumerate(sample['trajectory_maps'][:10]):
                img.save(f"{output_dir}/trajectory_map_{i}.png")
