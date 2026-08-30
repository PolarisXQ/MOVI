import imageio, os, torch, warnings, torchvision, argparse, json, time
from typing import List, Dict, Any
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard is not available. Install with: pip install tensorboard")
DEBUG_TIME = False


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class MultiDatasetLoader(torch.utils.data.Dataset):
    """
    Unified dataloader that combines multiple datasets for training.
    
    Supports:
    - DAVIS17_VACE_Dataset
    - VIPSeg_origin_mix_dataloader
    - Any other torch.utils.data.Dataset
    
    Usage:
        dataset_configs = [
            {
                'type': 'davis17',
                'weight': 1.0,  # Optional: sampling weight
                'dataset_path': os.getenv("DAVIS17_DATASET_PATH"),
                'split_name': 'train',
                'length': 81,
                'target_resolution': (480, 832),
                # ... other DAVIS17 parameters
            },
            {
                'type': 'vipseg',
                'weight': 0.5,  # Sample from VIPSeg half as often
                'vipseg_dataset_path': os.getenv("VIPSEG_DATASET_PATH"),
                'vspw_dataset_path': os.getenv("VSPW_DATASET_PATH"),
                'split_name': 'all',
                'length': 81,
                'target_resolution': (480, 270),
                # ... other VIPSeg parameters
            },
        ]
        dataset = MultiDatasetLoader(dataset_configs)
    """
    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        sampling_strategy: str = "proportional",  # "proportional", "uniform", "weighted"
        **kwargs,
    ):
        """
        Args:
            dataset_configs: List of dataset configuration dictionaries.
                Each dict must have a 'type' key specifying the dataset type.
                Supported types: 'davis17', 'vipseg', 'custom'
            sampling_strategy: How to sample from datasets:
                - "proportional": Sample proportionally to dataset sizes
                - "uniform": Sample uniformly from each dataset
                - "weighted": Use 'weight' field in configs for sampling
        """
        self.dataset_configs = dataset_configs
        self.sampling_strategy = sampling_strategy
        self.datasets = []
        self.dataset_weights = []
        self.dataset_cumulative_sizes = []
        
        # Import dataset classes
        try:
            from diffsynth.trainers.davis17_vace_dataset import DAVIS17_VACE_Dataset
            self.DAVIS17_VACE_Dataset = DAVIS17_VACE_Dataset
        except ImportError:
            self.DAVIS17_VACE_Dataset = None
            print("Warning: DAVIS17_VACE_Dataset not available")
        
        try:
            from diffsynth.trainers.VIPSeg_origin_mix_loader30hz import VIPSeg_origin_mix_dataloader
            self.VIPSeg_origin_mix_dataloader = VIPSeg_origin_mix_dataloader
        except ImportError:
            self.VIPSeg_origin_mix_dataloader = None
            print("Warning: VIPSeg_origin_mix_dataloader not available")
        
        try:
            from diffsynth.trainers.davis17_vace_dataset_multiview import DAVIS17_VACE_Dataset_Multiview
            self.DAVIS17_VACE_Dataset_Multiview = DAVIS17_VACE_Dataset_Multiview
        except ImportError:
            self.DAVIS17_VACE_Dataset_Multiview = None
            print("Warning: DAVIS17_VACE_Dataset_Multiview not available")

        try:
            from diffsynth.trainers.rose_vace_dataset import ROSE_VACE_Dataset
            self.ROSE_VACE_Dataset = ROSE_VACE_Dataset
        except ImportError:
            self.ROSE_VACE_Dataset = None
            print("Warning: ROSE_VACE_Dataset not available")

        try:
            from diffsynth.trainers.humanm3_vace_dataset import HumanM3_VACE_Dataset
            self.HumanM3_VACE_Dataset = HumanM3_VACE_Dataset
        except ImportError:
            self.HumanM3_VACE_Dataset = None
            print("Warning: HumanM3_VACE_Dataset not available")

        try:
            from diffsynth.trainers.penn_action_vace_dataset import PennAction_VACE_Dataset
            self.PennAction_VACE_Dataset = PennAction_VACE_Dataset
        except ImportError:
            self.PennAction_VACE_Dataset = None
            print("Warning: PennAction_VACE_Dataset not available")

        try:
            from diffsynth.trainers.dpw3d_vace_dataset import DPW3D_VACE_Dataset
            self.DPW3D_VACE_Dataset = DPW3D_VACE_Dataset
        except ImportError:
            self.DPW3D_VACE_Dataset = None
            print("Warning: DPW3D_VACE_Dataset not available")
        
        # Initialize datasets
        self._initialize_datasets()
    
    def _initialize_datasets(self):
        """Initialize all datasets from configurations."""
        total_size = 0
        self.dataset_types = []  # Store types for get_dataset_info
        
        for i, config in enumerate(self.dataset_configs):
            # Create a copy to avoid modifying the original config
            config = config.copy()
            dataset_type = config.pop('type', None)
            if dataset_type is None:
                raise ValueError(f"Dataset config {i} must have a 'type' field")
            
            self.dataset_types.append(dataset_type)  # Store type for later use
            
            weight = config.pop('weight', 1.0)
            
            # Create dataset based on type
            if dataset_type == 'davis17':
                if self.DAVIS17_VACE_Dataset is None:
                    raise ImportError("DAVIS17_VACE_Dataset is not available")
                dataset = self.DAVIS17_VACE_Dataset(**config)
            elif dataset_type == 'vipseg':
                if self.VIPSeg_origin_mix_dataloader is None:
                    raise ImportError("VIPSeg_origin_mix_dataloader is not available")
                dataset = self.VIPSeg_origin_mix_dataloader(**config)
            elif dataset_type == 'davis17_multiview':
                if self.DAVIS17_VACE_Dataset_Multiview is None:
                    raise ImportError("DAVIS17_VACE_Dataset_Multiview is not available")
                dataset = self.DAVIS17_VACE_Dataset_Multiview(**config)
            elif dataset_type == 'rose':
                if self.ROSE_VACE_Dataset is None:
                    raise ImportError("ROSE_VACE_Dataset is not available")
                dataset = self.ROSE_VACE_Dataset(**config)
            elif dataset_type == 'humanm3':
                if self.HumanM3_VACE_Dataset is None:
                    raise ImportError("HumanM3_VACE_Dataset is not available")
                dataset = self.HumanM3_VACE_Dataset(**config)
            elif dataset_type == 'penn_action':
                if self.PennAction_VACE_Dataset is None:
                    raise ImportError("PennAction_VACE_Dataset is not available")
                dataset = self.PennAction_VACE_Dataset(**config)
            elif dataset_type == 'dpw3d':
                if self.DPW3D_VACE_Dataset is None:
                    raise ImportError("DPW3D_VACE_Dataset is not available")
                dataset = self.DPW3D_VACE_Dataset(**config)
            elif dataset_type == 'custom':
                # For custom datasets, expect 'dataset' key with an already-instantiated dataset
                if 'dataset' in config:
                    dataset = config['dataset']
                else:
                    raise ValueError("Custom dataset type requires 'dataset' key with instantiated dataset")
            else:
                raise ValueError(
                    f"Unknown dataset type: {dataset_type}. "
                    "Supported: 'davis17', 'vipseg', 'davis17_multiview', 'rose', "
                    "'humanm3', 'penn_action', 'dpw3d', 'custom'"
                )
            
            dataset_size = len(dataset)
            self.datasets.append(dataset)
            self.dataset_weights.append(weight)
            total_size += dataset_size
            self.dataset_cumulative_sizes.append(total_size)
            
            print(f"Loaded {dataset_type} dataset: {dataset_size} samples (weight: {weight})")
        
        print(f"Total combined dataset size: {total_size}")
        
        # Calculate sampling probabilities based on strategy
        if self.sampling_strategy == "proportional":
            # Sample proportionally to dataset sizes
            self.sampling_probs = [len(ds) / total_size for ds in self.datasets]
        elif self.sampling_strategy == "uniform":
            # Sample uniformly from each dataset
            self.sampling_probs = [1.0 / len(self.datasets)] * len(self.datasets)
        elif self.sampling_strategy == "weighted":
            # Use provided weights, normalized
            total_weight = sum(self.dataset_weights)
            self.sampling_probs = [w / total_weight for w in self.dataset_weights]
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")
        
        print(f"Sampling strategy: {self.sampling_strategy}")
        print(f"Sampling probabilities: {self.sampling_probs}")
    
    def __len__(self):
        """Return total number of samples across all datasets."""
        return sum(len(ds) for ds in self.datasets)
    
    def __getitem__(self, index):
        """
        Get item from combined datasets.
        
        For proportional sampling: index maps directly to samples across all datasets.
        For uniform/weighted sampling: we use random sampling based on probabilities.
        """
        import random
        
        if self.sampling_strategy == "proportional":
            # Map index to dataset based on cumulative sizes
            dataset_idx = 0
            for i, cum_size in enumerate(self.dataset_cumulative_sizes):
                if index < cum_size:
                    dataset_idx = i
                    break
            # Calculate index within the selected dataset
            prev_cum_size = self.dataset_cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0
            dataset_index = index - prev_cum_size
        else:
            # For uniform/weighted: sample dataset based on probabilities
            # Use index as seed for deterministic behavior (can be made random if needed)
            r = (index * 1103515245 + 12345) % (2**31) / (2**31)  # Pseudo-random based on index
            cum_prob = 0
            dataset_idx = 0
            for i, prob in enumerate(self.sampling_probs):
                cum_prob += prob
                if r < cum_prob:
                    dataset_idx = i
                    break
            
            # Sample uniformly within the selected dataset
            dataset_index = (index // len(self.datasets)) % len(self.datasets[dataset_idx])
        
        # Get sample from selected dataset
        sample = self.datasets[dataset_idx][dataset_index]
        
        # Add dataset source information for debugging
        if isinstance(sample, dict):
            sample['dataset_source'] = self.dataset_types[dataset_idx]
            sample['dataset_index'] = dataset_idx
        return sample
    
    def get_dataset_info(self):
        """Return information about all datasets."""
        info = {
            'num_datasets': len(self.datasets),
            'total_size': len(self),
            'sampling_strategy': self.sampling_strategy,
            'datasets': []
        }
        
        for i, (dataset_type, dataset, weight, prob) in enumerate(zip(
            self.dataset_types, self.datasets, self.dataset_weights, self.sampling_probs
        )):
            info['datasets'].append({
                'index': i,
                'type': dataset_type,
                'size': len(dataset),
                'weight': weight,
                'sampling_probability': prob,
            })
        
        return info


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            # Support multiple prefixes separated by commas
            prefixes = [p.strip() for p in remove_prefix.split(",") if p.strip()]
            for name, param in state_dict.items():
                # Try each prefix
                removed = False
                for prefix in prefixes:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        removed = True
                        break
                # If no prefix matched, keep original name
                if not removed:
                    # Check if it's a trainable parameter that should be included
                    pass
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        import re
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            # Group split model files together (e.g., diffusion_pytorch_model-00001-of-00007.safetensors)
            # Pattern to match split model files: filename-00001-of-00007.safetensors
            split_file_pattern = re.compile(r'(.+)-(\d{5})-of-(\d{5})\.(safetensors|bin|pth|pt)$')
            
            # Separate split files from regular files
            split_file_groups = {}
            regular_files = []
            
            for path in model_paths:
                match = split_file_pattern.search(path)
                if match:
                    # This is a split file
                    base_name = match.group(1)  # e.g., "diffusion_pytorch_model"
                    file_num = int(match.group(2))  # e.g., 1
                    total_num = int(match.group(3))  # e.g., 7
                    ext = match.group(4)  # e.g., "safetensors"
                    
                    # Create a key to group files from the same split model
                    group_key = (base_name, total_num, ext)
                    if group_key not in split_file_groups:
                        split_file_groups[group_key] = {}
                    split_file_groups[group_key][file_num] = path
                else:
                    # Regular file (not split)
                    regular_files.append(path)
            
            # Create ModelConfig for each split model group
            for (base_name, total_num, ext), file_dict in split_file_groups.items():
                # Sort by file number and create a list
                split_paths = [file_dict[i] for i in sorted(file_dict.keys())]
                # Verify we have all files
                if len(split_paths) == total_num:
                    model_configs.append(ModelConfig(path=split_paths, offload_dtype=offload_dtype))
                else:
                    # If not all files are present, treat as regular files
                    print(f"Warning: Split model {base_name} has {len(split_paths)}/{total_num} files. Treating as separate files.")
                    regular_files.extend(split_paths)
            
            # Create ModelConfig for regular files (one per file)
            for path in regular_files:
                model_configs.append(ModelConfig(path=path, offload_dtype=offload_dtype))
        
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
        train_controlnet=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        if train_controlnet:
            # ControlNet training mode: freeze all models except ControlNet
            print("[INFO] ControlNet training mode: freezing all models except ControlNet", flush=True)
            
            # Load and freeze LoRA if checkpoint is provided
            if lora_checkpoint is not None and lora_base_model is not None:
                print(f"[INFO] Loading LoRA from checkpoint: {lora_checkpoint}", flush=True)
                model = self.add_lora_to_model(
                    getattr(pipe, lora_base_model),
                    target_modules=lora_target_modules.split(","),
                    lora_rank=lora_rank,
                    upcast_dtype=pipe.torch_dtype,
                )
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys", flush=True)
                if len(load_result[1]) > 0:
                    print(f"[WARNING] LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}", flush=True)
                # Freeze LoRA
                for param in model.parameters():
                    param.requires_grad = False
                setattr(pipe, lora_base_model, model)
                print("[INFO] LoRA frozen", flush=True)
            
            # Freeze all models except ControlNet
            # Determine which models to freeze (all except trajectory_controlnet)
            all_models = ["dit", "dit2", "vace", "vace2", "text_encoder", "text_encoder2", 
                         "motion_controller", "animate_adapter", "vae", "vae2"]
            trainable_models_list = ["trajectory_controlnet", "trajectory_controlnet2"]
            pipe.freeze_except(trainable_models_list)
            print(f"[INFO] Only training: {trainable_models_list}", flush=True)
            
            # Ensure ControlNet exists, if not, it should be initialized elsewhere
            if not hasattr(pipe, "trajectory_controlnet") or pipe.trajectory_controlnet is None:
                print("[WARNING] trajectory_controlnet is None. Please ensure ControlNet is initialized.", flush=True)
        else:
            # Normal training mode
            # Freeze untrainable models
            pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
            
            # Enable FP8 if pipeline supports
            if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
                pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
            
            # Add LoRA to the base models
            if lora_base_model is not None:
                model = self.add_lora_to_model(
                    getattr(pipe, lora_base_model),
                    target_modules=lora_target_modules.split(","),
                    lora_rank=lora_rank,
                    upcast_dtype=pipe.torch_dtype,
                )
                if lora_checkpoint is not None:
                    state_dict = load_state_dict(lora_checkpoint)
                    state_dict = self.mapping_lora_state_dict(state_dict)
                    load_result = model.load_state_dict(state_dict, strict=False)
                    print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                    if len(load_result[1]) > 0:
                        print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
                setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def save_training_state(self, accelerator, optimizer, scheduler, global_step, epoch_id, save_steps=None):
        """Save training state including optimizer, scheduler, global_step, and epoch."""
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Use num_steps as global_step when saving alongside step-N.safetensors so resume step count is correct
            step_for_state = self.num_steps if (save_steps is not None and self.num_steps % save_steps == 0) else global_step
            training_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": step_for_state,
                "epoch_id": epoch_id,
                "num_steps": self.num_steps,
            }
            os.makedirs(self.output_path, exist_ok=True)
            # Save training state alongside model checkpoint
            if save_steps is not None and self.num_steps % save_steps == 0:
                state_path = os.path.join(self.output_path, f"step-{self.num_steps}_training_state.pt")
            else:
                state_path = os.path.join(self.output_path, "latest_training_state.pt")
            torch.save(training_state, state_path)
            print(f"[INFO] Training state saved to: {state_path}", flush=True)


    def load_training_state(self, checkpoint_path, accelerator, optimizer, scheduler):
        """Load training state from checkpoint directory."""
        # Try to find training state file
        checkpoint_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path
        checkpoint_name = os.path.basename(checkpoint_path) if os.path.isfile(checkpoint_path) else None
        
        # Try to load from step-specific state file first
        print(f"[INFO] Loading training state from: {checkpoint_path}", flush=True)
        print(f"[INFO] Checkpoint directory: {checkpoint_dir}", flush=True)
        print(f"[INFO] Checkpoint name: {checkpoint_name}", flush=True)
        if checkpoint_name and checkpoint_name.startswith("step-"):
            step_num = checkpoint_name.replace("step-", "").replace(".safetensors", "")
            print(f"[INFO] Step number: {step_num}", flush=True)
            state_path = os.path.join(checkpoint_dir, f"step-{step_num}_training_state.pt")
            print(f"[INFO] State path: {state_path}", flush=True)
            if os.path.exists(state_path):
                training_state = torch.load(state_path, map_location="cpu")
                print(f"[INFO] Loaded training state from: {state_path}", flush=True)
                return training_state
        
        # Try to load from latest state file
        latest_state_path = os.path.join(checkpoint_dir, "latest_training_state.pt")
        if os.path.exists(latest_state_path):
            training_state = torch.load(latest_state_path, map_location="cpu")
            print(f"[INFO] Loaded training state from: {latest_state_path}", flush=True)
            return training_state
        
        print(f"[WARNING] No training state file found in {checkpoint_dir}. Starting from scratch.", flush=True)
        return None


    def on_step_end(self, accelerator, model, save_steps=None, optimizer=None, scheduler=None, global_step=None, epoch_id=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            # Save training state alongside model checkpoint
            if optimizer is not None and scheduler is not None and global_step is not None and epoch_id is not None:
                self.save_training_state(accelerator, optimizer, scheduler, global_step, epoch_id, save_steps)


    def on_epoch_end(self, accelerator, model, epoch_id, optimizer=None, scheduler=None, global_step=None, save_steps=None, validation_dataset=None, validation_sample_idx=-1, writer=None):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            # save training state alongside model checkpoint
            self.save_training_state(accelerator, optimizer, scheduler, global_step, epoch_id, save_steps)
            
            # Generate validation video if validation dataset is provided
            if validation_dataset is not None:
                self.generate_validation_video(accelerator, model, epoch_id, validation_dataset, validation_sample_idx, writer=writer, global_step=global_step)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
    
    def generate_validation_video(self, accelerator, model, epoch_id, validation_dataset, sample_idx=-1, writer=None, global_step=None):
        """Generate a validation video at the end of an epoch."""
        try:
            from diffsynth import save_video
            import numpy as np
            import random
            
            print(f"[INFO] Generating validation video for epoch {epoch_id}...", flush=True)
            
            # Get the unwrapped model to access the pipeline
            unwrapped_model = accelerator.unwrap_model(model)
            if not hasattr(unwrapped_model, 'pipe'):
                print(f"[WARNING] Model does not have 'pipe' attribute. Skipping validation video generation.", flush=True)
                return
            
            pipe = unwrapped_model.pipe
            
            # Store original training state
            was_training = pipe.training
            
            # Set model to eval mode (this disables dropout, batch norm updates, etc.)
            pipe.eval()
            
            # Get a sample from validation dataset
            if sample_idx >= len(validation_dataset):
                sample_idx = 0
            if sample_idx < 0:
                # random sample
                sample_idx = random.randint(0, len(validation_dataset) - 1)
            sample = validation_dataset[sample_idx]
            
            # Prepare inputs for inference
            with torch.no_grad():
                # Prepare inputs similar to training but for inference
                prompt = sample.get('prompt', '')
                negative_prompt = sample.get('negative_prompt', '')
                
                # Get VACE inputs
                vace_video = sample.get('vace_video', None)
                vace_video_mask = sample.get('vace_video_mask', None)
                
                # Get reference images
                vace_reference_image = None
                if 'vace_reference_image' in sample and sample['vace_reference_image']:
                    vace_reference_image = [sample['vace_reference_image'][0]] if isinstance(sample['vace_reference_image'], list) else [sample['vace_reference_image']]
                
                multiview_reference_image = sample.get('multiview_reference_image', None)
                if multiview_reference_image is not None and len(multiview_reference_image) == 0:
                    multiview_reference_image = None
                normal_maps = sample.get('normal_maps', None)
                position_maps = sample.get('position_maps', None)
                trajectory_maps = sample.get('trajectory_maps', None)
                inpaint_video = sample.get('inpaint_video', None)
                
                # Determine number of frames
                num_frames = len(vace_video) if vace_video else 81
                
                # Get resolution from first frame if available
                if vace_video and len(vace_video) > 0:
                    width, height = vace_video[0].size
                else:
                    height, width = 480, 832  # Default resolution
                
                # Generate video using the pipeline
                try:
                    pipe_kwargs = dict(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt else "",
                        vace_video=vace_video,
                        vace_video_mask=vace_video_mask,
                        vace_reference_image=vace_reference_image,
                        multiview_reference_image=multiview_reference_image,
                        normal_maps=normal_maps,
                        position_maps=position_maps,
                        trajectory_maps=trajectory_maps,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        seed=42,  # Fixed seed for reproducibility
                        tiled=True,
                        cfg_scale=1.0,  # Use cfg_scale=1.0 for faster inference during training
                    )
                    if inpaint_video is not None:
                        pipe_kwargs["input_video"] = inpaint_video
                        pipe_kwargs["denoising_strength"] = 0.85
                    generated_video = pipe(**pipe_kwargs)
                    
                    # Convert video to tensor format for TensorBoard (B, T, C, H, W)
                    # generated_video is a list of PIL Images
                    video_frames = []
                    for frame in generated_video:
                        # Convert PIL Image to numpy array and then to tensor
                        frame_array = np.array(frame)  # Shape: (H, W, C)
                        video_frames.append(frame_array)
                    
                    # Stack frames: (T, H, W, C) -> (T, C, H, W)
                    video_tensor = torch.from_numpy(np.stack(video_frames)).permute(0, 3, 1, 2)  # (T, C, H, W)
                    # Add batch dimension: (1, T, C, H, W)
                    video_tensor = video_tensor.unsqueeze(0).float()  # TensorBoard expects float
                    # Normalize to [0, 1] range (frames are already in [0, 255])
                    video_tensor = video_tensor / 255.0
                    # Move to CPU (TensorBoard requires CPU tensors)
                    video_tensor = video_tensor.cpu()
                    
                    # Save to TensorBoard log directory
                    if writer is not None:
                        # Log video to TensorBoard
                        writer.add_video(
                            "validation/generated_video",
                            video_tensor,
                            global_step=global_step if global_step is not None else epoch_id,
                            fps=15
                        )
                        writer.flush()  # Ensure data is written immediately
                        print(f"[INFO] Validation video logged to TensorBoard at step {global_step if global_step is not None else epoch_id}", flush=True)
                    
                    # Also save video file to TensorBoard log directory
                    tensorboard_log_dir = os.path.join(self.output_path, "tensorboard_logs")
                    if os.path.exists(tensorboard_log_dir):
                        validation_dir = os.path.join(tensorboard_log_dir, "validation_videos")
                        os.makedirs(validation_dir, exist_ok=True)
                        video_path = os.path.join(validation_dir, f"epoch-{epoch_id}_sample-{sample_idx}.mp4")
                        save_video(generated_video, video_path, fps=15, quality=5)
                        print(f"[INFO] Validation video saved to: {video_path}", flush=True)
                    else:
                        # Fallback to output directory if tensorboard_logs doesn't exist
                        validation_dir = os.path.join(self.output_path, "validation_videos")
                        os.makedirs(validation_dir, exist_ok=True)
                        video_path = os.path.join(validation_dir, f"epoch-{epoch_id}_sample-{sample_idx}.mp4")
                        save_video(generated_video, video_path, fps=15, quality=5)
                        print(f"[INFO] Validation video saved to: {video_path}", flush=True)
                    
                except Exception as e:
                    print(f"[WARNING] Failed to generate validation video: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
            
            # Restore original training state
            if was_training:
                pipe.train()
            
        except Exception as e:
            print(f"[WARNING] Error during validation video generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Try to restore training mode even if there was an error
            try:
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, 'pipe'):
                    unwrapped_model.pipe.train()
            except:
                pass

    def load_model_checkpoint(self, checkpoint_path, accelerator, model):
        """Load model checkpoint. The checkpoint should contain only trainable parameters (e.g., LoRA weights)."""
        if os.path.isfile(checkpoint_path):
            print(f"[INFO] Loading model checkpoint from: {checkpoint_path}", flush=True)
            checkpoint_state_dict = load_state_dict(checkpoint_path)
            # Map LoRA state dict if needed
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'mapping_lora_state_dict'):
                state_dict = unwrapped_model.mapping_lora_state_dict(checkpoint_state_dict.copy())
            else:
                state_dict = checkpoint_state_dict.copy()
            # Load into the unwrapped model (similar to how lora_checkpoint is loaded)
            # We need to add the prefix back if it was removed during saving
            if self.remove_prefix_in_ckpt: # TODO: simplify this
                prefixed_state_dict = {}
                # Support multiple prefixes separated by commas
                prefixes = [p.strip() for p in self.remove_prefix_in_ckpt.split(",") if p.strip()]
                
                for key, value in state_dict.items():
                    # Try to determine which prefix to use based on key name
                    # For multiview_feature_bank_adapter: "fb_modules.*" or "image_proj.*"
                    # For multiview_ipadapter keys: "image_proj" or "ipadapter_modules"
                    # For vace keys: "vace_*" or "vace_blocks", etc.
                    prefixed_key = None
                    
                    # Check if this is a multiview_feature_bank_adapter key (fb_modules is unique to feature_bank)
                    is_feature_bank_key = (
                        key.startswith("fb_modules.") or
                        (key.startswith("image_proj.") and any("feature_bank" in p for p in prefixes))
                    )
                    
                    # Check if this is a multiview_ipadapter key (and not already claimed by feature_bank)
                    is_ipadapter_key = (
                        not is_feature_bank_key and
                        (key.startswith("image_proj") or
                         key.startswith("ipadapter_modules") or
                         ("ipadapter" in key.lower() and "image_proj" not in key and "ipadapter_modules" not in key))
                    )
                    
                    # Check if this is a vace key (not ipadapter / not feature_bank)
                    is_vace_key = (
                        (key.startswith("vace_") or "vace_blocks" in key or "vace_patch_embedding" in key) and
                        not is_ipadapter_key and not is_feature_bank_key
                    )
                    
                    # Assign prefix based on key type (feature_bank first, then ipadapter, then vace)
                    for prefix in prefixes:
                        if is_feature_bank_key and "feature_bank" in prefix:
                            prefixed_key = prefix + key
                            break
                        if is_ipadapter_key and ("multiview_ipadapter" in prefix or "ipadapter" in prefix.lower()):
                            prefixed_key = prefix + key
                            break
                        if is_vace_key and "vace" in prefix.lower() and "ipadapter" not in prefix.lower() and "feature_bank" not in prefix:
                            prefixed_key = prefix + key
                            break
                    
                    # If no prefix matched, try the first prefix as fallback (but prefer vace for non-ipadapter keys)
                    if prefixed_key is None and len(prefixes) > 0:
                        # For non-ipadapter keys, prefer vace prefix
                        if not is_ipadapter_key and not is_feature_bank_key:
                            for prefix in prefixes:
                                if "vace" in prefix.lower() and "ipadapter" not in prefix.lower() and "feature_bank" not in prefix:
                                    prefixed_key = prefix + key
                                    break
                        # If still no match, use first prefix
                        if prefixed_key is None:
                            prefixed_key = prefixes[0] + key
                    
                    if prefixed_key is not None:
                        prefixed_state_dict[prefixed_key] = value
                    else:
                        # If no prefix could be determined, keep original key
                        prefixed_state_dict[key] = value
                
                state_dict = prefixed_state_dict
            load_result = unwrapped_model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Model checkpoint loaded. Missing keys: {len(load_result[0])}, Unexpected keys: {len(load_result[1])}", flush=True)
            if len(load_result[0]) > 0:
                print(f"[WARNING] Missing keys in checkpoint (showing first 5): {load_result[0][:5]}", flush=True)
            if len(load_result[1]) > 0:
                print(f"[WARNING] Unexpected keys in checkpoint (showing first 5): {load_result[1][:5]}", flush=True)
            
            # Explicitly load depth_head and perception_head state dicts
            if hasattr(unwrapped_model, 'pipe') and unwrapped_model.pipe is not None:
                try:
                    from diffsynth.models.wan_video_segmentation import SemanticFPNHead
                    pipe = unwrapped_model.pipe
                    
                    # Load perception_head and depth_head for dit model
                    if hasattr(pipe, "dit") and pipe.dit is not None:
                        dit_model = pipe.dit
                        
                        # Initialize and load perception_head for dit
                        if not hasattr(dit_model, "perception_head") or dit_model.perception_head is None:
                            num_layers = len(dit_model.blocks) if hasattr(dit_model, "blocks") else 30
                            dit_model.start_layer = 0
                            dit_model.end_layer = num_layers - 1
                            dit_model.perception_head = SemanticFPNHead(
                                in_channels=getattr(dit_model, "dim", 1536),
                                out_channels=getattr(dit_model, "in_dim", 16),
                                num_tensors=dit_model.end_layer - dit_model.start_layer + 1,
                                patch_size=getattr(dit_model, "patch_size", (1, 2, 2)),
                            )
                            dit_model.perception_head = dit_model.perception_head.to(dtype=pipe.torch_dtype).to(pipe.device)
                            print(f"[INFO] Perception head initialized for dit model with {num_layers} layers", flush=True)
                        
                        # Load perception_head state dict if available
                        perception_head_state_dict = {}
                        for key, value in checkpoint_state_dict.items():
                            # Handle different prefix patterns
                            if key.startswith("pipe.dit.perception_head."):
                                new_key = key.replace("pipe.dit.perception_head.", "")
                                perception_head_state_dict[new_key] = value
                            elif key.startswith("dit.perception_head."):
                                new_key = key.replace("dit.perception_head.", "")
                                perception_head_state_dict[new_key] = value
                            elif key.startswith("perception_head."):
                                new_key = key.replace("perception_head.", "")
                                perception_head_state_dict[new_key] = value
                            # Also check with remove_prefix applied
                            elif self.remove_prefix_in_ckpt and key.startswith(self.remove_prefix_in_ckpt + "perception_head."):
                                new_key = key.replace(self.remove_prefix_in_ckpt + "perception_head.", "")
                                perception_head_state_dict[new_key] = value
                        
                        if perception_head_state_dict:
                            try:
                                load_result = dit_model.perception_head.load_state_dict(perception_head_state_dict, strict=False)
                                print(f"[INFO] Loaded perception_head {len(perception_head_state_dict)-len(load_result[0])}/{len(perception_head_state_dict)} parameters from checkpoint", flush=True)
                            except Exception as e:
                                print(f"[WARNING] Failed to load perception_head state dict: {e}", flush=True)
                        
                        # Initialize and load depth_head for dit
                        if not hasattr(dit_model, "depth_head") or dit_model.depth_head is None:
                            num_layers = len(dit_model.blocks) if hasattr(dit_model, "blocks") else 30
                            if not hasattr(dit_model, "start_layer"):
                                dit_model.start_layer = 0
                            if not hasattr(dit_model, "end_layer"):
                                dit_model.end_layer = num_layers - 1
                            dit_model.depth_head = SemanticFPNHead(
                                in_channels=getattr(dit_model, "dim", 1536),
                                out_channels=getattr(dit_model, "in_dim", 16),
                                num_tensors=dit_model.end_layer - dit_model.start_layer + 1,
                                patch_size=getattr(dit_model, "patch_size", (1, 2, 2)),
                            )
                            dit_model.depth_head = dit_model.depth_head.to(dtype=pipe.torch_dtype).to(pipe.device)
                            print(f"[INFO] Depth head initialized for dit model with {num_layers} layers", flush=True)
                        
                        # Load depth_head state dict if available
                        depth_head_state_dict = {}
                        for key, value in checkpoint_state_dict.items():
                            # Handle different prefix patterns
                            if key.startswith("pipe.dit.depth_head."):
                                new_key = key.replace("pipe.dit.depth_head.", "")
                                depth_head_state_dict[new_key] = value
                            elif key.startswith("dit.depth_head."):
                                new_key = key.replace("dit.depth_head.", "")
                                depth_head_state_dict[new_key] = value
                            elif key.startswith("depth_head."):
                                new_key = key.replace("depth_head.", "")
                                depth_head_state_dict[new_key] = value
                            # Also check with remove_prefix applied
                            elif self.remove_prefix_in_ckpt and key.startswith(self.remove_prefix_in_ckpt + "depth_head."):
                                new_key = key.replace(self.remove_prefix_in_ckpt + "depth_head.", "")
                                depth_head_state_dict[new_key] = value
                        
                        if depth_head_state_dict:
                            try:
                                load_result = dit_model.depth_head.load_state_dict(depth_head_state_dict, strict=False)
                                print(f"[INFO] Loaded depth_head {len(depth_head_state_dict)-len(load_result[0])}/{len(depth_head_state_dict)} parameters from checkpoint", flush=True)
                            except Exception as e:
                                print(f"[WARNING] Failed to load depth_head state dict: {e}", flush=True)
                    
                    # Load perception_head and depth_head for dit2 model if exists
                    if hasattr(pipe, "dit2") and pipe.dit2 is not None:
                        dit2_model = pipe.dit2
                        
                        # Initialize and load perception_head for dit2
                        if not hasattr(dit2_model, "perception_head") or dit2_model.perception_head is None:
                            num_layers = len(dit2_model.blocks) if hasattr(dit2_model, "blocks") else 30
                            dit2_model.start_layer = 0
                            dit2_model.end_layer = num_layers - 1
                            dit2_model.perception_head = SemanticFPNHead(
                                in_channels=getattr(dit2_model, "dim", 1536),
                                out_channels=getattr(dit2_model, "in_dim", 16),
                                num_tensors=dit2_model.end_layer - dit2_model.start_layer + 1,
                                patch_size=getattr(dit2_model, "patch_size", (1, 2, 2)),
                            )
                            dit2_model.perception_head = dit2_model.perception_head.to(dtype=pipe.torch_dtype).to(pipe.device)
                            print(f"[INFO] Perception head initialized for dit2 model with {num_layers} layers", flush=True)
                        
                        # Load perception_head state dict for dit2 if available
                        perception_head_state_dict_dit2 = {}
                        for key, value in checkpoint_state_dict.items():
                            # Handle different prefix patterns
                            if key.startswith("vace.dit2.perception_head."):
                                new_key = key.replace("vace.dit2.perception_head.", "")
                                perception_head_state_dict_dit2[new_key] = value
                            elif key.startswith("pipe.dit2.perception_head."):
                                new_key = key.replace("pipe.dit2.perception_head.", "")
                                perception_head_state_dict_dit2[new_key] = value
                            elif key.startswith("dit2.perception_head."):
                                new_key = key.replace("dit2.perception_head.", "")
                                perception_head_state_dict_dit2[new_key] = value
                            # Also check with remove_prefix applied
                            elif self.remove_prefix_in_ckpt and key.startswith(self.remove_prefix_in_ckpt + "dit2.perception_head."):
                                new_key = key.replace(self.remove_prefix_in_ckpt + "dit2.perception_head.", "")
                                perception_head_state_dict_dit2[new_key] = value
                        
                        if perception_head_state_dict_dit2:
                            try:
                                dit2_model.perception_head.load_state_dict(perception_head_state_dict_dit2, strict=False)
                                print(f"[INFO] Loaded perception_head state dict for dit2 with {len(perception_head_state_dict_dit2)} parameters", flush=True)
                            except Exception as e:
                                print(f"[WARNING] Failed to load perception_head state dict for dit2: {e}", flush=True)
                        
                        # Initialize and load depth_head for dit2
                        if not hasattr(dit2_model, "depth_head") or dit2_model.depth_head is None:
                            num_layers = len(dit2_model.blocks) if hasattr(dit2_model, "blocks") else 30
                            if not hasattr(dit2_model, "start_layer"):
                                dit2_model.start_layer = 0
                            if not hasattr(dit2_model, "end_layer"):
                                dit2_model.end_layer = num_layers - 1
                            dit2_model.depth_head = SemanticFPNHead(
                                in_channels=getattr(dit2_model, "dim", 1536),
                                out_channels=getattr(dit2_model, "in_dim", 16),
                                num_tensors=dit2_model.end_layer - dit2_model.start_layer + 1,
                                patch_size=getattr(dit2_model, "patch_size", (1, 2, 2)),
                            )
                            dit2_model.depth_head = dit2_model.depth_head.to(dtype=pipe.torch_dtype).to(pipe.device)
                            print(f"[INFO] Depth head initialized for dit2 model with {num_layers} layers", flush=True)
                        
                        # Load depth_head state dict for dit2 if available
                        depth_head_state_dict_dit2 = {}
                        for key, value in checkpoint_state_dict.items():
                            # Handle different prefix patterns
                            if key.startswith("vace.dit2.depth_head."):
                                new_key = key.replace("vace.dit2.depth_head.", "")
                                depth_head_state_dict_dit2[new_key] = value
                            elif key.startswith("pipe.dit2.depth_head."):
                                new_key = key.replace("pipe.dit2.depth_head.", "")
                                depth_head_state_dict_dit2[new_key] = value
                            elif key.startswith("dit2.depth_head."):
                                new_key = key.replace("dit2.depth_head.", "")
                                depth_head_state_dict_dit2[new_key] = value
                            # Also check with remove_prefix applied
                            elif self.remove_prefix_in_ckpt and key.startswith(self.remove_prefix_in_ckpt + "dit2.depth_head."):
                                new_key = key.replace(self.remove_prefix_in_ckpt + "dit2.depth_head.", "")
                                depth_head_state_dict_dit2[new_key] = value
                        
                        if depth_head_state_dict_dit2:
                            try:
                                dit2_model.depth_head.load_state_dict(depth_head_state_dict_dit2, strict=False)
                                print(f"[INFO] Loaded depth_head state dict for dit2 with {len(depth_head_state_dict_dit2)} parameters", flush=True)
                            except Exception as e:
                                print(f"[WARNING] Failed to load depth_head state dict for dit2: {e}", flush=True)
                    
                    # Explicitly load multiview_ipadapter state dict if it exists
                    if hasattr(pipe, "multiview_ipadapter") and pipe.multiview_ipadapter is not None:
                        print(f"[INFO] Detected multiview_ipadapter model, loading parameters from checkpoint...", flush=True)
                        
                        # Collect multiview_ipadapter parameters from checkpoint
                        # Check both with and without prefix
                        ipadapter_state_dict = {}
                        prefixes = [p.strip() for p in self.remove_prefix_in_ckpt.split(",") if p.strip()] if self.remove_prefix_in_ckpt else []
                        
                        for key, value in checkpoint_state_dict.items():
                            # Check if this is an ipadapter key
                            is_ipadapter_key = (
                                key.startswith("image_proj") or 
                                key.startswith("ipadapter_modules") or
                                "ipadapter" in key.lower()
                            )
                            
                            if is_ipadapter_key:
                                # Remove prefix if present
                                new_key = key
                                for prefix in prefixes:
                                    if new_key.startswith(prefix):
                                        new_key = new_key.replace(prefix, "", 1)
                                        break
                                # Also check for pipe.multiview_ipadapter prefix
                                if new_key.startswith("pipe.multiview_ipadapter."):
                                    new_key = new_key.replace("pipe.multiview_ipadapter.", "")
                                elif new_key.startswith("multiview_ipadapter."):
                                    new_key = new_key.replace("multiview_ipadapter.", "")
                                
                                ipadapter_state_dict[new_key] = value
                        
                        if ipadapter_state_dict:
                            try:
                                # Load multiview_ipadapter parameters
                                load_result = pipe.multiview_ipadapter.load_state_dict(ipadapter_state_dict, strict=False)
                                loaded_count = len(ipadapter_state_dict) - len(load_result[0])
                                print(f"[INFO] Loaded {loaded_count}/{len(ipadapter_state_dict)} multiview_ipadapter parameters from checkpoint", flush=True)
                                if len(load_result[0]) > 0:
                                    print(f"[WARNING] {len(load_result[0])} multiview_ipadapter parameters not found in model (showing first 5): {load_result[0][:5]}", flush=True)
                                if len(load_result[1]) > 0:
                                    print(f"[WARNING] {len(load_result[1])} unexpected multiview_ipadapter parameters (showing first 5): {load_result[1][:5]}", flush=True)
                            except Exception as e:
                                print(f"[WARNING] Failed to load multiview_ipadapter parameters: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"[WARNING] No multiview_ipadapter parameters found in checkpoint (checked {len(checkpoint_state_dict)} keys)", flush=True)
                    else:
                        # Check if checkpoint contains ipadapter parameters but model doesn't have it
                        has_ipadapter_in_ckpt = any(
                            key.startswith("image_proj") or key.startswith("ipadapter_modules") or "ipadapter" in key.lower()
                            for key in checkpoint_state_dict.keys()
                        )
                        if has_ipadapter_in_ckpt:
                            print(f"[WARNING] Checkpoint contains multiview_ipadapter parameters, but model doesn't have multiview_ipadapter initialized", flush=True)
                            print(f"[WARNING] Make sure to set multiview_reference_mode to 'ipadapter' or 'temporal_concat+ipadapter' to initialize it", flush=True)
                    
                    # Explicitly load multiview_feature_bank_adapter state dict if it exists (resume_from_checkpoint)
                    if hasattr(pipe, "multiview_feature_bank_adapter") and pipe.multiview_feature_bank_adapter is not None:
                        print(f"[INFO] Detected multiview_feature_bank_adapter, loading parameters from checkpoint...", flush=True)
                        fb_adapter_state_dict = {}
                        prefixes = [p.strip() for p in self.remove_prefix_in_ckpt.split(",") if p.strip()] if self.remove_prefix_in_ckpt else []
                        for key, value in checkpoint_state_dict.items():
                            is_fb_key = key.startswith("fb_modules.") or key.startswith("image_proj.")
                            if is_fb_key:
                                new_key = key
                                for prefix in prefixes:
                                    if "feature_bank" in prefix and new_key.startswith(prefix):
                                        new_key = new_key.replace(prefix, "", 1)
                                        break
                                if new_key.startswith("pipe.multiview_feature_bank_adapter."):
                                    new_key = new_key.replace("pipe.multiview_feature_bank_adapter.", "")
                                elif new_key.startswith("multiview_feature_bank_adapter."):
                                    new_key = new_key.replace("multiview_feature_bank_adapter.", "")
                                fb_adapter_state_dict[new_key] = value
                        if fb_adapter_state_dict:
                            try:
                                load_result = pipe.multiview_feature_bank_adapter.load_state_dict(fb_adapter_state_dict, strict=False)
                                loaded_count = len(fb_adapter_state_dict) - len(load_result[0])
                                print(f"[INFO] Loaded {loaded_count}/{len(fb_adapter_state_dict)} multiview_feature_bank_adapter parameters from checkpoint", flush=True)
                                if len(load_result[0]) > 0:
                                    print(f"[WARNING] {len(load_result[0])} multiview_feature_bank_adapter parameters not found in model (showing first 5): {load_result[0][:5]}", flush=True)
                                if len(load_result[1]) > 0:
                                    print(f"[WARNING] {len(load_result[1])} unexpected multiview_feature_bank_adapter parameters (showing first 5): {load_result[1][:5]}", flush=True)
                            except Exception as e:
                                print(f"[WARNING] Failed to load multiview_feature_bank_adapter parameters: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"[WARNING] No multiview_feature_bank_adapter parameters found in checkpoint (checked keys with fb_modules./image_proj.)", flush=True)
                    
                    # Explicitly load multiview cross-view attention parameters if they exist
                    if hasattr(pipe, "vace") and pipe.vace is not None:
                        vace_model = pipe.vace
                        # Check if this is a multiview VACE model
                        if hasattr(vace_model, "vace_blocks") and len(vace_model.vace_blocks) > 0:
                            first_block = vace_model.vace_blocks[0]
                            if hasattr(first_block, "dit_block") and hasattr(first_block.dit_block, "cross_view_attn"):
                                print(f"[INFO] Detected multiview VACE model, loading cross-view attention parameters...", flush=True)
                                
                                # Collect cross-view attention parameters from checkpoint
                                cross_view_attn_state_dict = {}
                                for key, value in checkpoint_state_dict.items():
                                    # Handle different prefix patterns for cross-view attention
                                    if "cross_view_attn" in key:
                                        # Remove various prefixes
                                        new_key = key
                                        if key.startswith("pipe.vace."):
                                            new_key = key.replace("pipe.vace.", "")
                                        elif key.startswith("vace."):
                                            new_key = key.replace("vace.", "")
                                        # Also check with remove_prefix applied
                                        if self.remove_prefix_in_ckpt:
                                            prefixes = [p.strip() for p in self.remove_prefix_in_ckpt.split(",") if p.strip()]
                                            for prefix in prefixes:
                                                if new_key.startswith(prefix):
                                                    new_key = new_key.replace(prefix, "", 1)
                                                    break
                                        
                                        cross_view_attn_state_dict[new_key] = value
                                
                                if cross_view_attn_state_dict:
                                    try:
                                        # Load cross-view attention parameters
                                        load_result = vace_model.load_state_dict(cross_view_attn_state_dict, strict=False)
                                        loaded_count = len(cross_view_attn_state_dict) - len(load_result[0])
                                        print(f"[INFO] Loaded {loaded_count} cross-view attention parameters from checkpoint", flush=True)
                                        if len(load_result[0]) > 0:
                                            print(f"[WARNING] {len(load_result[0])} cross-view attention parameters not found in model (showing first 5): {load_result[0][:5]}", flush=True)
                                        if len(load_result[1]) > 0:
                                            print(f"[WARNING] {len(load_result[1])} unexpected cross-view attention parameters (showing first 5): {load_result[1][:5]}", flush=True)
                                    except Exception as e:
                                        print(f"[WARNING] Failed to load cross-view attention parameters: {e}", flush=True)
                                        import traceback
                                        traceback.print_exc()
                                else:
                                    print(f"[WARNING] No cross-view attention parameters found in checkpoint. They will be re-initialized.", flush=True)
                                    # If checkpoint doesn't have cross-view attention parameters, re-initialize them
                                    if hasattr(vace_model, "init_cross_view_attn_from_self_attn"):
                                        try:
                                            vace_model.init_cross_view_attn_from_self_attn()
                                            print(f"[INFO] Re-initialized cross-view attention from self-attention weights", flush=True)
                                        except Exception as e:
                                            print(f"[WARNING] Failed to re-initialize cross-view attention: {e}", flush=True)
                except ImportError as e:
                    print(f"[WARNING] Could not import SemanticFPNHead: {e}. Skipping depth_head and perception_head loading.", flush=True)
                except Exception as e:
                    print(f"[WARNING] Error loading depth_head and perception_head: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
            
            return True
        elif os.path.isdir(checkpoint_path):
            # Try to find the latest checkpoint in the directory
            checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors") and f.startswith("step-")]
            if checkpoint_files:
                # Sort by step number
                checkpoint_files.sort(key=lambda x: int(x.replace("step-", "").replace(".safetensors", "")))
                latest_checkpoint = os.path.join(checkpoint_path, checkpoint_files[-1])
                print(f"[INFO] Found latest checkpoint: {latest_checkpoint}", flush=True)
                return self.load_model_checkpoint(latest_checkpoint, accelerator, model)
            else:
                print(f"[WARNING] No checkpoint files found in {checkpoint_path}", flush=True)
                return False
        else:
            print(f"[WARNING] Checkpoint path does not exist: {checkpoint_path}", flush=True)
            return False


def _simple_collate_fn(batch):
    """
    Simple collate function that returns the first element of the batch.
    This is a top-level function (not a lambda) so it can be pickled when using 'spawn' multiprocessing.
    """
    return batch[0]


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    resume_from_checkpoint: str = None,
    resume_from_checkpoint_weights_only: bool = False,
    validation_dataset: torch.utils.data.Dataset = None,
    validation_sample_idx: int = -1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        if hasattr(args, 'resume_from_checkpoint'):
            resume_from_checkpoint = args.resume_from_checkpoint
        resume_from_checkpoint_weights_only = getattr(args, 'resume_from_checkpoint_weights_only', False)
        prefetch_factor = getattr(args, 'prefetch_factor', 4) if num_workers > 0 else None
        # Check if gradient checkpointing offload is enabled - if so, disable pin_memory and persistent_workers
        # to reduce memory pressure (gradient checkpointing offload moves tensors to CPU, competing with DataLoader workers)
        use_gradient_checkpointing_offload = getattr(args, 'use_gradient_checkpointing_offload', False)
    else:
        prefetch_factor = 4 if num_workers > 0 else None
        use_gradient_checkpointing_offload = False
    
    # Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization errors
    # This is required when using CUDA with DataLoader workers
    # Must be set after num_workers is reassigned from args.dataset_num_workers
    import multiprocessing
    if num_workers > 0:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError as e:
            print(f"[WARNING] Failed to set multiprocessing start method to 'spawn': {e}", flush=True)
            # Start method can only be set once, so if it's already set, that's fine
            pass
    else:
        print(f"[INFO] No workers for data loading", flush=True)
    
    # When using gradient checkpointing offload, disable pin_memory and persistent_workers
    # to avoid memory competition between DataLoader workers and CPU offloading
    if use_gradient_checkpointing_offload and num_workers > 0:
        pin_memory = False
        persistent_workers = False
        print("[INFO] Gradient checkpointing offload detected - disabling pin_memory and persistent_workers to reduce memory pressure", flush=True)
    else:
        pin_memory = True  # Faster GPU transfer
        persistent_workers = True if num_workers > 0 else False  # Keep workers alive between epochs
    
    print("[INFO] using num_workers: ", num_workers, flush=True)
    print("[INFO] using prefetch_factor: ", prefetch_factor, flush=True)
    print("[INFO] using use_gradient_checkpointing_offload: ", use_gradient_checkpointing_offload, flush=True)
    print("[INFO] using pin_memory: ", pin_memory, flush=True)
    print("[INFO] using persistent_workers: ", persistent_workers, flush=True)
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    # Optimized DataLoader with multiple workers for faster image loading
    # pin_memory=True speeds up GPU transfer, persistent_workers=True keeps workers alive between epochs
    # However, these are disabled when using gradient checkpointing offload to avoid memory competition
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=True, 
        collate_fn=_simple_collate_fn, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,  # Prefetch batches
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # Load checkpoint and training state if resuming
    global_step = 0
    start_epoch = 0
    if resume_from_checkpoint is not None:
        print(f"[INFO] Resuming training from checkpoint: {resume_from_checkpoint}", flush=True)
        # Load model checkpoint
        model_logger.load_model_checkpoint(resume_from_checkpoint, accelerator, model)
        # Load training state (skip optimizer/scheduler when weights_only to avoid OOM)
        if resume_from_checkpoint_weights_only:
            print(f"[INFO] resume_from_checkpoint_weights_only=True: skipping optimizer/scheduler state to reduce memory", flush=True)
            print(f"[WARNING] Training curve may show a jump: optimizer/scheduler/step are not restored.", flush=True)
        else:
            training_state = model_logger.load_training_state(resume_from_checkpoint, accelerator, optimizer, scheduler)
            if training_state is not None:
                # Restore optimizer and scheduler states
                optimizer.load_state_dict(training_state["optimizer"])
                scheduler.load_state_dict(training_state["scheduler"])
                # Use num_steps as canonical step so global_step matches checkpoint (e.g. step-500 => next log at 500)
                model_logger.num_steps = training_state.get("num_steps", training_state["global_step"])
                global_step = model_logger.num_steps
                start_epoch = training_state["epoch_id"]
                print(f"[INFO] Resumed from step {global_step}, epoch {start_epoch} (optimizer/scheduler state restored)", flush=True)
            else:
                print(f"[WARNING] No training state file found (e.g. step-XXX_training_state.pt). Only model weights were loaded.", flush=True)
                print(f"[WARNING] Training curve may show a jump: optimizer/scheduler/global_step start from zero.", flush=True)
    
    # Initialize TensorBoard writer (will continue from existing logs automatically)
    writer = None
    log_dir = None
    if TENSORBOARD_AVAILABLE and accelerator.is_main_process:
        log_dir = os.path.join(model_logger.output_path, "tensorboard_logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        if resume_from_checkpoint is not None:
            print(f"[INFO] Continuing TensorBoard logging from step {global_step}", flush=True)
        print(f"[INFO] To view logs, run: tensorboard --logdir {log_dir}", flush=True)
    
    for epoch_id in range(start_epoch, num_epochs):
        for step, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_id}")):
            if DEBUG_TIME:
                # Timing: data loading is already done by dataloader, we track from here
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_start = time.time()
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                if DEBUG_TIME:
                    # Timing: forward pass
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    forward_start = time.time()
                
                if hasattr(dataset, 'load_from_cache') and dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                
                if DEBUG_TIME:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    forward_end = time.time()
                    forward_time = forward_end - forward_start
                
                    # Timing: backward pass
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    backward_start = time.time()
                
                accelerator.backward(loss)

                if DEBUG_TIME:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    backward_end = time.time()
                    backward_time = backward_end - backward_start
                
                # Print the loss (every 10 steps, or on every step).
                if accelerator.is_main_process and (step % 10 == 0 or step == 0):
                    print(f"Epoch {epoch_id}, Step {step}, Global Step {global_step}, Loss: {loss.item():.6f}", flush=True)

                if DEBUG_TIME:
                    # Get timing info from model if available (handle wrapped models)
                    unwrapped_model = model.module if hasattr(model, 'module') else model
                    preprocess_time = getattr(unwrapped_model, '_preprocess_time', 0.0)
                    training_loss_time = getattr(unwrapped_model, '_training_loss_time', 0.0)
                    forward_time_total = getattr(unwrapped_model, '_forward_time', forward_time)
                    
                    print(f"  [Timing] Preprocess: {preprocess_time*1000:.2f}ms | Training Loss (sampling+forward): {training_loss_time*1000:.2f}ms | Forward Total: {forward_time_total*1000:.2f}ms | Backward: {backward_time*1000:.2f}ms", flush=True)
            
            # Timing: optimizer step
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_start = time.time()
            
            optimizer.step()
            
            if DEBUG_TIME:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                optimizer_end = time.time()
                optimizer_time = optimizer_end - optimizer_start
            
            model_logger.on_step_end(accelerator, model, save_steps, optimizer, scheduler, global_step, epoch_id)
            
            if DEBUG_TIME:
                # Timing: scheduler step
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                scheduler_start = time.time()
            
            scheduler.step()
            
            if DEBUG_TIME:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                scheduler_end = time.time()
                scheduler_time = scheduler_end - scheduler_start
            
            # Print detailed timing every 10 steps
            if accelerator.is_main_process and DEBUG_TIME:
                step_end = time.time()
                total_step_time = step_end - step_start
                print(f"  [Timing] Optimizer Step: {optimizer_time*1000:.2f}ms | Scheduler Step: {scheduler_time*1000:.2f}ms | Total Step: {total_step_time*1000:.2f}ms", flush=True)
            
            # Log to TensorBoard
            if writer is not None and accelerator.is_main_process:
                # Log loss
                writer.add_scalar("train/loss", loss.item(), global_step)
                # Log learning rate
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else learning_rate
                writer.add_scalar("train/learning_rate", current_lr, global_step)
                # Flush to ensure data is written
                if step % 10 == 0:
                    writer.flush()
            
            global_step += 1
        
        # Save training state at end of epoch
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id, optimizer, scheduler, global_step, save_steps, validation_dataset, validation_sample_idx, writer=writer)
            model_logger.save_training_state(accelerator, optimizer, scheduler, global_step, epoch_id + 1, save_steps)
        
        # Log epoch-level metrics
        if writer is not None and accelerator.is_main_process:
            writer.add_scalar("train/epoch", epoch_id, global_step)
    
    # Close TensorBoard writer
    if writer is not None and log_dir is not None:
        writer.close()
        print(f"[INFO] TensorBoard logging completed. Logs saved to: {log_dir}", flush=True)
    
    model_logger.on_training_end(accelerator, model, save_steps)
    # Save final training state
    if accelerator.is_main_process:
        model_logger.save_training_state(accelerator, optimizer, scheduler, global_step, num_epochs, save_steps)


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        prefetch_factor = getattr(args, 'prefetch_factor', 2) if num_workers > 0 else None
    else:
        prefetch_factor = 2 if num_workers > 0 else None
    
    # Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization errors
    # This is required when using CUDA with DataLoader workers
    # Must be set after num_workers is reassigned from args.dataset_num_workers
    import multiprocessing
    if num_workers > 0:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError as e:
            print(f"[WARNING] Failed to set multiprocessing start method to 'spawn': {e}", flush=True)
            # Start method can only be set once, so if it's already set, that's fine
            pass
    else:
        print(f"[INFO] No workers for data loading", flush=True)
        
    # Optimized DataLoader with multiple workers for faster image loading
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=False, 
        collate_fn=_simple_collate_fn, 
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=prefetch_factor,  # Prefetch batches
    )
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer directory, e.g., /path/to/google/umt5-xxl/")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Model ID with origin paths to the audio processor config, e.g., Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--prefetch_factor", type=int, default=0, help="Number of batches to prefetch per worker. Only used when num_workers > 0.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory or file to resume training from. Will load model, optimizer, scheduler, and training state.")
    parser.add_argument("--resume_from_checkpoint_weights_only", action="store_true", help="When resuming, load only model weights and skip optimizer/scheduler state. Use this to avoid OOM when full resume loads too much memory (optimizer states are ~2x model size).")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=2, help="Number of workers for data loading. Recommended: 4-8 for faster training.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser
