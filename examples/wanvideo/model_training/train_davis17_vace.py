"""
使用 VACE 数据集（DAVIS17 / HumanM3 / PennAction / DPW3D / ROSE / VIPSeg 等）
进行 Wan2.1-VACE-1.3B 训练的脚本
"""
import torch, os, json, sys, importlib, warnings, time, random
import numpy as np
import multiprocessing
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser, MultiDatasetLoader
from diffsynth.trainers.davis17_vace_dataset import DAVIS17_VACE_Dataset
from diffsynth.trainers.humanm3_vace_dataset import HumanM3_VACE_Dataset
from diffsynth.trainers.penn_action_vace_dataset import PennAction_VACE_Dataset
from diffsynth.trainers.dpw3d_vace_dataset import DPW3D_VACE_Dataset
import shutil, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEBUG_TIME = False
# Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization errors
# This must be done before any CUDA operations or DataLoader creation
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method can only be set once, so if it's already set, that's fine
    pass

# Suppress CUDA compilation-related warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")

def _ensure_hf_datasets():
    """
    Ensure that the Hugging Face `datasets` package is imported instead of the
    similarly named module shipped with croco (added to sys.path by dust3r).
    """
    croco_paths = [p for p in sys.path if p.endswith("/V2M4/extensions/croco")]
    removed = []
    for path in croco_paths:
        try:
            sys.path.remove(path)
            removed.append(path)
        except ValueError:
            continue

    hf_module = None
    try:
        hf_module = importlib.import_module("datasets")
    except ImportError as exc:
        for path in reversed(removed):
            if path not in sys.path:
                sys.path.insert(0, path)
        raise ImportError(
            "Unable to import the Hugging Face `datasets` package. "
            "Please ensure it is installed (pip install datasets)."
        ) from exc

    for path in reversed(removed):
        if path not in sys.path:
            sys.path.insert(0, path)

    if hf_module is not None:
        sys.modules["datasets"] = hf_module

_ensure_hf_datasets()


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        multiview_reference_mode="temporal_concat",
        multiview_zero_conv_scale=1.0,
        multiview_ipadapter_scale=1.0,
        use_multiview_consistency_check=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        use_perception_head=False,
        use_depth_head=False,
        lambda_latent_segmentation=1.0,
        lambda_latent_depth=1.0,
        lambda_temporal_coherence=0.0,
        temporal_coherence_method="raft",
        raft_model_path=None,
        lambda_multiview_dino_viewpoint=0.0,
        multiview_dino_model_path=None,
        frame_wise_mask_weighting=False,
        mask_weight_min_ratio=0.01,
        mask_weight_max_weight=10.0,
        mask_weight_power=0.5,
        mask_weight_base=0.5,
        train_vace_multiscale_fusion=False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        # Initialize tokenizer config
        tokenizer_config = None
        if tokenizer_path is not None:
            tokenizer_config = ModelConfig(path=tokenizer_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config, tokenizer_config=tokenizer_config)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Initialize perception head if needed
        if use_perception_head:
            from diffsynth.models.wan_video_segmentation import SemanticFPNHead
            # Add perception head to dit model if it doesn't exist
            if hasattr(self.pipe, "dit") and self.pipe.dit is not None:
                dit_model = self.pipe.dit
                if not hasattr(dit_model, "perception_head") or dit_model.perception_head is None:
                    num_layers = len(dit_model.blocks) if hasattr(dit_model, "blocks") else 30
                    dit_model.start_layer = 0
                    dit_model.end_layer = num_layers - 1
                    dit_model.perception_head = SemanticFPNHead(
                        in_channels=getattr(dit_model, "dim", 1536),
                        out_channels=getattr(dit_model, "in_dim", 16),  # Match VAE channels
                        num_tensors=dit_model.end_layer - dit_model.start_layer + 1,
                        patch_size=getattr(dit_model, "patch_size", (1, 2, 2)),
                    )
                    dit_model.perception_head = dit_model.perception_head.to(dtype=torch.bfloat16)
                    print(f"[INFO] Perception head initialized for dit model with {num_layers} layers", flush=True)
            # Also add to dit2 if exists
            if hasattr(self.pipe, "dit2") and self.pipe.dit2 is not None:
                dit2_model = self.pipe.dit2
                if not hasattr(dit2_model, "perception_head") or dit2_model.perception_head is None:
                    num_layers = len(dit2_model.blocks) if hasattr(dit2_model, "blocks") else 30
                    dit2_model.start_layer = 0
                    dit2_model.end_layer = num_layers - 1
                    dit2_model.perception_head = SemanticFPNHead(
                        in_channels=getattr(dit2_model, "dim", 1536),
                        out_channels=getattr(dit2_model, "in_dim", 16),  # Match VAE channels
                        num_tensors=dit2_model.end_layer - dit2_model.start_layer + 1,
                        patch_size=getattr(dit2_model, "patch_size", (1, 2, 2)),
                    )
                    dit2_model.perception_head = dit2_model.perception_head.to(dtype=torch.bfloat16)
                    print(f"[INFO] Perception head initialized for dit2 model with {num_layers} layers", flush=True)
        
        # Initialize depth head if needed
        if use_depth_head:
            from diffsynth.models.wan_video_segmentation import SemanticFPNHead
            # Add depth head to dit model if it doesn't exist
            if hasattr(self.pipe, "dit") and self.pipe.dit is not None:
                dit_model = self.pipe.dit
                if not hasattr(dit_model, "depth_head") or dit_model.depth_head is None:
                    num_layers = len(dit_model.blocks) if hasattr(dit_model, "blocks") else 30
                    if not hasattr(dit_model, "start_layer"):
                        dit_model.start_layer = 0
                    if not hasattr(dit_model, "end_layer"):
                        dit_model.end_layer = num_layers - 1
                    dit_model.depth_head = SemanticFPNHead(
                        in_channels=getattr(dit_model, "dim", 1536),
                        out_channels=getattr(dit_model, "in_dim", 16),  # Match VAE channels
                        num_tensors=dit_model.end_layer - dit_model.start_layer + 1,
                        patch_size=getattr(dit_model, "patch_size", (1, 2, 2)),
                    )
                    dit_model.depth_head = dit_model.depth_head.to(dtype=torch.bfloat16)
                    print(f"[INFO] Depth head initialized for dit model with {num_layers} layers", flush=True)
            # Also add to dit2 if exists
            if hasattr(self.pipe, "dit2") and self.pipe.dit2 is not None:
                dit2_model = self.pipe.dit2
                if not hasattr(dit2_model, "depth_head") or dit2_model.depth_head is None:
                    num_layers = len(dit2_model.blocks) if hasattr(dit2_model, "blocks") else 30
                    if not hasattr(dit2_model, "start_layer"):
                        dit2_model.start_layer = 0
                    if not hasattr(dit2_model, "end_layer"):
                        dit2_model.end_layer = num_layers - 1
                    dit2_model.depth_head = SemanticFPNHead(
                        in_channels=getattr(dit2_model, "dim", 1536),
                        out_channels=getattr(dit2_model, "in_dim", 16),  # Match VAE channels
                        num_tensors=dit2_model.end_layer - dit2_model.start_layer + 1,
                        patch_size=getattr(dit2_model, "patch_size", (1, 2, 2)),
                    )
                    dit2_model.depth_head = dit2_model.depth_head.to(dtype=torch.bfloat16)
                    print(f"[INFO] Depth head initialized for dit2 model with {num_layers} layers", flush=True)
        
        # Ensure perception_head is trainable if use_perception_head is True
        if use_perception_head:
            # Set perception_head parameters to trainable for dit model
            if hasattr(self.pipe, "dit") and self.pipe.dit is not None:
                if hasattr(self.pipe.dit, "perception_head") and self.pipe.dit.perception_head is not None:
                    for param in self.pipe.dit.perception_head.parameters():
                        param.requires_grad = True
                    self.pipe.dit.perception_head.train()
                    print("[INFO] Perception head for dit model set to trainable", flush=True)
            # Set perception_head parameters to trainable for dit2 model
            if hasattr(self.pipe, "dit2") and self.pipe.dit2 is not None:
                if hasattr(self.pipe.dit2, "perception_head") and self.pipe.dit2.perception_head is not None:
                    for param in self.pipe.dit2.perception_head.parameters():
                        param.requires_grad = True
                    self.pipe.dit2.perception_head.train()
                    print("[INFO] Perception head for dit2 model set to trainable", flush=True)
        
        # Ensure depth_head is trainable if use_depth_head is True
        if use_depth_head:
            # Set depth_head parameters to trainable for dit model
            if hasattr(self.pipe, "dit") and self.pipe.dit is not None:
                if hasattr(self.pipe.dit, "depth_head") and self.pipe.dit.depth_head is not None:
                    for param in self.pipe.dit.depth_head.parameters():
                        param.requires_grad = True
                    self.pipe.dit.depth_head.train()
                    print("[INFO] Depth head for dit model set to trainable", flush=True)
            # Set depth_head parameters to trainable for dit2 model
            if hasattr(self.pipe, "dit2") and self.pipe.dit2 is not None:
                if hasattr(self.pipe.dit2, "depth_head") and self.pipe.dit2.depth_head is not None:
                    for param in self.pipe.dit2.depth_head.parameters():
                        param.requires_grad = True
                    self.pipe.dit2.depth_head.train()
                    print("[INFO] Depth head for dit2 model set to trainable", flush=True)
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.multiview_reference_mode = multiview_reference_mode or "temporal_concat"
        self.multiview_zero_conv_scale = 1.0 if multiview_zero_conv_scale is None else multiview_zero_conv_scale
        self.multiview_ipadapter_scale = 1.0 if multiview_ipadapter_scale is None else multiview_ipadapter_scale
        self.use_multiview_consistency_check = use_multiview_consistency_check

        # Initialize Multiview IP-Adapter if mode is "ipadapter" or "temporal_concat+ipadapter"
        use_ipadapter = (
            self.multiview_reference_mode == "ipadapter" or 
            (isinstance(self.multiview_reference_mode, str) and "ipadapter" in self.multiview_reference_mode)
        )
        if use_ipadapter:
            from diffsynth.models.wan_video_multiview_ipadapter import MultiviewIPAdapter
            # Get VACE layers from VACE model if available
            vace_layers = None
            if hasattr(self.pipe, "vace") and self.pipe.vace is not None:
                if hasattr(self.pipe.vace, "vace_layers_mapping"):
                    vace_layers = tuple(self.pipe.vace.vace_layers_mapping.keys())
                elif hasattr(self.pipe.vace, "vace_layers"):
                    vace_layers = self.pipe.vace.vace_layers
            
            # Get DiT config for IP-Adapter
            if hasattr(self.pipe, "dit") and self.pipe.dit is not None:
                dit_model = self.pipe.dit
                # Get num_heads from the first block's cross_attn (more reliable than model attribute)
                if hasattr(dit_model, "blocks") and len(dit_model.blocks) > 0:
                    first_block = dit_model.blocks[0]
                    if hasattr(first_block, "cross_attn") and hasattr(first_block.cross_attn, "num_heads"):
                        num_attention_heads = first_block.cross_attn.num_heads
                    else:
                        num_attention_heads = getattr(dit_model, "num_heads", 24)
                else:
                    num_attention_heads = getattr(dit_model, "num_heads", 24)
                dim = getattr(dit_model, "dim", 1536)
                attention_head_dim = dim // num_attention_heads
                print(f"[INFO] Detected num_attention_heads: {num_attention_heads}, dim: {dim}, attention_head_dim: {attention_head_dim}", flush=True)
                
                # Initialize IP-Adapter
                self.pipe.multiview_ipadapter = MultiviewIPAdapter(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=dim,
                    clip_embeddings_dim=1280,  # CLIP embedding dimension
                    num_tokens=16,
                    num_views=4,  # Default, can be adjusted
                    vace_layers=vace_layers if vace_layers is not None else (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
                )
                self.pipe.multiview_ipadapter = self.pipe.multiview_ipadapter.to(dtype=torch.bfloat16)
                print(f"[INFO] Multiview IP-Adapter initialized with {len(self.pipe.multiview_ipadapter.vace_layers)} layers", flush=True)
                
                # Set IP-Adapter to trainable if training
                if trainable_models is not None and "multiview_ipadapter" in trainable_models:
                    for param in self.pipe.multiview_ipadapter.parameters():
                        param.requires_grad = True
                    self.pipe.multiview_ipadapter.train()
                    print("[INFO] Multiview IP-Adapter set to trainable", flush=True)
                else:
                    # Freeze IP-Adapter by default (can be enabled via trainable_models)
                    for param in self.pipe.multiview_ipadapter.parameters():
                        param.requires_grad = False
                    self.pipe.multiview_ipadapter.eval()
                    print("[INFO] Multiview IP-Adapter frozen (set trainable_models='multiview_ipadapter' to train)", flush=True)

        # Initialize Multiview Feature Bank Adapter when mode is "feature_bank" or "temporal_concat+feature_bank"
        use_feature_bank = (
            self.multiview_reference_mode == "feature_bank" or
            (isinstance(self.multiview_reference_mode, str) and "feature_bank" in self.multiview_reference_mode)
        )
        if use_feature_bank:
            from diffsynth.models.wan_video_multiview_ipadapter import MultiviewFeatureBankAdapter
            vace_layers = None
            if hasattr(self.pipe, "vace") and self.pipe.vace is not None:
                if hasattr(self.pipe.vace, "vace_layers_mapping"):
                    vace_layers = tuple(self.pipe.vace.vace_layers_mapping.keys())
                elif hasattr(self.pipe.vace, "vace_layers"):
                    vace_layers = self.pipe.vace.vace_layers
            if hasattr(self.pipe, "dit") and self.pipe.dit is not None:
                dit_model = self.pipe.dit
                if hasattr(dit_model, "blocks") and len(dit_model.blocks) > 0:
                    first_block = dit_model.blocks[0]
                    if hasattr(first_block, "cross_attn") and hasattr(first_block.cross_attn, "num_heads"):
                        num_attention_heads = first_block.cross_attn.num_heads
                    else:
                        num_attention_heads = getattr(dit_model, "num_heads", 24)
                else:
                    num_attention_heads = getattr(dit_model, "num_heads", 24)
                dim = getattr(dit_model, "dim", 1536)
                attention_head_dim = dim // num_attention_heads
                print(f"[INFO] Multiview Feature Bank Adapter: num_heads={num_attention_heads}, dim={dim}", flush=True)
                self.pipe.multiview_feature_bank_adapter = MultiviewFeatureBankAdapter(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=dim,
                    clip_embeddings_dim=1280,
                    vace_layers=vace_layers if vace_layers is not None else (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
                )
                self.pipe.multiview_feature_bank_adapter = self.pipe.multiview_feature_bank_adapter.to(dtype=torch.bfloat16)
                print(f"[INFO] Multiview Feature Bank Adapter initialized with {len(self.pipe.multiview_feature_bank_adapter.vace_layers)} layers", flush=True)
                if trainable_models is not None and "multiview_feature_bank_adapter" in trainable_models:
                    for param in self.pipe.multiview_feature_bank_adapter.parameters():
                        param.requires_grad = True
                    self.pipe.multiview_feature_bank_adapter.train()
                    print("[INFO] Multiview Feature Bank Adapter set to trainable", flush=True)
                else:
                    for param in self.pipe.multiview_feature_bank_adapter.parameters():
                        param.requires_grad = False
                    self.pipe.multiview_feature_bank_adapter.eval()
                    print("[INFO] Multiview Feature Bank Adapter frozen (set trainable_models='multiview_feature_bank_adapter' to train)", flush=True)

        # Optional: train only VACE multi-scale fusion 1x1 heads (small bbox / FPN path).
        self.train_vace_multiscale_fusion = bool(train_vace_multiscale_fusion)
        if self.train_vace_multiscale_fusion:
            for vace_attr in ("vace", "vace2"):
                vace_m = getattr(self.pipe, vace_attr, None)
                if vace_m is None:
                    continue
                for fusion_name in ("vace_multiscale_fusion", "control_multiscale_fusion"):
                    mod_list = getattr(vace_m, fusion_name, None)
                    if mod_list is None or len(mod_list) == 0:
                        continue
                    for layer in mod_list:
                        for p in layer.parameters():
                            p.requires_grad = True
                        layer.train()
                    print(
                        f"[INFO] {vace_attr}.{fusion_name}: trainable (multiscale FPN fusion)",
                        flush=True,
                    )

        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.use_perception_head = use_perception_head
        self.use_depth_head = use_depth_head
        self.lambda_latent_segmentation = lambda_latent_segmentation
        self.lambda_latent_depth = lambda_latent_depth
        self.lambda_temporal_coherence = lambda_temporal_coherence
        self.temporal_coherence_method = temporal_coherence_method
        self.raft_model_path = raft_model_path
        self.lambda_multiview_dino_viewpoint = lambda_multiview_dino_viewpoint
        self.multiview_dino_model_path = multiview_dino_model_path
        self.frame_wise_mask_weighting = frame_wise_mask_weighting
        self.mask_weight_min_ratio = mask_weight_min_ratio
        self.mask_weight_max_weight = mask_weight_max_weight
        self.mask_weight_power = mask_weight_power
        self.mask_weight_base = mask_weight_base
        
        
    def forward_preprocess(self, data):
        if DEBUG_TIME:
            # Timing: start preprocessing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            preprocess_start = time.time()
        
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Always add multiview parameters (they're class parameters, not data inputs)
        inputs_shared["multiview_reference_mode"] = self.multiview_reference_mode
        inputs_shared["multiview_zero_conv_scale"] = self.multiview_zero_conv_scale
        inputs_shared["multiview_ipadapter_scale"] = self.multiview_ipadapter_scale
        # Prompt for cross-modal consistency check (CLIP text–image); optional multiview_reference_weight from data
        inputs_shared["prompt"] = data["prompt"]
        if self.use_multiview_consistency_check:
            if "multiview_reference_weight" in data:
                inputs_shared["multiview_reference_weight"] = data["multiview_reference_weight"]
            # else: leave unset so pipeline unit computes CLIP consistency
        else:
            inputs_shared["multiview_reference_weight"] = 1.0  # disable consistency check: always full reference weight
        # print("self.use_multiview_consistency_check:", self.use_multiview_consistency_check, flush=True)
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input in ("multiview_reference_mode", "multiview_zero_conv_scale", "multiview_ipadapter_scale"):
                # These are already injected from class configs above
                continue
            else:
                inputs_shared[extra_input] = data[extra_input]
                        
        # Add trajectory_maps if available
        if "trajectory_maps" in data:
            inputs_shared["trajectory_maps"] = data["trajectory_maps"]
            save_visualization = False
            if save_visualization:
                if not os.path.exists(f"./visualization"):
                    os.makedirs(f"./visualization")
                data["trajectory_maps"][0].save(f"./visualization/trajectory_maps_0.png")
        
        # Encode vace_video_mask to target_mask_latent if perception head is enabled
        if self.use_perception_head and "trajectory_maps" in data and data["trajectory_maps"] is not None:
            # Load VAE to device
            self.pipe.load_models_to_device(["vae"])
            
            # Preprocess mask images (same as VACE unit does)
            trajectory_maps = self.pipe.preprocess_video(data["trajectory_maps"], min_value=0, max_value=1)
            
            # Encode mask to latent space using same VAE encoder as video
            # Use same encoding parameters as training (tiled=False for training)
            target_mask_latent = self.pipe.vae.encode(
                trajectory_maps,
                device=self.pipe.device,
                tiled=inputs_shared.get("tiled", False),
                tile_size=inputs_shared.get("tile_size", None),
                tile_stride=inputs_shared.get("tile_stride", None)
            ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            
            # Convert from [B, C, T, H, W] to [B, T, C, H, W] to match segmentation head output format
            target_mask_latent = target_mask_latent.permute(0, 2, 1, 3, 4)
            
            # Segmentation head now outputs in_dim channels (matching VAE), so no channel truncation needed
            inputs_shared["target_mask_latent"] = target_mask_latent
        
        # Add target_mask_latent if already provided in data (for segmentation loss)
        elif "target_mask_latent" in data:
            inputs_shared["target_mask_latent"] = data["target_mask_latent"]
        
        # Encode depth_video to target_depth_latent if depth head is enabled
        if self.use_depth_head and "depth_video" in data and data["depth_video"] is not None:
            # Load VAE to device
            self.pipe.load_models_to_device(["vae"])

            save_visualization = False
            if save_visualization:
                if not os.path.exists(f"./visualization"):
                    os.makedirs(f"./visualization")
                data["depth_video"][0].save(f"./visualization/depth_video_0.png")
            
            # Preprocess depth video images (same as regular video)
            depth_video = self.pipe.preprocess_video(data["depth_video"], min_value=0, max_value=1)
            
            # Encode depth video to latent space using same VAE encoder as video
            # Use same encoding parameters as training (tiled=False for training)
            target_depth_latent = self.pipe.vae.encode(
                depth_video,
                device=self.pipe.device,
                tiled=inputs_shared.get("tiled", False),
                tile_size=inputs_shared.get("tile_size", None),
                tile_stride=inputs_shared.get("tile_stride", None)
            ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            
            # Convert from [B, C, T, H, W] to [B, T, C, H, W] to match depth head output format
            target_depth_latent = target_depth_latent.permute(0, 2, 1, 3, 4)
            
            # Depth head outputs in_dim channels (matching VAE), so no channel truncation needed
            inputs_shared["target_depth_latent"] = target_depth_latent
        
        # Add target_depth_latent if already provided in data (for depth loss)
        elif "target_depth_latent" in data:
            inputs_shared["target_depth_latent"] = data["target_depth_latent"]
        
        # Add lambda_latent_segmentation if available (for segmentation loss weight)
        if "lambda_latent_segmentation" in data:
            inputs_shared["lambda_latent_segmentation"] = data["lambda_latent_segmentation"]
        elif self.use_perception_head:
            # Use the configured lambda value
            inputs_shared["lambda_latent_segmentation"] = self.lambda_latent_segmentation
        
        # Add lambda_latent_depth if available (for depth loss weight)
        if "lambda_latent_depth" in data:
            inputs_shared["lambda_latent_depth"] = data["lambda_latent_depth"]
        elif self.use_depth_head:
            # Use the configured lambda value
            inputs_shared["lambda_latent_depth"] = self.lambda_latent_depth
        
        # Add lambda_temporal_coherence if available (for temporal coherence loss weight)
        if "lambda_temporal_coherence" in data:
            inputs_shared["lambda_temporal_coherence"] = data["lambda_temporal_coherence"]
        elif self.lambda_temporal_coherence > 0:
            # Use the configured lambda value
            inputs_shared["lambda_temporal_coherence"] = self.lambda_temporal_coherence
        if inputs_shared.get("lambda_temporal_coherence", 0.0) > 0:
            inputs_shared["temporal_coherence_method"] = data.get("temporal_coherence_method", self.temporal_coherence_method)
            if "raft_model_path" in data:
                inputs_shared["raft_model_path"] = data["raft_model_path"]
            elif self.raft_model_path is not None:
                inputs_shared["raft_model_path"] = self.raft_model_path

        if "lambda_multiview_dino_viewpoint" in data:
            inputs_shared["lambda_multiview_dino_viewpoint"] = data["lambda_multiview_dino_viewpoint"]
        elif self.lambda_multiview_dino_viewpoint > 0:
            inputs_shared["lambda_multiview_dino_viewpoint"] = self.lambda_multiview_dino_viewpoint

        if self.multiview_dino_model_path is not None:
            inputs_shared["multiview_dino_model_path"] = self.multiview_dino_model_path
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        
        # Calculate mask ratio for loss weighting (to enhance small object learning)
        # Support both global and frame-wise mask ratio calculation
        if "vace_video_mask" in inputs_shared and inputs_shared["vace_video_mask"] is not None:
            vace_video_mask = inputs_shared["vace_video_mask"]
            frame_mask_ratios = None
            
            # Handle different mask formats
            if isinstance(vace_video_mask, list):
                # List of PIL Images - calculate ratio per frame for frame-wise weighting
                from PIL import Image
                import numpy as np
                frame_mask_ratios_list = []
                total_pixels = 0
                foreground_pixels = 0
                for mask_img in vace_video_mask:
                    if isinstance(mask_img, Image.Image):
                        mask_array = np.array(mask_img.convert('L'))
                        total_pixels += mask_array.size
                        # Count pixels > 128 as foreground (assuming 0-255 range)
                        frame_foreground = (mask_array > 128).sum()
                        foreground_pixels += frame_foreground
                        # Calculate ratio for this frame
                        if mask_array.size > 0:
                            frame_mask_ratios_list.append(frame_foreground / mask_array.size)
                        else:
                            frame_mask_ratios_list.append(0.0)
                if frame_mask_ratios_list:
                    frame_mask_ratios = frame_mask_ratios_list
            
            if frame_mask_ratios is not None:
                inputs_shared["frame_mask_ratios"] = frame_mask_ratios
                if self.frame_wise_mask_weighting:
                    inputs_shared["frame_wise_mask_weighting"] = True
                    inputs_shared["mask_weight_min_ratio"] = self.mask_weight_min_ratio
                    inputs_shared["mask_weight_max_weight"] = self.mask_weight_max_weight
                    inputs_shared["mask_weight_power"] = self.mask_weight_power
                    inputs_shared["mask_weight_base"] = self.mask_weight_base
        
        if DEBUG_TIME:
            # Timing: end preprocessing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            preprocess_end = time.time()
            self._preprocess_time = preprocess_end - preprocess_start
        # print(f"inputs_shared keys: {inputs_shared.keys()}", flush=True)
        # print(f"inputs_posi keys: {inputs_posi.keys()}", flush=True)
        # print(f"inputs_nega keys: {inputs_nega.keys()}", flush=True)
        # print("--------------------------------")
        # for key in inputs_shared.keys():
        #     if isinstance(inputs_shared[key], torch.Tensor):
        #         print(f"{key}:", inputs_shared[key].shape)
        #     elif isinstance(inputs_shared[key], list):
        #         print(f"{key}:", len(inputs_shared[key]))
        #     else:
        #         print(f"{key}:", type(inputs_shared[key]))
        # for key in inputs_posi.keys():
        #     if isinstance(inputs_posi[key], torch.Tensor):
        #         print(f"{key}:", inputs_posi[key].shape)
        #     elif isinstance(inputs_posi[key], list):
        #         print(f"{key}:", len(inputs_posi[key]))
        #     else:
        #         print(f"{key}:", type(inputs_posi[key]))
        # for key in inputs_nega.keys():
        #     if isinstance(inputs_nega[key], torch.Tensor):
        #         print(f"{key}:", inputs_nega[key].shape)
        #     elif isinstance(inputs_nega[key], list):
        #         print(f"{key}:", len(inputs_nega[key]))
        #     else:
        #         print(f"{key}:", type(inputs_nega[key]))
        
        # inputs_shared keys: dict_keys(['input_video', 'height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'max_timestep_boundary', 'min_timestep_boundary', 'vace_video', 'vace_video_mask', 'multiview_reference_image', 'trajectory_maps', 'target_mask_latent', 'target_depth_latent', 'lambda_latent_segmentation', 'lambda_latent_depth', 'lambda_temporal_coherence', 'noise', 'latents', 'input_latents', 'vace_context', 'animate_pose_video', 'animate_face_video', 'animate_inpaint_video', 'animate_mask_video'])
        # inputs_posi keys: dict_keys(['prompt', 'context'])
        # inputs_nega keys: dict_keys(['context'])
        # --------------------------------
        # input_video: 81
        # height: <class 'int'>
        # width: <class 'int'>
        # num_frames: <class 'int'>
        # cfg_scale: <class 'int'>
        # tiled: <class 'bool'>
        # rand_device: <class 'torch.device'>
        # use_gradient_checkpointing: <class 'bool'>
        # use_gradient_checkpointing_offload: <class 'bool'>
        # cfg_merge: <class 'bool'>
        # vace_scale: <class 'int'>
        # max_timestep_boundary: <class 'float'>
        # min_timestep_boundary: <class 'float'>
        # vace_video: 81
        # vace_video_mask: 81
        # multiview_reference_image: 4
        # trajectory_maps: 81
        # target_mask_latent: torch.Size([1, 21, 16, 60, 104])
        # target_depth_latent: torch.Size([1, 21, 16, 60, 104])
        # lambda_latent_segmentation: <class 'float'>
        # lambda_latent_depth: <class 'float'>
        # lambda_temporal_coherence: <class 'float'>
        # noise: torch.Size([1, 16, 25, 60, 104])
        # latents: torch.Size([1, 16, 25, 60, 104])
        # input_latents: torch.Size([1, 16, 25, 60, 104])
        # vace_context: torch.Size([1, 96, 25, 60, 104])
        # animate_pose_video: <class 'NoneType'>
        # animate_face_video: <class 'NoneType'>
        # animate_inpaint_video: <class 'NoneType'>
        # animate_mask_video: <class 'NoneType'>
        # prompt: <class 'str'>
        # context: torch.Size([1, 512, 4096])
        # context: torch.Size([1, 512, 4096])
        
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if DEBUG_TIME:
            # Initialize timing attributes
            self._preprocess_time = 0.0
            self._training_loss_time = 0.0
            self._forward_time = 0.0
            
            # Timing: start forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_start = time.time()
        
        # Timing: preprocessing (only if inputs not provided)
        if inputs is None:
            inputs = self.forward_preprocess(data)
        else:
            if DEBUG_TIME:
                # If inputs are pre-provided, preprocessing time is 0
                self._preprocess_time = 0.0
        
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        if DEBUG_TIME:
        # Timing: start training_loss (sampling + forward)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_loss_start = time.time()
        
        loss = self.pipe.training_loss(**models, **inputs)
        
        if DEBUG_TIME:
            # Timing: end training_loss
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_loss_end = time.time()
            self._training_loss_time = training_loss_end - training_loss_start
            
            # Timing: end forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_end = time.time()
            self._forward_time = forward_end - forward_start
        
        return loss


if __name__ == "__main__":
    # from accelerate import Accelerator

    # acc = Accelerator()
    # print(">>> World size =", acc.num_processes)
    # print(">>> Global rank =", acc.process_index)
    # print(">>> Local rank  =", acc.local_process_index)
    # print(">>> Machine rank =", acc.process_index // acc.num_processes_per_node)
    parser = wan_parser()
    # Add the seed argument.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    # Add DAVIS17-specific arguments.
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Dataset split (DAVIS17: train/val; HumanM3/PennAction/DPW3D: train/test; ROSE: train/val/all).",
    )
    parser.add_argument("--min_instance_ratio", type=float, default=0.00, help="Minimum instance ratio for filtering.")
    parser.add_argument("--min_bbox_ratio", type=float, default=0.00, help="Minimum bbox ratio for filtering.")
    parser.add_argument("--use_masked_vace_video", action="store_true", help="If True, use masked images for vace_video; otherwise use original images.")
    parser.add_argument("--prompt_template", type=str, default=None, help="Prompt template string, e.g., 'a video of {class_name}'.")
    parser.add_argument("--debug_mode", action="store_true", help="If set, limit dataset to 10 samples for quick debugging.")
    parser.add_argument("--mask_foreground", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help="If True, mask foreground; otherwise mask background. Accepts: true/1/yes or false/0/no.")
    parser.add_argument("--use_rgb_depth_video", action="store_true", help="If True, use RGBD video; otherwise use RGB video.")
    parser.add_argument("--mask_strategy", type=str, default="bbox_cover_traj", choices=["fine", "bbox_with_traj", "bbox_cover_traj", "first_bbox_then_point"], help="Mask strategy: 'fine' uses fine mask, 'bbox_with_traj' uses bbox per frame, 'bbox_cover_traj' uses bbox covering all frames, 'first_bbox_then_point' fixes object size from the first bbox and moves it with per-frame center points.")
    parser.add_argument("--frame_selecting_strategy", type=str, default="farthest", choices=["nearest", "farthest"], help="Frame selecting strategy for mesh loading: 'nearest' or 'farthest'.")
    parser.add_argument("--bbox_scale", type=float, default=1.2, help="Scale factor for bbox size (bbox_scale times the size of the bbox).")
    parser.add_argument("--trajectory_type", type=str, default="mask", choices=["mask", "box", "sparse_box"], help="Type of trajectory maps: 'mask' uses fine mask, 'box' uses bounding boxes, 'sparse_box' uses sparse bounding boxes.")
    parser.add_argument("--sparse_box_interval", type=int, default=5, help="Interval for sparse_box trajectory (e.g., 5 means keep box every 5 frames).")
    parser.add_argument("--use_perception_head", action="store_true", help="If set, enable segmentation head for latent segmentation loss.")
    parser.add_argument("--use_depth_head", action="store_true", help="If set, enable depth head for latent depth loss.")
    parser.add_argument("--lambda_latent_segmentation", type=float, default=0.005, help="The weight of the latent segmentation loss in the total loss. Default is 0.005.")
    parser.add_argument("--lambda_latent_depth", type=float, default=0.005, help="The weight of the latent depth loss in the total loss. Default is 0.005.")
    parser.add_argument("--lambda_temporal_coherence", type=float, default=0.0, help="The weight of the temporal coherence loss in the total loss. Default is 0.0.")
    parser.add_argument("--temporal_coherence_method", type=str, default="raft", choices=["raft", "simple"], help="Optical flow backend for temporal coherence loss. Default is RAFT; use simple for the previous gradient-based method.")
    parser.add_argument("--raft_model_path", type=str, default=os.getenv("RAFT_MODEL_PATH", None), help="Path to RAFT checkpoint for temporal coherence loss. Defaults to RAFT_MODEL_PATH or VBench cache.")
    parser.add_argument("--lambda_multiview_dino_viewpoint", type=float, default=0.0, help="Weight of the DINO viewpoint control loss between generated first frame and multiview_reference_image[0]. Default is 0.0.")
    parser.add_argument("--multiview_dino_model_path", type=str, default=os.getenv("DINO_VIEW_CONTROL_MODEL_PATH", "facebook/dinov2-base"), help="Local path or HuggingFace id for the DINO model used by the multiview viewpoint control loss.")
    parser.add_argument("--frame_wise_mask_weighting", action="store_true", default=False, help="If set (default), use frame-wise mask weighting for diffusion loss when vace_video_mask is available.")
    parser.add_argument("--mask_weight_min_ratio", type=float, default=0.01, help="Minimum clamp for frame mask ratio in weight computation. Default 0.01.")
    parser.add_argument("--mask_weight_max_weight", type=float, default=10.0, help="Maximum frame weight in frame-wise loss. Default 10.0.")
    parser.add_argument("--mask_weight_power", type=float, default=0.5, help="Power exponent for frame weight: base * (1/ratio)^power. Default 0.5.")
    parser.add_argument("--mask_weight_base", type=float, default=0.5, help="Base multiplier for frame weight. Default 0.5.")
    parser.add_argument("--downsample_10hz", action="store_true", help="If set, downsample video frames to 10Hz (every 2nd or 3rd frame depending on dataset).")
    parser.add_argument("--enable_validation", action="store_true", help="If set, generate a validation video at the end of each epoch.")
    parser.add_argument("--validation_sample_idx", type=int, default=-1, help="Index of the validation sample to use for video generation. Default is 0.")
    parser.add_argument("--multiview_reference_mode", type=str, default="temporal_concat", choices=["temporal_concat", "zero_conv", "ipadapter", "temporal_concat+ipadapter", "temporal_concat+zero_conv", "temporal_concat+ref_gating", "temporal_concat+feature_bank", "feature_bank"], help="How to inject multiview_reference_image: temporal_concat, zero_conv, ipadapter, temporal_concat+ipadapter, or temporal_concat+zero_conv (temporal_concat+zero_conv uses both direct concat and control branch).")
    parser.add_argument("--multiview_zero_conv_scale", type=float, default=1.0, help="Scale for multiview zero-conv injection.")
    parser.add_argument("--multiview_ipadapter_scale", type=float, default=1.0, help="Scale for multiview IP-Adapter injection.")
    parser.add_argument("--use_multiview_consistency_check", action="store_true", help="If set, use CLIP text–image consistency to down-weight multiview reference when 3D render is corrupted; otherwise reference weight is always 1.0.")
    parser.add_argument(
        "--train_vace_multiscale_fusion",
        action="store_true",
        help="If set, set requires_grad on VACE vace_multiscale_fusion / control_multiscale_fusion (FPN-style branches). Use with VaceWanModel multiscale enabled.",
    )
    # IP-Adapter two-stage training (stage1: only multiview_ipadapter; stage2: load stage1 ckpt, train vace+multiview_ipadapter with small lr)
    parser.add_argument("--ipadapter_training_stage", type=int, default=None, choices=[1, 2], help="If 1: train only multiview_ipadapter (VACE/DiT frozen). If 2: resume from stage1 ckpt and train vace+multiview_ipadapter with small LR.")
    parser.add_argument("--ipadapter_stage2_resume_from", type=str, default=None, help="Path to stage-1 checkpoint (e.g. step-XXX.safetensors or dir). Required when ipadapter_training_stage=2.")
    parser.add_argument("--ipadapter_stage2_learning_rate", type=float, default=None, help="Learning rate for stage 2. If None, uses 0.1 * learning_rate.")
    # Multi-dataset support
    parser.add_argument("--dataset_list", type=str, default="DAVIS17,VIPSeg", help="Names of datasets to load, separated by commas.")
    parser.add_argument("--sampling_strategy", type=str, default="proportional", choices=["proportional", "uniform", "weighted"], help="Sampling strategy for MultiDatasetLoader: 'proportional' (default), 'uniform', or 'weighted'.")
    
    args = parser.parse_args()

    # Set the random seed to ensure reproducibility.
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic CUDA operations.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"[INFO] Random seed set to: {seed}", flush=True)
    
    set_seed(args.seed)

    # asser that extra_inputs not contain multiview_reference_image and vace_reference_image at the same time
    if "multiview_reference_image" in args.extra_inputs and "vace_reference_image" in args.extra_inputs:
        raise ValueError("multiview_reference_image and vace_reference_image cannot be used at the same time")

    # Check if we're on the main process (rank 0) for multi-machine training
    # RANK environment variable is set by distributed launchers (torchrun, accelerate, etc.)
    # If not set, we're likely running on a single machine, so proceed with directory creation
    rank = int(os.getenv("RANK", "0"))
    is_main_process = (rank == 0)

    # IP-Adapter two-stage training overrides
    if getattr(args, "ipadapter_training_stage", None) == 1:
        if "ipadapter" not in (args.multiview_reference_mode or ""):
            raise ValueError("ipadapter_training_stage=1 requires multiview_reference_mode to contain 'ipadapter' (e.g. temporal_concat+ipadapter).")
        args.trainable_models = "multiview_ipadapter"
        args.lora_base_model = None  # no VACE LoRA in stage 1; only train IP-Adapter
        if is_main_process:
            print("\n[INFO] IP-Adapter stage 1: training only multiview_ipadapter (VACE/DiT frozen)", flush=True)
    elif getattr(args, "ipadapter_training_stage", None) == 2:
        resume_from = getattr(args, "ipadapter_stage2_resume_from", None)
        if not resume_from:
            raise ValueError("ipadapter_training_stage=2 requires --ipadapter_stage2_resume_from pointing to a stage-1 checkpoint.")
        args.resume_from_checkpoint = resume_from
        lr_stage2 = getattr(args, "ipadapter_stage2_learning_rate", None)
        args.learning_rate = lr_stage2 if lr_stage2 is not None else (args.learning_rate * 0.1)
        if is_main_process:
            print(f"[INFO] IP-Adapter stage 2: resuming from {resume_from}, learning_rate={args.learning_rate}", flush=True)

    args.output_path = os.path.join(args.output_path, time.strftime("%Y%m%d_%H%M%S"))
    
    # Only create directories and copy files on the main process to avoid conflicts in multi-machine training
    if is_main_process:
        os.makedirs(args.output_path, exist_ok=True)

        # Dump all config to JSON file
        config_dict = vars(args)
        config_json_path = os.path.join(args.output_path, "training_config.json")
        os.makedirs(args.output_path, exist_ok=True)
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Config saved to: {config_json_path}", flush=True)

        # make a copy of key files to output path
        src_path = os.path.join(args.output_path, "src")
        os.makedirs(src_path, exist_ok=True)
        # DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-DAVIS17.sh
        # DiffSynth-Studio/diffsynth/models/wan_video_*.py
        # DiffSynth-Studio/diffsynth/pipelines/wan_video_new.py
        # DiffSynth-Studio/diffsynth/trainers/davis17_vace_dataset.py
        # DiffSynth-Studio/examples/wanvideo/model_training/train_davis17_vace.py
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        shutil.copy(os.path.join(project_root, "examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-DAVIS17.sh"), src_path)
        model_files = glob.glob(os.path.join(project_root, "diffsynth/models/wan_video_*.py"))
        for model_file in model_files:
            shutil.copy(model_file, src_path)
        shutil.copy(os.path.join(project_root, "diffsynth/pipelines/wan_video_new.py"), src_path)
        shutil.copy(os.path.join(project_root, "diffsynth/trainers/davis17_vace_dataset.py"), src_path)
        shutil.copy(os.path.join(project_root, "examples/wanvideo/model_training/train_davis17_vace.py"), src_path)
        dataset_names_upper = {n.strip().upper() for n in args.dataset_list.split(",")}
        if "HUMANM3" in dataset_names_upper:
            humanm3_sh = os.path.join(project_root, "examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-HumanM3.sh")
            if os.path.isfile(humanm3_sh):
                shutil.copy(humanm3_sh, src_path)
            shutil.copy(os.path.join(project_root, "diffsynth/trainers/humanm3_vace_dataset.py"), src_path)
        if "PENNACTION" in dataset_names_upper or "PENN_ACTION" in dataset_names_upper:
            penn_sh = os.path.join(project_root, "examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-PennAction.sh")
            if os.path.isfile(penn_sh):
                shutil.copy(penn_sh, src_path)
            shutil.copy(os.path.join(project_root, "diffsynth/trainers/penn_action_vace_dataset.py"), src_path)
        if "DPW3D" in dataset_names_upper or "3DPW" in dataset_names_upper:
            dpw_sh = os.path.join(project_root, "examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-DPW3D.sh")
            if os.path.isfile(dpw_sh):
                shutil.copy(dpw_sh, src_path)
            shutil.copy(os.path.join(project_root, "diffsynth/trainers/dpw3d_vace_dataset.py"), src_path)
    
    print("[INFO] ========================================", flush=True)
    print("[INFO] Starting training setup...\n", flush=True)
    print("[INFO] Note: First run may have delays due to CUDA kernel compilation", flush=True)
    print("[INFO] ========================================", flush=True)
    
    # Create dataset(s) - support both single and multi-dataset modes
    dataset_configs = []
    for dataset_name in args.dataset_list.split(","):
        if dataset_name == "DAVIS17":
            dataset_configs.append({
                'type': 'davis17',
                'weight': 1.0,
                'dataset_path': os.getenv("DAVIS17_DATASET_PATH"),
                'split_name': args.split_name,
                'length': args.num_frames if args.num_frames else 81,
                'target_resolution': (args.width, args.height),
                'min_instance_ratio': args.min_instance_ratio,
                'min_bbox_ratio': args.min_bbox_ratio,
                'use_masked_vace_video': args.use_masked_vace_video,
                'dataset_repeat': args.dataset_repeat,
                'max_samples': 3 if args.debug_mode else None,
                'mask_foreground': args.mask_foreground,
                'mask_strategy': args.mask_strategy,
                'frame_selecting_strategy': args.frame_selecting_strategy,
                'bbox_scale': args.bbox_scale,
                'trajectory_type': args.trajectory_type,
                'sparse_box_interval': args.sparse_box_interval,
                'use_rgb_depth_video': args.use_rgb_depth_video,
                'downsample_10hz': args.downsample_10hz,
            })
        elif dataset_name == "VIPSeg":
            dataset_configs.append({
                'type': 'vipseg',
                'weight': 1.0,
                'vipseg_dataset_path': os.getenv("VIPSEG_DATASET_PATH"),
                'vspw_dataset_path': os.getenv("VSPW_DATASET_PATH"),
                'split_name': args.split_name,
                'length': args.num_frames if args.num_frames else 81,
                'target_resolution': (args.width, args.height),
                'min_instance_ratio': 0.1,
                'min_bbox_ratio': 0.1,
                'use_masked_vace_video': args.use_masked_vace_video,
                'dataset_repeat': args.dataset_repeat,
                'mask_foreground': args.mask_foreground,
                'mask_strategy': args.mask_strategy,
                'frame_selecting_strategy': args.frame_selecting_strategy,
                'bbox_scale': args.bbox_scale,
                'trajectory_type': args.trajectory_type,
                'sparse_box_interval': args.sparse_box_interval,
                'downsample_10hz': args.downsample_10hz,
            })
        elif dataset_name.upper() == "ROSE":
            dataset_configs.append({
                'type': 'rose',
                'weight': 1.0,
                'dataset_path': os.getenv("ROSE_DATASET_PATH"),
                'split_name': args.split_name,
                'length': args.num_frames if args.num_frames else 81,
                'target_resolution': (args.width, args.height),
                'mask_strategy': args.mask_strategy,
                'trajectory_type': args.trajectory_type,
                'bbox_scale': args.bbox_scale,
                'use_masked_vace_video': False
            })
        elif dataset_name.upper() == "HUMANM3":
            humanm3_path = os.getenv("HUMANM3_DATASET_PATH")
            if not humanm3_path:
                raise ValueError(
                    "HUMANM3_DATASET_PATH is not set. Please source path_setup.sh first."
                )
            dataset_configs.append({
                'type': 'humanm3',
                'weight': 1.0,
                'dataset_path': humanm3_path,
                'split_name': args.split_name,
                'length': args.num_frames if args.num_frames else 81,
                'target_resolution': (args.width, args.height),
                'dataset_repeat': args.dataset_repeat,
                'bbox_scale': args.bbox_scale,
                'max_samples': 3 if args.debug_mode else None,
            })
        elif dataset_name.upper() in ("PENNACTION", "PENN_ACTION"):
            penn_action_path = os.getenv("PENN_ACTION_DATASET_PATH")
            if not penn_action_path:
                raise ValueError(
                    "PENN_ACTION_DATASET_PATH is not set. Please source path_setup.sh first."
                )
            dataset_configs.append({
                'type': 'penn_action',
                'weight': 1.0,
                'dataset_path': penn_action_path,
                'split_name': args.split_name,
                'length': args.num_frames if args.num_frames else 81,
                'target_resolution': (args.width, args.height),
                'dataset_repeat': args.dataset_repeat,
                'bbox_scale': args.bbox_scale,
                'max_samples': 3 if args.debug_mode else None,
            })
        elif dataset_name.upper() in ("DPW3D", "3DPW"):
            dpw3d_path = os.getenv("DPW3D_DATASET_PATH")
            if not dpw3d_path:
                raise ValueError(
                    "DPW3D_DATASET_PATH is not set. Please source path_setup.sh first."
                )
            dataset_configs.append({
                'type': 'dpw3d',
                'weight': 1.0,
                'dataset_path': dpw3d_path,
                'split_name': args.split_name,
                'length': args.num_frames if args.num_frames else 81,
                'target_resolution': (args.width, args.height),
                'dataset_repeat': args.dataset_repeat,
                'bbox_scale': args.bbox_scale,
                'max_samples': 3 if args.debug_mode else None,
            })
        else:
            raise ValueError(
                f"Invalid dataset name: {dataset_name}. "
                "Supported: DAVIS17, VIPSeg, ROSE, HumanM3, PennAction, DPW3D"
            )
    
    # Create validation dataset if validation split is requested
    validation_dataset = None
    if hasattr(args, 'enable_validation') and args.enable_validation:
        print("[INFO] Creating validation dataset...", flush=True)
        primary_dataset = args.dataset_list.split(",")[0].strip().upper()
        val_length = args.num_frames if args.num_frames else 10
        val_resolution = (args.width, args.height)
        val_kwargs = dict(
            split_name="test",
            length=val_length,
            target_resolution=val_resolution,
            dataset_repeat=1,
            max_samples=1,
            bbox_scale=args.bbox_scale,
        )
        if primary_dataset == "HUMANM3":
            validation_dataset = HumanM3_VACE_Dataset(
                dataset_path=os.getenv("HUMANM3_DATASET_PATH"),
                **val_kwargs,
            )
        elif primary_dataset in ("PENNACTION", "PENN_ACTION"):
            validation_dataset = PennAction_VACE_Dataset(
                dataset_path=os.getenv("PENN_ACTION_DATASET_PATH"),
                **val_kwargs,
            )
        elif primary_dataset in ("DPW3D", "3DPW"):
            validation_dataset = DPW3D_VACE_Dataset(
                dataset_path=os.getenv("DPW3D_DATASET_PATH"),
                **val_kwargs,
            )
        else:
            validation_dataset = DAVIS17_VACE_Dataset(
                dataset_path=os.getenv("DAVIS17_DATASET_PATH"),
                split_name="val",  # Always use "val" split for validation
                length=args.num_frames if args.num_frames else 10,
                target_resolution=(args.width, args.height),
                min_instance_ratio=args.min_instance_ratio,
                min_bbox_ratio=args.min_bbox_ratio,
                use_masked_vace_video=args.use_masked_vace_video,
                dataset_repeat=1,  # No repeat for validation
                max_samples=1,  # Only need one sample for validation video
                mask_foreground=args.mask_foreground,
                mask_strategy=args.mask_strategy,
                frame_selecting_strategy=args.frame_selecting_strategy,
                bbox_scale=args.bbox_scale,
                trajectory_type=args.trajectory_type,
                sparse_box_interval=args.sparse_box_interval,
                downsample_10hz=args.downsample_10hz,
            )
        print(f"[INFO] Validation dataset created with {len(validation_dataset)} samples", flush=True)
    
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        multiview_reference_mode=args.multiview_reference_mode,
        multiview_zero_conv_scale=args.multiview_zero_conv_scale,
        multiview_ipadapter_scale=args.multiview_ipadapter_scale,
        use_multiview_consistency_check=args.use_multiview_consistency_check,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        use_perception_head=args.use_perception_head,
        use_depth_head=args.use_depth_head,
        lambda_latent_segmentation=args.lambda_latent_segmentation,
        lambda_latent_depth=args.lambda_latent_depth,
        lambda_temporal_coherence=args.lambda_temporal_coherence,
        temporal_coherence_method=args.temporal_coherence_method,
        raft_model_path=args.raft_model_path,
        lambda_multiview_dino_viewpoint=args.lambda_multiview_dino_viewpoint,
        multiview_dino_model_path=args.multiview_dino_model_path,
        frame_wise_mask_weighting=args.frame_wise_mask_weighting,
        mask_weight_min_ratio=args.mask_weight_min_ratio,
        mask_weight_max_weight=args.mask_weight_max_weight,
        mask_weight_power=args.mask_weight_power,
        mask_weight_base=args.mask_weight_base,
        train_vace_multiscale_fusion=getattr(args, "train_vace_multiscale_fusion", False),
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(
        MultiDatasetLoader(dataset_configs, sampling_strategy=args.sampling_strategy),
        model, 
        model_logger, 
        args=args,
        validation_dataset=validation_dataset,
        validation_sample_idx=args.validation_sample_idx,
    )
