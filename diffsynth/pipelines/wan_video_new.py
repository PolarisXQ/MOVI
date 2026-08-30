import torch, warnings, glob, os, sys, types, importlib
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union, Dict
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import torch
import torch.nn.functional as F
import random

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner, DINOFeatureLossHelper
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_multiview_ipadapter import MultiviewIPAdapter, MultiviewFeatureBankAdapter
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader
from ..utils.clip_consistency import CLIPConsistencyChecker


_RAFT_MODEL_CACHE = {}
_RAFT_COMPONENTS = None
_RAFT_IMPORT_ERROR = None
_TEMPORAL_COHERENCE_WARNED = set()


class _RAFTArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, key):
        return hasattr(self, key)


def _warn_once(key, message):
    if key not in _TEMPORAL_COHERENCE_WARNED:
        warnings.warn(message)
        _TEMPORAL_COHERENCE_WARNED.add(key)


def _default_raft_model_path():
    model_path = os.environ.get("RAFT_MODEL_PATH") or os.environ.get("TEMPORAL_COHERENCE_RAFT_MODEL_PATH")
    if model_path:
        return model_path
    cache_dir = os.environ.get("VBENCH_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "vbench"))
    return os.path.join(cache_dir, "raft_model", "models", "raft-things.pth")


def _load_raft_components():
    global _RAFT_COMPONENTS, _RAFT_IMPORT_ERROR
    if _RAFT_COMPONENTS is not None:
        return _RAFT_COMPONENTS
    if _RAFT_IMPORT_ERROR is not None:
        raise _RAFT_IMPORT_ERROR

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    vbench_root = os.path.join(repo_root, "VBench")
    if vbench_root not in sys.path:
        sys.path.insert(0, vbench_root)

    try:
        raft_module = importlib.import_module("vbench.third_party.RAFT.core.raft")
        utils_module = importlib.import_module("vbench.third_party.RAFT.core.utils_core.utils")
        RAFT = raft_module.RAFT
        InputPadder = utils_module.InputPadder
    except Exception as exc:
        _RAFT_IMPORT_ERROR = exc
        raise

    _RAFT_COMPONENTS = (RAFT, InputPadder)
    return _RAFT_COMPONENTS


def _get_raft_model(device, model_path=None):
    RAFT, _ = _load_raft_components()
    model_path = model_path or _default_raft_model_path()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"RAFT checkpoint not found: {model_path}. Set RAFT_MODEL_PATH or "
            "TEMPORAL_COHERENCE_RAFT_MODEL_PATH, pass raft_model_path, or use "
            "temporal_coherence_method='simple'."
        )

    cache_key = (os.path.abspath(model_path), str(device))
    if cache_key in _RAFT_MODEL_CACHE:
        return _RAFT_MODEL_CACHE[cache_key]

    args = _RAFTArgs(model=model_path, small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args)
    ckpt = torch.load(model_path, map_location="cpu")
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    _RAFT_MODEL_CACHE[cache_key] = model
    return model


def _prepare_raft_frame(frame):
    if frame.shape[1] != 3:
        raise ValueError(f"RAFT temporal coherence expects RGB frames, got {frame.shape[1]} channels.")
    frame = frame.detach().float()
    if frame.amin() < 0:
        frame = (frame + 1.0) / 2.0
    return frame.clamp(0.0, 1.0) * 255.0


def compute_optical_flow_raft(frame1, frame2, raft_model_path=None, iters=20):
    """
    Compute RAFT optical flow from frame1 to frame2.

    Args:
        frame1: Tensor of shape [B, 3, H, W]
        frame2: Tensor of shape [B, 3, H, W]
        raft_model_path: Optional path to RAFT checkpoint.
        iters: Number of RAFT update iterations.

    Returns:
        flow: Tensor of shape [B, 2, H, W]
    """
    _, InputPadder = _load_raft_components()
    model = _get_raft_model(frame1.device, model_path=raft_model_path)
    image1 = _prepare_raft_frame(frame1)
    image2 = _prepare_raft_frame(frame2)

    with torch.no_grad():
        padder = InputPadder(image1.shape)
        image1_pad, image2_pad = padder.pad(image1, image2)
        _, flow_up = model(image1_pad, image2_pad, iters=iters, test_mode=True)
        flow_up = padder.unpad(flow_up)
    return flow_up.to(device=frame1.device, dtype=frame1.dtype)


def compute_optical_flow_simple(frame1, frame2):
    """
    Compute optical flow between two frames using a simple gradient-based method.
    
    Args:
        frame1: Tensor of shape [B, C, H, W] - first frame
        frame2: Tensor of shape [B, C, H, W] - second frame
    
    Returns:
        flow: Tensor of shape [B, 2, H, W] - optical flow (dx, dy)
    """
    device = frame1.device
    B, C, H, W = frame1.shape
    
    # Convert to grayscale if needed (average across channels)
    if C > 1:
        gray1 = frame1.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        gray2 = frame2.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    else:
        gray1 = frame1
        gray2 = frame2
    
    # Compute spatial gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=gray1.dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=gray1.dtype, device=device).view(1, 1, 3, 3)
    
    # Compute gradients for both frames
    Ix1 = torch.nn.functional.conv2d(gray1, sobel_x, padding=1)
    Iy1 = torch.nn.functional.conv2d(gray1, sobel_y, padding=1)
    Ix2 = torch.nn.functional.conv2d(gray2, sobel_x, padding=1)
    Iy2 = torch.nn.functional.conv2d(gray2, sobel_y, padding=1)
    
    # Temporal gradient
    It = gray2 - gray1
    
    # Lucas-Kanade style flow estimation (simplified)
    # Use a simple approximation: flow proportional to temporal gradient / spatial gradient
    # To avoid division by zero, we use a regularized version
    epsilon = 1e-6
    Ix = (Ix1 + Ix2) / 2
    Iy = (Iy1 + Iy2) / 2
    
    # Simple flow estimation: u = -It * Ix / (Ix^2 + Iy^2 + eps), v = -It * Iy / (Ix^2 + Iy^2 + eps)
    denominator = Ix ** 2 + Iy ** 2 + epsilon
    u = -It * Ix / denominator
    v = -It * Iy / denominator
    
    # Stack to get flow field [B, 2, H, W]
    flow = torch.cat([u, v], dim=1)
    
    return flow


def warp_frame(frame, flow):
    """
    Warp a frame using optical flow.
    
    Args:
        frame: Tensor of shape [B, C, H, W] - frame to warp
        flow: Tensor of shape [B, 2, H, W] - optical flow (dx, dy)
    
    Returns:
        warped_frame: Tensor of shape [B, C, H, W] - warped frame
    """
    device = frame.device
    dtype = frame.dtype  # Use the same dtype as the frame
    B, C, H, W = frame.shape
    
    # Create coordinate grid with the same dtype as frame
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing='ij'
    )
    grid = torch.stack([x_coords, y_coords], dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
    
    # Ensure flow has the same dtype as frame
    flow = flow.to(dtype=dtype)
    
    # Add flow to grid coordinates
    grid = grid + flow
    
    # Normalize to [-1, 1] for grid_sample
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0  # x
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0  # y
    
    # Permute for grid_sample: [B, H, W, 2]
    grid = grid.permute(0, 2, 3, 1)
    
    # Warp frame
    warped_frame = torch.nn.functional.grid_sample(
        frame, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    
    return warped_frame


def compute_temporal_coherence_loss(
    decoded_frames,
    method="raft",
    raft_model_path=None,
    raft_iters=20,
):
    """
    Compute temporal coherence loss using optical flow.
    
    Args:
        decoded_frames: Tensor of shape [B, C, T, H, W] - decoded video frames (typically RGB, C=3)
        method: "raft" for RAFT optical flow, or "simple" for the previous gradient-based flow.
        raft_model_path: Optional path to RAFT checkpoint.
        raft_iters: Number of RAFT update iterations.
    
    Returns:
        temporal_loss: Scalar tensor - temporal coherence loss
    """
    if decoded_frames.shape[2] < 2:  # Need at least 2 frames
        return torch.tensor(0.0, device=decoded_frames.device, dtype=decoded_frames.dtype, requires_grad=True)
    
    method = (method or "raft").lower()
    if method not in ("raft", "simple"):
        raise ValueError(f"Unsupported temporal coherence method: {method}. Use 'raft' or 'simple'.")
    
    B, C, T, H, W = decoded_frames.shape
    if method == "raft" and C != 3:
        _warn_once(
            "raft_non_rgb_temporal_coherence",
            f"RAFT temporal coherence expects RGB frames, got {C} channels. Falling back to simple flow.",
        )
        method = "simple"
    
    # Extract consecutive frame pairs
    frame_pairs = []
    for t in range(T - 1):
        frame1 = decoded_frames[:, :, t, :, :]  # [B, C, H, W]
        frame2 = decoded_frames[:, :, t + 1, :, :]  # [B, C, H, W]
        frame_pairs.append((frame1, frame2))
    
    # Compute temporal coherence loss for each pair
    total_loss = 0.0
    for frame1, frame2 in frame_pairs:
        if method == "raft":
            # RAFT returns forward flow. Estimate frame2 -> frame1 so grid_sample can
            # pull pixels from frame1 into frame2's coordinate system.
            flow = compute_optical_flow_raft(frame2, frame1, raft_model_path=raft_model_path, iters=raft_iters)
        else:
            # Previous simple gradient-based approximation.
            flow = compute_optical_flow_simple(frame1, frame2)
        
        # Warp frame1 using the flow
        warped_frame1 = warp_frame(frame1, flow)
        
        # Compute difference between warped frame1 and frame2
        # This measures how well the flow aligns the frames
        frame_diff = warped_frame1 - frame2
        pair_loss = torch.mean(frame_diff ** 2)
        total_loss = total_loss + pair_loss
    
    # Average over all frame pairs
    temporal_loss = total_loss / len(frame_pairs)
    
    return temporal_loss


def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss for segmentation - focuses on overlap, good for imbalanced classes.
    Particularly useful for binary segmentation (foreground/background).
    
    Args:
        pred: Predicted mask [B, C, T, H, W] or [B, C, H, W]
        target: Target mask [B, C, T, H, W] or [B, C, H, W]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice_loss: Scalar tensor
    """
    # Flatten spatial and temporal dimensions
    pred_flat = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.contiguous().view(target.shape[0], target.shape[1], -1)
    
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def compute_pos_weight_from_foreground_ratio(foreground_ratio):
    """
    Compute pos_weight for BCE loss based on foreground ratio.
    
    Formula: pos_weight = (1 - foreground_ratio) / foreground_ratio
    This balances the loss contribution from positive and negative classes.
    
    Args:
        foreground_ratio: Ratio of foreground pixels (0.0 to 1.0)
                         Can be a single value or tuple (min, max) for range
    
    Returns:
        pos_weight: Recommended pos_weight value
    """
    if isinstance(foreground_ratio, (tuple, list)) and len(foreground_ratio) == 2:
        # For range, use average
        avg_ratio = (foreground_ratio[0] + foreground_ratio[1]) / 2.0
        foreground_ratio = avg_ratio
    
    if foreground_ratio <= 0 or foreground_ratio >= 1:
        return 1.0  # No weighting if ratio is invalid
    
    pos_weight = (1.0 - foreground_ratio) / foreground_ratio
    return pos_weight


def binary_cross_entropy_loss(pred, target, pos_weight=None):
    """
    Binary Cross-Entropy loss for binary segmentation.
    Commonly combined with Dice loss for binary segmentation tasks.
    
    Args:
        pred: Predicted mask [B, C, T, H, W] or [B, C, H, W], values typically in [0, 1]
        target: Target mask [B, C, T, H, W] or [B, C, H, W], values in [0, 1]
        pos_weight: Weight for positive class (useful for imbalanced classes)
                    If None, no weighting is applied.
                    Typical values:
                    - foreground_ratio = 0.1 (10%): pos_weight ≈ 9.0
                    - foreground_ratio = 0.3 (30%): pos_weight ≈ 2.33
                    - foreground_ratio = 0.5 (50%): pos_weight = 1.0
    
    Returns:
        bce_loss: Scalar tensor
    """
    # Flatten all dimensions except batch and channel
    pred_flat = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.contiguous().view(target.shape[0], target.shape[1], -1)
    
    # Clamp predictions to avoid numerical issues
    pred_flat = torch.clamp(pred_flat, min=1e-7, max=1-1e-7)
    
    # Compute BCE loss
    bce = - (target_flat * torch.log(pred_flat) + (1 - target_flat) * torch.log(1 - pred_flat))
    
    # Apply positive weight if provided (for class imbalance)
    if pos_weight is not None:
        bce = bce * (target_flat * (pos_weight - 1) + 1)
    
    return bce.mean()


def boundary_loss(pred, target, kernel_size=3):
    """
    Boundary loss - penalizes errors at edges to make boundaries clearer.
    
    Args:
        pred: Predicted mask [B, C, T, H, W] or [B, C, H, W]
        target: Target mask [B, C, T, H, W] or [B, C, H, W]
        kernel_size: Size of Sobel kernel for edge detection
    
    Returns:
        boundary_loss: Scalar tensor
    """
    # Compute gradients (edges) using Sobel operator
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    
    # Handle temporal dimension
    if pred.ndim == 5:
        B, C, T, H, W = pred.shape
        pred_2d = pred.view(B * C * T, 1, H, W)
        target_2d = target.reshape(B * C * T, 1, H, W)
    else:
        B, C, H, W = pred.shape
        pred_2d = pred.view(B * C, 1, H, W)
        target_2d = target.reshape(B * C, 1, H, W)
    
    # Compute edge maps
    pred_grad_x = F.conv2d(pred_2d, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_2d, sobel_y, padding=1)
    pred_edges = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
    
    target_grad_x = F.conv2d(target_2d, sobel_x, padding=1)
    target_grad_y = F.conv2d(target_2d, sobel_y, padding=1)
    target_edges = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)
    
    # Compute boundary loss
    boundary_diff = (pred_edges - target_edges) ** 2
    return boundary_diff.mean()


def gradient_loss(pred, target):
    """
    Gradient loss for depth prediction - preserves sharp edges.
    
    Args:
        pred: Predicted depth [B, C, T, H, W] or [B, C, H, W]
        target: Target depth [B, C, T, H, W] or [B, C, H, W]
    
    Returns:
        gradient_loss: Scalar tensor
    """
    # Compute gradients in x and y directions
    if pred.ndim == 5:
        B, C, T, H, W = pred.shape
        pred_2d = pred.view(B * C * T, 1, H, W)
        target_2d = target.reshape(B * C * T, 1, H, W)
    else:
        B, C, H, W = pred.shape
        pred_2d = pred.view(B * C, 1, H, W)
        target_2d = target.reshape(B * C, 1, H, W)
    
    # Sobel operators for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    
    # Compute gradients
    pred_grad_x = F.conv2d(pred_2d, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_2d, sobel_y, padding=1)
    
    target_grad_x = F.conv2d(target_2d, sobel_x, padding=1)
    target_grad_y = F.conv2d(target_2d, sobel_y, padding=1)
    
    # L1 loss on gradients
    grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    return grad_loss


def scale_invariant_loss(pred, target):
    """
    Scale-invariant loss for depth - handles scale ambiguity in depth prediction.
    
    Args:
        pred: Predicted depth [B, C, T, H, W] or [B, C, H, W]
        target: Target depth [B, C, T, H, W] or [B, C, H, W]
    
    Returns:
        scale_invariant_loss: Scalar tensor
    """
    # Flatten spatial and temporal dimensions
    pred_flat = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.contiguous().view(target.shape[0], target.shape[1], -1)
    
    # Compute log difference
    log_diff = torch.log(pred_flat + 1e-6) - torch.log(target_flat + 1e-6)
    
    # Scale-invariant loss: variance of log differences
    loss = torch.var(log_diff, dim=2) + 0.5 * torch.mean(log_diff, dim=2) ** 2
    return loss.mean()


def _check_multiview_reference_mode(mode, target_mode):
    """
    Check if multiview_reference_mode contains target_mode.
    Supports both single mode and combination modes like "temporal_concat+ipadapter".
    
    Args:
        mode: multiview_reference_mode string
        target_mode: Target mode to check for (e.g., "temporal_concat", "ipadapter")
    
    Returns:
        bool: True if target_mode is in mode
    """
    if mode is None:
        return False
    if isinstance(mode, str):
        return mode == target_mode or target_mode in mode.split("+")
    return False


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.multiview_ipadapter: MultiviewIPAdapter = None
        self.multiview_feature_bank_adapter: MultiviewFeatureBankAdapter = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "multiview_ipadapter", "multiview_feature_bank_adapter")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "multiview_ipadapter", "multiview_feature_bank_adapter")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_MultiviewConsistencyCheck(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_MultiviewIPAdapter(),
            WanVideoPostUnit_AnimateVideoSplit(),
            WanVideoPostUnit_AnimatePoseLatents(),
            WanVideoPostUnit_AnimateFacePixelValues(),
            WanVideoPostUnit_AnimateInpaint(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video
    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)

    def update_vace_in_dim(self, new_vace_in_dim, vace_model=None):
        """
        Update VACE model's input dimension to accommodate multiview features.
        
        Args:
            new_vace_in_dim: New input dimension (e.g., 160 for multiview: 96 original + 64 multiview)
            vace_model: Optional specific VACE model to update. If None, updates both vace and vace2 if available.
        """
        vace_models = []
        if vace_model is not None:
            vace_models = [vace_model]
        else:
            if self.vace is not None:
                vace_models.append(self.vace)
            if self.vace2 is not None:
                vace_models.append(self.vace2)
        
        for vace in vace_models:
            if vace.vace_in_dim == new_vace_in_dim:
                print(f"VACE model already has vace_in_dim={new_vace_in_dim}, skipping update.")
                continue
            
            old_vace_in_dim = vace.vace_in_dim
            old_patch_embedding = vace.vace_patch_embedding
            
            # Unwrap the module if it's wrapped (e.g., AutoWrappedModule)
            if hasattr(old_patch_embedding, 'module'):
                unwrapped_patch_embedding = old_patch_embedding.module
            else:
                unwrapped_patch_embedding = old_patch_embedding
            
            # Get device and dtype - prefer self.device, fallback to module's device
            try:
                device = self.device
            except:
                device = next(unwrapped_patch_embedding.parameters()).device
            dtype = next(unwrapped_patch_embedding.parameters()).dtype
            
            # Create new Conv3d layer with updated input dimension
            new_patch_embedding = torch.nn.Conv3d(
                new_vace_in_dim,
                unwrapped_patch_embedding.out_channels,
                kernel_size=unwrapped_patch_embedding.kernel_size,
                stride=unwrapped_patch_embedding.stride,
                padding=unwrapped_patch_embedding.padding,
                dilation=unwrapped_patch_embedding.dilation,
                groups=unwrapped_patch_embedding.groups,
                bias=unwrapped_patch_embedding.bias is not None
            )
            
            # Move to device and set dtype
            new_patch_embedding = new_patch_embedding.to(device=device, dtype=dtype)
            
            # Copy weights from old layer for the first old_vace_in_dim channels
            with torch.no_grad():
                # Ensure we're copying from the unwrapped module's weight
                old_weight = unwrapped_patch_embedding.weight
                new_patch_embedding.weight[:, :old_vace_in_dim, :, :, :].copy_(old_weight)
                # Initialize new channels (remaining channels) with small random values
                if new_vace_in_dim > old_vace_in_dim:
                    torch.nn.init.kaiming_uniform_(
                        new_patch_embedding.weight[:, old_vace_in_dim:, :, :, :],
                        a=2.23606797749979  # sqrt(5)
                    )
                    # Scale down the initialization to match the magnitude of existing weights
                    scale_factor = old_weight.std().item() / new_patch_embedding.weight[:, old_vace_in_dim:, :, :, :].std().item()
                    new_patch_embedding.weight[:, old_vace_in_dim:, :, :, :] *= scale_factor
                
                if unwrapped_patch_embedding.bias is not None:
                    new_patch_embedding.bias.copy_(unwrapped_patch_embedding.bias)
            
            # Replace the old layer
            vace.vace_patch_embedding = new_patch_embedding
            vace.vace_in_dim = new_vace_in_dim
            print(f"Updated VACE model vace_in_dim from {old_vace_in_dim} to {new_vace_in_dim}")
    
    def _print_vace_trainable_status(self, stage=""):
        """Print VACE model trainable status."""
        if self.vace is None:
            print(f"[INFO] VACE trainable status ({stage}): VACE model is None", flush=True)
            return
        
        # Check if model is in training mode
        is_training_mode = self.vace.training
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.vace.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.vace.parameters())
        frozen_params = total_params - trainable_params
        
        # Check if any parameters require grad
        has_trainable_params = trainable_params > 0
        
        # Print status
        status_str = f"[INFO] VACE trainable status ({stage}):"
        status_str += f"\n  - Model training mode: {is_training_mode}"
        status_str += f"\n  - Has trainable parameters: {has_trainable_params}"
        status_str += f"\n  - Trainable parameters: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params if total_params > 0 else 0:.2f}%)"
        status_str += f"\n  - Frozen parameters: {frozen_params:,}"
        
        # Check VACE type
        vace_type = type(self.vace).__name__
        status_str += f"\n  - VACE model type: {vace_type}"
             
        print(status_str, flush=True)

    def _get_dino_view_control_helper(self, model_name=None):
        resolved_model_name = model_name or os.getenv("DINO_VIEW_CONTROL_MODEL_PATH") or "facebook/dinov2-base"
        cached_model_name = getattr(self, "_dino_view_control_model_name", None)
        cached_helper = getattr(self, "_dino_view_control_helper", None)
        if cached_helper is None or cached_model_name != resolved_model_name:
            self._dino_view_control_helper = DINOFeatureLossHelper(
                model_name=resolved_model_name,
                device=self.device,
            )
            self._dino_view_control_model_name = resolved_model_name
        return self._dino_view_control_helper

    def _decode_first_generated_frame(self, predicted_clean_latents, ref_len):
        if self.vae is None or predicted_clean_latents is None:
            return None
        if predicted_clean_latents.ndim != 5:
            return None
        if predicted_clean_latents.shape[2] <= ref_len:
            return None

        first_frame_latent = predicted_clean_latents[:, :, ref_len:ref_len + 1, :, :]
        vae_was_training = self.vae.training
        self.vae.eval()
        try:
            decoded_first_frame = self.vae.single_decode(first_frame_latent, self.device)
        finally:
            if vae_was_training:
                self.vae.train()
            else:
                self.vae.eval()
        return decoded_first_frame[:, :, 0, :, :]

    def _compute_multiview_dino_view_control_loss(self, predicted_clean_latents, ref_len, inputs):
        multiview_reference_image = inputs.get("multiview_reference_image")
        if multiview_reference_image is None:
            return None
        if isinstance(multiview_reference_image, list):
            if len(multiview_reference_image) == 0:
                return None
            reference_image = multiview_reference_image[0]
            if isinstance(reference_image, list):
                if len(reference_image) == 0:
                    return None
                reference_image = reference_image[0]
        else:
            reference_image = multiview_reference_image

        generated_first_frame = self._decode_first_generated_frame(predicted_clean_latents, ref_len)
        if generated_first_frame is None:
            return None

        foreground_mask_or_box = inputs.get("vace_video_mask")
        # if foreground_mask_or_box is None:
        #     foreground_mask_or_box = inputs.get("trajectory_maps")

        dino_helper = self._get_dino_view_control_helper(inputs.get("multiview_dino_model_path"))
        return dino_helper.compute_feature_loss(
            predicted_image=generated_first_frame,
            reference_image=reference_image,
            foreground_mask_or_box=foreground_mask_or_box,
            device=self.device,
        )
        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        model_output = self.model_fn(**inputs, timestep=timestep)
        
        # Handle segmentation/depth head output
        if isinstance(model_output, tuple):
            if len(model_output) == 3:
                noise_pred, mask_pred, depth_pred = model_output # noise_pred.shape torch.Size([1, 16, 25, 60, 104])
            else:
                raise ValueError(f"Model output should have 3 elements, but got {len(model_output)}")
        else:
            noise_pred = model_output
            mask_pred = None
            depth_pred = None

        predicted_clean_latents = None
        if "input_latents" in inputs and "noise" in inputs and inputs["input_latents"].ndim == 5:
            # For flow matching scheduler, predicted clean sample = noise - predicted target.
            predicted_clean_latents = inputs["noise"] - noise_pred

        ref_len = 0
        multiview_reference_mode = inputs.get("multiview_reference_mode", "temporal_concat")
        if _check_multiview_reference_mode(multiview_reference_mode, "temporal_concat"):
            if "multiview_reference_image" in inputs and inputs["multiview_reference_image"] is not None:
                ref_len = len(inputs["multiview_reference_image"])
            elif "vace_reference_image" in inputs and inputs["vace_reference_image"] is not None:
                ref_len = len(inputs["vace_reference_image"]) if isinstance(inputs["vace_reference_image"], list) else 1
        elif "vace_reference_image" in inputs and inputs["vace_reference_image"] is not None:
            ref_len = len(inputs["vace_reference_image"]) if isinstance(inputs["vace_reference_image"], list) else 1

        if ref_len == 0:
            print("[WARNING] ref_len is 0.", flush=True)
        
        # print("noise_pred shape:", noise_pred.shape, flush=True) # noise_pred shape: torch.Size([1, 16, 25, 60, 104]) noise_pred shape: torch.Size([1, 16, 88, 16, 8])
        # print("training_target shape:", training_target.shape, flush=True) # training_target shape: torch.Size([1, 16, 25, 60, 104]) training_target shape: torch.Size([1, 16, 88, 16, 8])
        # Diffusion loss with optional frame-wise mask weighting
        frame_wise_weighting = inputs.get("frame_wise_mask_weighting", False)
        if frame_wise_weighting and "frame_mask_ratios" not in inputs:
            print("[WARNING] frame_mask_ratios not found in inputs, using standard loss")
            frame_wise_weighting = False
        
        # Check if we should use frame-wise weighting
        if frame_wise_weighting and "frame_mask_ratios" in inputs:
            # print("frame_wise_weighting", frame_wise_weighting)
            # print("frame_mask_ratios", inputs["frame_mask_ratios"])
            # Frame-wise weighting: calculate loss per frame and apply frame-specific weights
            frame_mask_ratios = inputs["frame_mask_ratios"]
            if isinstance(frame_mask_ratios, list):
                frame_mask_ratios = torch.tensor(frame_mask_ratios)
            # Get parameters
            mask_weight_min_ratio = inputs.get("mask_weight_min_ratio", 0.01)
            mask_weight_max_weight = inputs.get("mask_weight_max_weight", 10.0)
            mask_weight_power = inputs.get("mask_weight_power", 0.5)
            mask_weight_base = inputs.get("mask_weight_base", 0.5)
            
            # Compute weights for each frame
            frame_mask_ratios_clamped = torch.clamp(frame_mask_ratios, min=mask_weight_min_ratio)
            frame_weights = mask_weight_base * (1.0 / frame_mask_ratios_clamped) ** mask_weight_power
            frame_weights = torch.clamp(frame_weights, max=mask_weight_max_weight)
            
            # noise_pred and training_target shape: [B, C, T, H, W]
            B, C, T, H, W = noise_pred.shape
            
            # Ensure frame_weights matches the temporal dimension
            if frame_weights.shape[0] != T-ref_len: # mismatch because frame weights is not encoded by vae, should be 81 and 25

                # Interpolate weights to match T
                frame_weights = torch.nn.functional.interpolate(
                    frame_weights.unsqueeze(0).unsqueeze(0),  # [1, 1, T_orig]
                    size=T-ref_len,
                    mode='linear',
                    align_corners=False
                ).squeeze()
            
            # Reshape for per-frame loss calculation
            if 'num_views' in inputs and inputs['num_views'] > 1:
                # ref image is at the beginning of each view, so we need to split the noise_pred and training_target into num_views parts first,
                # then skip the first ref_len frames for each view, and then concatenate the rest of the frames
                T = noise_pred.shape[2]
                print("T:", T, flush=True)
                T_sv = T // inputs['num_views']
                print("T_sv:", T_sv, flush=True)
                noise_pred_flat = []
                training_target_flat = []
                for view_idx in range(inputs['num_views']):
                    noise_pred_flat.append(noise_pred[:,:,view_idx*T_sv+ref_len:(view_idx+1)*T_sv,:,:].permute(0, 2, 1, 3, 4).contiguous().view(B, ref_len, -1))
                    training_target_flat.append(training_target[:,:,view_idx*T_sv+ref_len:(view_idx+1)*T_sv,:,:].permute(0, 2, 1, 3, 4).contiguous().view(B, ref_len, -1))
                    print("noise_pred_flat per view shape:", noise_pred_flat[-1].shape, flush=True)
                    print("training_target_flat per view shape:", training_target_flat[-1].shape, flush=True)
                noise_pred_flat = torch.cat(noise_pred_flat, dim=1)
                training_target_flat = torch.cat(training_target_flat, dim=1)
                print("noise_pred_flat shape:", noise_pred_flat.shape, flush=True)
                print("training_target_flat shape:", training_target_flat.shape, flush=True)
            else:
                noise_pred_flat = noise_pred[:,:,ref_len:,:,:].permute(0, 2, 1, 3, 4).contiguous().view(B, T-ref_len, -1)  # [B, T, C*H*W]
                training_target_flat = training_target[:,:,ref_len:,:,:].permute(0, 2, 1, 3, 4).contiguous().view(B, T-ref_len, -1)  # [B, T, C*H*W]
            # Per-frame MSE loss: [B, T]
            per_frame_loss = torch.nn.functional.mse_loss(
                noise_pred_flat.float(), 
                training_target_flat.float(), 
                reduction='none'
            ).mean(dim=2)  # Average over spatial+channel dimensions: [B, T]
            
            # Apply scheduler weight
            per_frame_loss = per_frame_loss * self.scheduler.training_weight(timestep)
            
            # Apply frame-wise weights: [B, T] * [T] -> [B, T]
            frame_weights = frame_weights.to(device=per_frame_loss.device, dtype=per_frame_loss.dtype)
            weighted_per_frame_loss = per_frame_loss * frame_weights.unsqueeze(0)  # [B, T]
            
            # Average over batch and temporal dimensions
            loss = weighted_per_frame_loss.mean()
        else:
            # Standard loss calculation
            if 'num_views' in inputs and inputs['num_views'] > 1:
                # print("noise_pred shape:", noise_pred.shape, flush=True)
                # print("training_target shape:", training_target.shape, flush=True)
                # noise_pred shape: torch.Size([1, 16, 88, 16, 8])
                # training_target shape: torch.Size([1, 16, 88, 16, 8])
                noise_pred_sv = []
                training_target_sv = []
                T = noise_pred.shape[2]
                T_sv = T // inputs['num_views']
                ref_len = ref_len // inputs['num_views']
                for view_idx in range(inputs['num_views']):
                    noise_pred_sv.append(noise_pred[:,:,view_idx*T_sv+ref_len:(view_idx+1)*T_sv,:,:])
                    training_target_sv.append(training_target[:,:,view_idx*T_sv+ref_len:(view_idx+1)*T_sv,:,:])
                    # print("noise_pred_sv per view shape:", noise_pred_sv[-1].shape, flush=True)
                    # print("training_target_sv per view shape:", training_target_sv[-1].shape, flush=True)
                    # noise_pred_sv per view shape: torch.Size([1, 16, 21, 16, 8])
                    # training_target_sv per view shape: torch.Size([1, 16, 21, 16, 8])
                noise_pred = torch.cat(noise_pred_sv, dim=2)
                training_target = torch.cat(training_target_sv, dim=2)
                # print("noise_pred_sv shape:", noise_pred_sv.shape, flush=True)
                # print("training_target_sv shape:", training_target_sv.shape, flush=True)
                # noise_pred_sv shape: torch.Size([1, 16, 84, 16, 8])
                # training_target_sv shape: torch.Size([1, 16, 84, 16, 8])
                loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
            else:
                loss = torch.nn.functional.mse_loss(noise_pred[:,:,ref_len:,:,:].float(), training_target[:,:,ref_len:,:,:].float())
            loss = loss * self.scheduler.training_weight(timestep)
        
        # print("diffusion_loss", loss)
        
        # Latent segmentation loss
        if mask_pred is not None and "target_mask_latent" in inputs and len(inputs["target_mask_latent"]) > 0:
            lambda_latent_segmentation = inputs.get("lambda_latent_segmentation", 1.0)
            target_mask_latent = inputs["target_mask_latent"]
            
            # Loss function configuration
            use_mse_loss = inputs.get("segmentation_use_mse", True)
            use_bce_loss = inputs.get("segmentation_use_bce", False)  # Good for binary segmentation
            use_dice_loss = inputs.get("segmentation_use_dice", False)  # Good for binary segmentation
            use_boundary_loss = inputs.get("segmentation_use_boundary", False)
            lambda_mse = inputs.get("lambda_segmentation_mse", 1.0)
            lambda_bce = inputs.get("lambda_segmentation_bce", 0.4)
            lambda_dice = inputs.get("lambda_segmentation_dice", 0.4)
            lambda_boundary = inputs.get("lambda_segmentation_boundary", 0.5)
            # For class imbalance in binary segmentation
            # Can provide either pos_weight directly or foreground_ratio (will compute pos_weight)
            segmentation_pos_weight = inputs.get("segmentation_pos_weight", None)
            segmentation_foreground_ratio = inputs.get("segmentation_foreground_ratio", [0.1, 0.5])
            
            # Auto-compute pos_weight from foreground_ratio if provided
            if segmentation_pos_weight is None and segmentation_foreground_ratio is not None:
                segmentation_pos_weight = compute_pos_weight_from_foreground_ratio(segmentation_foreground_ratio)
            
            # Ensure shapes match
            t = target_mask_latent.shape[1]
            # print("mask_pred", mask_pred.shape, "target_mask_latent", target_mask_latent.shape)
            mask_pred = mask_pred[:,-t:,:,:,:] # remove the first reference frames
            # print("mask_pred.shape", mask_pred.shape, "target_mask_latent.shape", target_mask_latent.shape)
            if mask_pred.shape != target_mask_latent.shape:
                print("[WARNING] mask_pred.shape != target_mask_latent.shape, interpolate mask_pred to match target_mask_latent shape")
                # Interpolate mask_pred to match target_mask_latent shape
                mask_pred = torch.nn.functional.interpolate(
                    mask_pred.flatten(0, 1),  # [B*T, C, H, W]
                    size=(target_mask_latent.shape[3], target_mask_latent.shape[4]),
                    mode='bilinear',
                    align_corners=False
                )
                mask_pred = mask_pred.view(
                    target_mask_latent.shape[0],
                    target_mask_latent.shape[1],
                    target_mask_latent.shape[2],
                    target_mask_latent.shape[3],
                    target_mask_latent.shape[4]
                )
            
            # Normalize predictions and targets to [0, 1] range for BCE/Dice (if needed)
            # Both mask_pred and target_mask_latent are in latent space, so we need proper normalization
            target_min = target_mask_latent.min()
            target_max = target_mask_latent.max()
            target_range = target_max - target_min
            
            mask_pred_min = mask_pred.min()
            mask_pred_max = mask_pred.max()
            mask_pred_range = mask_pred_max - mask_pred_min
            
            # print("target mask min max:", target_min, target_max)
            # print("mask_pred min max:", mask_pred_min, mask_pred_max)
            
            # For BCE and Dice loss, normalize both to [0, 1] using min-max normalization
            if use_bce_loss or use_dice_loss:
                # Normalize target_mask_latent to [0, 1] using its own range
                if target_range > 1e-6:  # Avoid division by zero
                    target_normalized = (target_mask_latent - target_min) / target_range
                else:
                    target_normalized = torch.zeros_like(target_mask_latent)
                
                # Normalize mask_pred to [0, 1] using its own range
                if mask_pred_range > 1e-6:
                    mask_pred_normalized = (mask_pred - mask_pred_min) / mask_pred_range
                    # Clamp to [0, 1] to ensure valid probability range
                    mask_pred_normalized = torch.clamp(mask_pred_normalized, 0, 1)
                else:
                    # If range is too small, use sigmoid as fallback
                    mask_pred_normalized = torch.sigmoid(mask_pred)
            # Compute segmentation loss components
            segmentation_loss = 0.0
            if use_mse_loss:
                mse_loss = torch.nn.functional.mse_loss(
                    mask_pred.float(), 
                    target_mask_latent.float()
                )
                segmentation_loss = segmentation_loss + lambda_mse * mse_loss
                # print("mse_loss", mse_loss)
            
            if use_bce_loss:
                bce_loss_val = binary_cross_entropy_loss(
                    mask_pred_normalized.float(), 
                    target_normalized.float(),
                    pos_weight=segmentation_pos_weight
                )
                segmentation_loss = segmentation_loss + lambda_bce * bce_loss_val
                # print("bce_loss_val", bce_loss_val)
            
            if use_dice_loss:
                # Dice loss works on normalized values too
                dice_loss_val = dice_loss(mask_pred_normalized.float(), target_normalized.float())
                segmentation_loss = segmentation_loss + lambda_dice * dice_loss_val
                # print("dice_loss_val", dice_loss_val)
            
            if use_boundary_loss:
                boundary_loss_val = boundary_loss(mask_pred.float(), target_mask_latent.float())
                segmentation_loss = segmentation_loss + lambda_boundary * boundary_loss_val
                # print("boundary_loss_val", boundary_loss_val)
            # If no loss is enabled, fall back to BCE + Dice (standard for binary segmentation)
            if segmentation_loss == 0.0:
                bce_loss_val = binary_cross_entropy_loss(
                    mask_pred_normalized.float(), 
                    target_normalized.float(),
                    pos_weight=segmentation_pos_weight
                )
                dice_loss_val = dice_loss(mask_pred_normalized.float(), target_normalized.float())
                segmentation_loss = 0.5 * bce_loss_val + 0.5 * dice_loss_val
                # print("bce_loss_val", bce_loss_val)
                # print("dice_loss_val", dice_loss_val)
            loss = loss + lambda_latent_segmentation * segmentation_loss
            # print("segmentation_loss", segmentation_loss)
            
            # Decode and save first frame for visualization
            save_visualization = False # random.random() < 0.1 # 10% chance to save visualization
            if self.vae is not None and save_visualization:
                try:
                    # Extract first frame: [B, T, C, H, W] -> [B, 1, C, H, W]
                    target_mask_first_frame = target_mask_latent[:, 0:1, :, :, :]  # [B, 1, C, H, W]
                    mask_pred_first_frame = mask_pred[:, 0:1, :, :, :]  # [B, 1, C, H, W]
                    
                    # Rearrange to [B, C, T, H, W] for VAE decode
                    target_mask_first_frame = rearrange(target_mask_first_frame, "b t c h w -> b c t h w")
                    mask_pred_first_frame = rearrange(mask_pred_first_frame, "b t c h w -> b c t h w")
                    
                    # Channels now match VAE (16 channels), no padding needed
                    # Decode using VAE
                    vae_was_training = self.vae.training
                    # print("vae_was_training", vae_was_training)
                    self.vae.eval()  # Use eval mode for inference
                    
                    with torch.no_grad():
                        target_mask_decoded = self.vae.single_decode(target_mask_first_frame, self.device)
                        mask_pred_decoded = self.vae.single_decode(mask_pred_first_frame, self.device)
                
                    # Restore original training mode
                    if vae_was_training:
                        self.vae.train()
                    else:
                        self.vae.eval()
                    
                    # Convert to images and save
                    # target_mask_decoded: [B, C, T, H, W] -> take first frame
                    target_mask_img = target_mask_decoded[0, :, 0, :, :].cpu().float()  # [C, H, W]
                    mask_pred_img = mask_pred_decoded[0, :, 0, :, :].cpu().float()  # [C, H, W]
                    
                    # Convert to PIL Image: [C, H, W] -> [H, W, C] -> PIL
                    target_mask_img_np = target_mask_img.permute(1, 2, 0).numpy()
                    target_mask_img_np = ((target_mask_img_np / 2 + 0.5).clip(0, 1) * 255).astype("uint8")
                    target_mask_pil = Image.fromarray(target_mask_img_np)
                    
                    mask_pred_img_np = mask_pred_img.permute(1, 2, 0).numpy()
                    mask_pred_img_np = ((mask_pred_img_np / 2 + 0.5).clip(0, 1) * 255).astype("uint8")
                    mask_pred_pil = Image.fromarray(mask_pred_img_np)
                    
                    # Save images
                    os.makedirs("visualization", exist_ok=True)
                    target_mask_pil.save(f"visualization/target_mask_frame0.png")
                    mask_pred_pil.save(f"visualization/mask_pred_frame0.png")
                    
                except Exception as e:
                    print(f"Warning: Failed to decode and save mask visualization: {e}")
        
        # Latent depth loss
        if depth_pred is not None and "target_depth_latent" in inputs:
            lambda_latent_depth = inputs.get("lambda_latent_depth", 1.0)
            target_depth_latent = inputs["target_depth_latent"]
            
            # Loss function configuration
            use_mse_loss = inputs.get("depth_use_mse", True)
            use_l1_loss = inputs.get("depth_use_l1", False)
            use_gradient_loss = inputs.get("depth_use_gradient", False)
            use_scale_invariant_loss = inputs.get("depth_use_scale_invariant", False)
            lambda_mse = inputs.get("lambda_depth_mse", 1.0)
            lambda_l1 = inputs.get("lambda_depth_l1", 0.5)
            lambda_gradient = inputs.get("lambda_depth_gradient", 0.5)
            lambda_scale_invariant = inputs.get("lambda_depth_scale_invariant", 0.1)
            
            # Ensure shapes match
            t = target_depth_latent.shape[1]
            depth_pred = depth_pred[:,-t:,:,:,:] # remove the first reference frame
            if depth_pred.shape != target_depth_latent.shape:
                print("[WARNING] depth_pred.shape != target_depth_latent.shape, interpolate depth_pred to match target_depth_latent shape")
                # Interpolate depth_pred to match target_depth_latent shape
                depth_pred = torch.nn.functional.interpolate(
                    depth_pred.flatten(0, 1),  # [B*T, C, H, W]
                    size=(target_depth_latent.shape[3], target_depth_latent.shape[4]),
                    mode='bilinear',
                    align_corners=False
                )
                depth_pred = depth_pred.view(
                    target_depth_latent.shape[0],
                    target_depth_latent.shape[1],
                    target_depth_latent.shape[2],
                    target_depth_latent.shape[3],
                    target_depth_latent.shape[4]
                )
            
            # Compute depth loss components
            depth_loss = 0.0
            if use_mse_loss:
                mse_loss = torch.nn.functional.mse_loss(
                    depth_pred.float(), 
                    target_depth_latent.float()
                )
                depth_loss = depth_loss + lambda_mse * mse_loss
                # print("mse_loss", mse_loss)
            if use_l1_loss:
                l1_loss = torch.nn.functional.l1_loss(
                    depth_pred.float(), 
                    target_depth_latent.float()
                )
                depth_loss = depth_loss + lambda_l1 * l1_loss
                # print("l1_loss", l1_loss)
            if use_gradient_loss:
                grad_loss = gradient_loss(depth_pred.float(), target_depth_latent.float())
                depth_loss = depth_loss + lambda_gradient * grad_loss
                # print("grad_loss", grad_loss)
            if use_scale_invariant_loss:
                scale_inv_loss = scale_invariant_loss(depth_pred.float(), target_depth_latent.float())
                depth_loss = depth_loss + lambda_scale_invariant * scale_inv_loss
                # print("scale_inv_loss", scale_inv_loss)
            # If no loss is enabled, fall back to L1 (better than MSE for depth)
            if depth_loss == 0.0:
                depth_loss = torch.nn.functional.l1_loss(
                    depth_pred.float(), 
                    target_depth_latent.float()
                )
                # print("l1_loss", l1_loss)
            loss = loss + lambda_latent_depth * depth_loss
            # print("depth_loss", depth_loss)

            # Decode and save first frame for visualization
            save_visualization = False # random.random() < 0.1 # 10% chance to save visualization
            if self.vae is not None and save_visualization:
                try:
                    # Extract first frame: [B, T, C, H, W] -> [B, 1, C, H, W]
                    target_depth_first_frame = target_depth_latent[:, 0:1, :, :, :]  # [B, 1, C, H, W]
                    depth_pred_first_frame = depth_pred[:, 0:1, :, :, :]  # [B, 1, C, H, W]
                    
                    # Rearrange to [B, C, T, H, W] for VAE decode
                    target_depth_first_frame = rearrange(target_depth_first_frame, "b t c h w -> b c t h w")
                    depth_pred_first_frame = rearrange(depth_pred_first_frame, "b t c h w -> b c t h w")
                    
                    # Decode using VAE
                    vae_was_training = self.vae.training
                    self.vae.eval()  # Use eval mode for inference
                    
                    with torch.no_grad():
                        target_depth_decoded = self.vae.single_decode(target_depth_first_frame, self.device)
                        depth_pred_decoded = self.vae.single_decode(depth_pred_first_frame, self.device)
                
                    # Restore original training mode
                    if vae_was_training:
                        self.vae.train()
                    else:
                        self.vae.eval()
                    
                    # Convert to images and save
                    # target_depth_decoded: [B, C, T, H, W] -> take first frame
                    target_depth_img = target_depth_decoded[0, :, 0, :, :].cpu().float()  # [C, H, W]
                    depth_pred_img = depth_pred_decoded[0, :, 0, :, :].cpu().float()  # [C, H, W]
                    
                    # Convert to PIL Image: [C, H, W] -> [H, W, C] -> PIL
                    target_depth_img_np = target_depth_img.permute(1, 2, 0).numpy()
                    target_depth_img_np = ((target_depth_img_np / 2 + 0.5).clip(0, 1) * 255).astype("uint8")
                    target_depth_pil = Image.fromarray(target_depth_img_np)
                    
                    depth_pred_img_np = depth_pred_img.permute(1, 2, 0).numpy()
                    depth_pred_img_np = ((depth_pred_img_np / 2 + 0.5).clip(0, 1) * 255).astype("uint8")
                    depth_pred_pil = Image.fromarray(depth_pred_img_np)
                    
                    # Save images
                    os.makedirs("visualization", exist_ok=True)
                    target_depth_pil.save(f"visualization/target_depth_frame0.png")
                    depth_pred_pil.save(f"visualization/depth_pred_frame0.png")
                    
                except Exception as e:
                    print(f"Warning: Failed to decode and save depth visualization: {e}")

        lambda_multiview_dino_viewpoint = inputs.get("lambda_multiview_dino_viewpoint", 0.0)
        if lambda_multiview_dino_viewpoint > 0 and predicted_clean_latents is not None:
            multiview_dino_view_loss = self._compute_multiview_dino_view_control_loss(
                predicted_clean_latents=predicted_clean_latents,
                ref_len=ref_len,
                inputs=inputs,
            )
            if multiview_dino_view_loss is not None:
                print("multiview_dino_view_loss", multiview_dino_view_loss)
                loss = loss + lambda_multiview_dino_viewpoint * multiview_dino_view_loss
        
        # Temporal coherence loss: Minimize appearance jitter between frames using optical flow
        if "input_latents" in inputs and "noise" in inputs:
            lambda_temporal = inputs.get("lambda_temporal_coherence", 0.0)
            if lambda_temporal > 0:
                temporal_coherence_method = inputs.get("temporal_coherence_method", "raft")
                raft_model_path = inputs.get("raft_model_path", None)
                raft_iters = inputs.get("raft_iters", 20)
                if predicted_clean_latents is not None:
                    predicted_clean_latents = predicted_clean_latents[:,:,ref_len:,:,:] # remove the first reference frames
                    # Decode latents to frames for temporal coherence computation
                    compute_optical_flow_in_latent = False
                    if self.vae is not None and not compute_optical_flow_in_latent:
                        # Ensure VAE is in training mode and on the correct device
                        tiled = inputs.get("tiled", False)
                        tile_size = inputs.get("tile_size", None)
                        tile_stride = inputs.get("tile_stride", None)
                        # Use no_grad to avoid storing activations for backprop, reducing memory usage
                        # Temporal coherence loss doesn't need gradients through VAE decode
                        with torch.no_grad():
                            decoded_frames = self.vae.decode(predicted_clean_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                        # Detach to ensure no gradients flow through VAE decode
                        decoded_frames = decoded_frames.detach()
                        save_visualization = False # random.random() < 0.1 # 10% chance to save visualization
                        if save_visualization:
                            # Extract first frame: [B, C, T, H, W] -> [C, H, W] (first batch, first frame)
                            decoded_frames_first_frame = decoded_frames[0, :, 0, :, :].cpu().float()  # [C, H, W]
                            # Convert to [H, W, C] for PIL Image
                            decoded_frames_first_frame = decoded_frames_first_frame.permute(1, 2, 0)  # [H, W, C]
                            decoded_frames_first_frame = (decoded_frames_first_frame / 2 + 0.5).clamp(0, 1)
                            decoded_frames_first_frame = (decoded_frames_first_frame * 255).to(torch.uint8)
                            decoded_frames_first_frame = Image.fromarray(decoded_frames_first_frame.numpy())
                            decoded_frames_first_frame.save("visualization/decoded_frames_first_frame.png")
                        temporal_loss = compute_temporal_coherence_loss(
                            decoded_frames,
                            method=temporal_coherence_method,
                            raft_model_path=raft_model_path,
                            raft_iters=raft_iters,
                        )
                        loss = loss + lambda_temporal * temporal_loss
                        # print("temporal_loss", temporal_loss)
                    else:
                        # Fallback to latent space if VAE is not available
                        temporal_loss = compute_temporal_coherence_loss(
                            predicted_clean_latents,
                            method=temporal_coherence_method,
                            raft_model_path=raft_model_path,
                            raft_iters=raft_iters,
                        )
                        loss = loss + lambda_temporal * temporal_loss
        # print("loss", loss)
        
        return loss


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            # Handle both string path and list of paths (for split models)
            if isinstance(model_config.path, list):
                # Check all files in the list exist
                for path in model_config.path:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Model file {path} not found.")
            else:
                # Single file path
                if not os.path.exists(model_config.path):
                    raise FileNotFoundError(f"Model file {model_config.path} not found.")
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        vace = model_manager.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Union[Image.Image, list[Image.Image]]] = None,
        multiview_reference_image: Optional[list[Image.Image]] = None,
        multiview_reference_mode: Optional[str] = "temporal_concat",
        multiview_reference_weight: Optional[float] = None,
        multiview_zero_conv_scale: Optional[float] = 1.0,
        multiview_ipadapter_scale: Optional[float] = 1.0,
        normal_maps: Optional[list[Image.Image]] = None,
        position_maps: Optional[list[Image.Image]] = None,
        # ptv3_feature_maps: Optional[list[Image.Image]] = None,
        vace_scale: Optional[float] = 1.0,
        # Trajectory ControlNet
        trajectory_maps: Optional[Union[list[Image.Image], torch.Tensor]] = None,
        trajectory_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        # Save timesteps
        save_timesteps_callback: Optional[callable] = None,
        # Multiview parameters
        num_views: Optional[int] = None,
        view_indices: Optional[torch.Tensor] = None,
        use_multiview_consistency_check: Optional[bool] = True,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Detect multiview input
        is_multiview = False
        detected_num_views = 1
        if vace_video is not None and isinstance(vace_video, list) and len(vace_video) > 0:
            if isinstance(vace_video[0], list):
                # Multiview format: List[List[Image.Image]]
                is_multiview = True
                detected_num_views = len(vace_video)
                if num_views is None:
                    num_views = detected_num_views
        # Set default multiview parameters
        if num_views is None:
            num_views = 1
        
        # Process multiview inputs if detected
        if is_multiview and num_views > 1:
            # Multiview processing: process each view separately, then rearrange
            # Encode prompt once for all views (optimization)
            self.load_models_to_device(["prompter"])
            prompt_emb_posi = self.prompter.encode_prompt(prompt, positive=True, device=self.device)
            prompt_emb_nega = self.prompter.encode_prompt(negative_prompt, positive=False, device=self.device)
            
            # Process each view
            input_video_list = []
            vace_video_list = []
            vace_video_mask_list = []
            vace_reference_image_list = []
            multiview_reference_image_list = []
            trajectory_maps_list = []
            target_mask_latent_list = []
            noise_list = [] # list of tensor
            latents_list = []
            # input_latents_list = []
            vace_context_list = []
            animate_pose_video_list = []
            animate_face_video_list = []
            animate_inpaint_video_list = []
            animate_mask_video_list = []

            for view_idx in range(num_views):
                # Prepare view-specific inputs
                inputs_posi = {}
                inputs_nega = {}
                inputs_shared = {
                    "input_image": input_image,
                    "end_image": end_image,
                    "input_video": input_video, 
                    "denoising_strength": denoising_strength,
                    "control_video": control_video, 
                    "reference_image": reference_image,
                    "camera_control_direction": camera_control_direction, 
                    "camera_control_speed": camera_control_speed, 
                    "camera_control_origin": camera_control_origin,
                    "vace_video": vace_video[view_idx], 
                    "vace_video_mask": vace_video_mask[view_idx], 
                    "vace_reference_image": vace_reference_image[view_idx] if vace_reference_image is not None else None,
                    "vace_scale": vace_scale,
                    "multiview_reference_image": multiview_reference_image,
                    "multiview_reference_mode": multiview_reference_mode,
                    "multiview_zero_conv_scale": multiview_zero_conv_scale,
                    "multiview_ipadapter_scale": multiview_ipadapter_scale,
                    "normal_maps": normal_maps, 
                    "position_maps": position_maps,
                    "trajectory_maps": trajectory_maps[view_idx] if trajectory_maps is not None else None,
                    "trajectory_scale": trajectory_scale, 
                    "seed": seed, "rand_device": rand_device,
                    "height": height, "width": width,
                    "num_frames": num_frames,
                    "cfg_scale": cfg_scale, 
                    "cfg_merge": cfg_merge,
                    "sigma_shift": sigma_shift,
                    "motion_bucket_id": motion_bucket_id,
                    "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
                    "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
                    "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
                    "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
                    # Multiview parameters
                    "num_views": num_views,
                    "view_idx": view_idx,
                }
            
                # Process through pipeline units (skip prompt embedder since we pre-encoded)
                for unit in self.units:
                    if unit.__class__.__name__ == "WanVideoUnit_PromptEmbedder":
                        continue  # Skip prompt encoding, already done
                    inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                        unit, self, inputs_shared, inputs_posi, inputs_nega
                    )

                # if view_idx == 0:
                #     print("------------after unit--------------------")
                #     print(f"inputs_shared keys: {inputs_shared.keys()}", flush=True)
                #     print(f"inputs_posi keys: {inputs_posi.keys()}", flush=True)
                #     print(f"inputs_nega keys: {inputs_nega.keys()}", flush=True)
                #     print("--------------------------------")
                #     for key in inputs_shared.keys():
                #         if isinstance(inputs_shared[key], torch.Tensor):
                #             print(f"{key}:", inputs_shared[key].shape)
                #         elif isinstance(inputs_shared[key], list):
                #             print(f"{key}:", len(inputs_shared[key]))
                #         else:
                #             print(f"{key}:", type(inputs_shared[key]))
                #     for key in inputs_posi.keys():
                #         if isinstance(inputs_posi[key], torch.Tensor):
                #             print(f"{key}:", inputs_posi[key].shape)
                #         elif isinstance(inputs_posi[key], list):
                #             print(f"{key}:", len(inputs_posi[key]))
                #         else:
                #             print(f"{key}:", type(inputs_posi[key]))
                #     for key in inputs_nega.keys():
                #         if isinstance(inputs_nega[key], torch.Tensor):
                #             print(f"{key}:", inputs_nega[key].shape)
                #         elif isinstance(inputs_nega[key], list):
                #             print(f"{key}:", len(inputs_nega[key]))
                #         else:
                #             print(f"{key}:", type(inputs_nega[key]))

                        # ------------after unit--------------------
                        # inputs_shared keys: dict_keys(['input_image', 'end_image', 'input_video', 'denoising_strength', 'control_video', 'reference_image', 'camera_control_direction', 'camera_control_speed', 'camera_control_origin', 'vace_video', 'vace_video_mask', 'vace_reference_image', 'vace_scale', 'multiview_reference_image', 'normal_maps', 'position_maps', 'trajectory_maps', 'trajectory_scale', 'seed', 'rand_device', 'height', 'width', 'num_frames', 'cfg_scale', 'cfg_merge', 'sigma_shift', 'motion_bucket_id', 'tiled', 'tile_size', 'tile_stride', 'sliding_window_size', 'sliding_window_stride', 'input_audio', 'audio_sample_rate', 's2v_pose_video', 'audio_embeds', 's2v_pose_latents', 'motion_video', 'animate_pose_video', 'animate_face_video', 'animate_inpaint_video', 'animate_mask_video', 'num_views', 'view_idx', 'noise', 'latents', 'vace_context'])
                        # inputs_posi keys: dict_keys([])
                        # inputs_nega keys: dict_keys([])
                        # --------------------------------
                        # input_image: <class 'NoneType'>
                        # end_image: <class 'NoneType'>
                        # input_video: <class 'NoneType'>
                        # denoising_strength: <class 'float'>
                        # control_video: <class 'NoneType'>
                        # reference_image: <class 'NoneType'>
                        # camera_control_direction: <class 'NoneType'>
                        # camera_control_speed: <class 'float'>
                        # camera_control_origin: <class 'tuple'>
                        # vace_video: 81
                        # vace_video_mask: 81
                        # vace_reference_image: <class 'NoneType'>
                        # vace_scale: <class 'float'>
                        # multiview_reference_image: 4
                        # normal_maps: <class 'NoneType'>
                        # position_maps: <class 'NoneType'>
                        # trajectory_maps: <class 'NoneType'>
                        # trajectory_scale: <class 'NoneType'>
                        # seed: <class 'int'>
                        # rand_device: <class 'str'>
                        # height: <class 'int'>
                        # width: <class 'int'>
                        # num_frames: <class 'int'>
                        # cfg_scale: <class 'float'>
                        # cfg_merge: <class 'bool'>
                        # sigma_shift: <class 'float'>
                        # motion_bucket_id: <class 'NoneType'>
                        # tiled: <class 'bool'>
                        # tile_size: <class 'tuple'>
                        # tile_stride: <class 'tuple'>
                        # sliding_window_size: <class 'NoneType'>
                        # sliding_window_stride: <class 'NoneType'>
                        # input_audio: <class 'NoneType'>
                        # audio_sample_rate: <class 'int'>
                        # s2v_pose_video: <class 'NoneType'>
                        # audio_embeds: <class 'NoneType'>
                        # s2v_pose_latents: <class 'NoneType'>
                        # motion_video: <class 'NoneType'>
                        # animate_pose_video: <class 'NoneType'>
                        # animate_face_video: <class 'NoneType'>
                        # animate_inpaint_video: <class 'NoneType'>
                        # animate_mask_video: <class 'NoneType'>
                        # num_views: <class 'int'>
                        # view_idx: <class 'int'>
                        # noise: torch.Size([1, 16, 25, 8, 8])
                        # latents: torch.Size([1, 16, 25, 8, 8])
                        # vace_context: torch.Size([1, 96, 25, 8, 8])

                
                input_video_list.append(inputs_shared["input_video"])
                vace_video_list.append(inputs_shared["vace_video"])
                vace_video_mask_list.append(inputs_shared["vace_video_mask"])
                vace_reference_image_list.append(inputs_shared["vace_reference_image"])
                multiview_reference_image_list.append(inputs_shared["multiview_reference_image"])
                trajectory_maps_list.append(inputs_shared["trajectory_maps"])
                noise_list.append(inputs_shared["noise"])
                latents_list.append(inputs_shared["latents"])
                # input_latents_list.append(inputs_shared["input_latents"])
                vace_context_list.append(inputs_shared["vace_context"])
                animate_pose_video_list.append(inputs_shared["animate_pose_video"])
                animate_face_video_list.append(inputs_shared["animate_face_video"])
                animate_inpaint_video_list.append(inputs_shared["animate_inpaint_video"])
                animate_mask_video_list.append(inputs_shared["animate_mask_video"])

            input_video = input_video_list
            vace_video = vace_video_list
            vace_video_mask = vace_video_mask_list
            vace_reference_image = vace_reference_image_list
            multiview_reference_image = multiview_reference_image_list
            trajectory_maps = trajectory_maps_list
            noise = torch.stack(noise_list, dim=0) # [n_views, B, C, T, H, W]
            latents = torch.stack(latents_list, dim=0) # [n_views, B, C, T, H, W]
            # input_latents = torch.stack(input_latents_list, dim=0) # [n_views, B, C, T, H, W]
            vace_context = torch.stack(vace_context_list, dim=0) # [n_views, B, vace_dim, T, H, W]
            animate_pose_video = animate_pose_video_list
            animate_face_video = animate_face_video_list
            animate_inpaint_video = animate_inpaint_video_list
            animate_mask_video = animate_mask_video_list
            
            # to [B, C, V*T, H, W] - concatenate views along temporal dimension
            noise = rearrange(noise, "n b c t h w -> b c (n t) h w")
            latents = rearrange(latents, "n b c t h w -> b c (n t) h w")
            # input_latents = rearrange(input_latents, "n b c t h w -> b c (n t) h w")
            vace_context = rearrange(vace_context, "n b vace_dim t h w -> b vace_dim (n t) h w")  # Fixed: concatenate views in temporal dim
            
            inputs_shared_all_views = {
                "input_image": input_image,
                "end_image": end_image,
                "input_video": input_video, 
                "denoising_strength": denoising_strength,
                "control_video": control_video, 
                "reference_image": reference_image,
                "camera_control_direction": camera_control_direction, 
                "camera_control_speed": camera_control_speed, 
                "camera_control_origin": camera_control_origin,
                "vace_video": vace_video, 
                "vace_video_mask": vace_video_mask, 
                "vace_reference_image": vace_reference_image, 
                "multiview_reference_image": multiview_reference_image,
                "vace_scale": vace_scale,
                "multiview_reference_mode": multiview_reference_mode,
                "multiview_reference_weight": multiview_reference_weight,
                "multiview_zero_conv_scale": multiview_zero_conv_scale,
                "multiview_ipadapter_scale": multiview_ipadapter_scale,
                "normal_maps": normal_maps, 
                "position_maps": position_maps,
                "trajectory_maps": trajectory_maps, 
                "trajectory_scale": trajectory_scale, 
                "seed": seed, "rand_device": rand_device,
                "height": height, "width": width,
                "num_frames": num_frames,
                "cfg_scale": cfg_scale, 
                "cfg_merge": cfg_merge,
                "sigma_shift": sigma_shift,
                "motion_bucket_id": motion_bucket_id,
                "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
                "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
                "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
                "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
                # Multiview parameters
                "num_views": num_views,
                "noise": noise,
                "latents": latents,
                # "input_latents": input_latents,
                "vace_context": vace_context,
            }

            input_posi_all_views = {
                "prompt": prompt,
                "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
                "context": prompt_emb_posi,
            }
            input_nega_all_views = {
                "negative_prompt": negative_prompt,
                "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
                "context": prompt_emb_nega,
            }
            inputs_posi = input_posi_all_views
            inputs_nega = input_nega_all_views
            inputs_shared = inputs_shared_all_views

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

            # inputs_shared keys: dict_keys(['input_image', 'end_image', 'input_video', 'denoising_strength', 'control_video', 'reference_image', 'camera_control_direction', 'camera_control_speed', 'camera_control_origin', 'vace_video', 'vace_video_mask', 'vace_reference_image', 'vace_scale', 'multiview_reference_image', 'normal_maps', 'position_maps', 'trajectory_maps', 'trajectory_scale', 'seed', 'rand_device', 'height', 'width', 'num_frames', 'cfg_scale', 'cfg_merge', 'sigma_shift', 'motion_bucket_id', 'tiled', 'tile_size', 'tile_stride', 'sliding_window_size', 'sliding_window_stride', 'input_audio', 'audio_sample_rate', 's2v_pose_video', 'audio_embeds', 's2v_pose_latents', 'motion_video', 'animate_pose_video', 'animate_face_video', 'animate_inpaint_video', 'animate_mask_video', 'num_views', 'noise', 'latents', 'vace_context'])
            # inputs_posi keys: dict_keys(['prompt', 'tea_cache_l1_thresh', 'tea_cache_model_id', 'num_inference_steps', 'context'])
            # inputs_nega keys: dict_keys(['negative_prompt', 'tea_cache_l1_thresh', 'tea_cache_model_id', 'num_inference_steps', 'context'])
            # --------------------------------
            # input_image: <class 'NoneType'>
            # end_image: <class 'NoneType'>
            # input_video: 7
            # denoising_strength: <class 'float'>
            # control_video: <class 'NoneType'>
            # reference_image: <class 'NoneType'>
            # camera_control_direction: <class 'NoneType'>
            # camera_control_speed: <class 'float'>
            # camera_control_origin: <class 'tuple'>
            # vace_video: 7
            # vace_video_mask: 7
            # vace_reference_image: <class 'NoneType'>
            # vace_scale: <class 'float'>
            # multiview_reference_image: 4
            # normal_maps: <class 'NoneType'>
            # position_maps: <class 'NoneType'>
            # trajectory_maps: 7
            # trajectory_scale: <class 'NoneType'>
            # seed: <class 'int'>
            # rand_device: <class 'str'>
            # height: <class 'int'>
            # width: <class 'int'>
            # num_frames: <class 'int'>
            # cfg_scale: <class 'float'>
            # cfg_merge: <class 'bool'>
            # sigma_shift: <class 'float'>
            # motion_bucket_id: <class 'NoneType'>
            # tiled: <class 'bool'>
            # tile_size: <class 'tuple'>
            # tile_stride: <class 'tuple'>
            # sliding_window_size: <class 'NoneType'>
            # sliding_window_stride: <class 'NoneType'>
            # input_audio: <class 'NoneType'>
            # audio_sample_rate: <class 'int'>
            # s2v_pose_video: <class 'NoneType'>
            # audio_embeds: <class 'NoneType'>
            # s2v_pose_latents: <class 'NoneType'>
            # motion_video: <class 'NoneType'>
            # animate_pose_video: 7
            # animate_face_video: 7
            # animate_inpaint_video: 7
            # animate_mask_video: 7
            # num_views: <class 'int'>
            # noise: torch.Size([1, 16, 175, 8, 8])
            # latents: torch.Size([1, 16, 175, 8, 8])
            # vace_context: torch.Size([1, 96, 175, 8, 8])
            # prompt: <class 'str'>
            # tea_cache_l1_thresh: <class 'NoneType'>
            # tea_cache_model_id: <class 'str'>
            # num_inference_steps: <class 'int'>
            # context: torch.Size([1, 512, 4096])
            # negative_prompt: <class 'str'>
            # tea_cache_l1_thresh: <class 'NoneType'>
            # tea_cache_model_id: <class 'str'>
            # num_inference_steps: <class 'int'>
            # context: torch.Size([1, 512, 4096])
            # 0%|                                                                       | 0/5 [00:00<?, ?it/s]
            # x shape: torch.Size([1, 16, 175, 8, 8])
        else:
            # Standard single-view processing
            inputs_posi = {
                "prompt": prompt,
                "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
            }
            inputs_nega = {
                "negative_prompt": negative_prompt,
                "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
            }
            inputs_shared = {
                "input_image": input_image,
                "end_image": end_image,
                "input_video": input_video, "denoising_strength": denoising_strength,
                "control_video": control_video, "reference_image": reference_image,
                "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
                "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
                "multiview_reference_image": multiview_reference_image,
                "multiview_reference_mode": multiview_reference_mode,
                "multiview_reference_weight": multiview_reference_weight,
                "multiview_zero_conv_scale": multiview_zero_conv_scale,
                "multiview_ipadapter_scale": multiview_ipadapter_scale,
                "normal_maps": normal_maps, "position_maps": position_maps,
                "trajectory_maps": trajectory_maps, "trajectory_scale": trajectory_scale, 
                # "ptv3_feature_maps": ptv3_feature_maps,
                "seed": seed, "rand_device": rand_device,
                "height": height, "width": width, "num_frames": num_frames,
                "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
                "sigma_shift": sigma_shift,
                "motion_bucket_id": motion_bucket_id,
                "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
                "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
                "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
                "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
                "num_views": num_views,
                # "prompt": prompt,
            }

            if use_multiview_consistency_check:
                if not hasattr(self, "_clip_consistency_checker") or self._clip_consistency_checker is None:
                    # Prefer CLIP_CONSISTENCY_MODEL_PATH; else local release layout.
                    clip_model_path = os.getenv("CLIP_CONSISTENCY_MODEL_PATH") or os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "models",
                        "LongCLIP-GmP-ViT-L-14",
                    )
                    self._clip_consistency_checker = CLIPConsistencyChecker(
                        model_name=clip_model_path,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                inputs_shared["prompt"] = prompt
            for unit in self.units:
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
            # delete "prompt" from inputs_shared if it exists
            if "prompt" in inputs_shared:
                del inputs_shared["prompt"]
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

            ####### Standard single-view processing #######
            # inputs_shared keys: dict_keys(['input_image', 'end_image', 'input_video', 'denoising_strength', 'control_video', 'reference_image', 'camera_control_direction', 'camera_control_speed', 'camera_control_origin', 'vace_video', 'vace_video_mask', 'vace_reference_image', 'vace_scale', 'multiview_reference_image', 'normal_maps', 'position_maps', 'trajectory_maps', 'trajectory_scale', 'seed', 'rand_device', 'height', 'width', 'num_frames', 'cfg_scale', 'cfg_merge', 'sigma_shift', 'motion_bucket_id', 'tiled', 'tile_size', 'tile_stride', 'sliding_window_size', 'sliding_window_stride', 'input_audio', 'audio_sample_rate', 's2v_pose_video', 'audio_embeds', 's2v_pose_latents', 'motion_video', 'animate_pose_video', 'animate_face_video', 'animate_inpaint_video', 'animate_mask_video', 'num_views', 'noise
            # inputs_posi keys: dict_keys(['prompt', 'tea_cache_l1_thresh', 'tea_cache_model_id', 'num_inference_steps', 'context'])
            # inputs_nega keys: dict_keys(['negative_prompt', 'tea_cache_l1_thresh', 'tea_cache_model_id', 'num_inference_steps', 'context'])
            # --------------------------------
            # input_image: <class 'NoneType'>
            # end_image: <class 'NoneType'>
            # input_video: <class 'NoneType'>
            # denoising_strength: <class 'float'>
            # control_video: <class 'NoneType'>
            # reference_image: <class 'NoneType'>
            # camera_control_direction: <class 'NoneType'>
            # camera_control_speed: <class 'float'>
            # camera_control_origin: <class 'tuple'>
            # vace_video: 81
            # vace_video_mask: 81
            # vace_reference_image: <class 'NoneType'>
            # vace_scale: <class 'float'>
            # multiview_reference_image: 4
            # normal_maps: <class 'NoneType'>
            # position_maps: <class 'NoneType'>
            # trajectory_maps: <class 'NoneType'>
            # trajectory_scale: <class 'float'>
            # seed: <class 'int'>
            # rand_device: <class 'str'>
            # height: <class 'int'>
            # width: <class 'int'>
            # num_frames: <class 'int'>
            # cfg_scale: <class 'float'>
            # cfg_merge: <class 'bool'>
            # sigma_shift: <class 'float'>
            # motion_bucket_id: <class 'NoneType'>
            # tiled: <class 'bool'>
            # tile_size: <class 'tuple'>
            # tile_stride: <class 'tuple'>
            # sliding_window_size: <class 'NoneType'>
            # sliding_window_stride: <class 'NoneType'>
            # input_audio: <class 'NoneType'>
            # audio_sample_rate: <class 'int'>
            # s2v_pose_video: <class 'NoneType'>
            # audio_embeds: <class 'NoneType'>
            # s2v_pose_latents: <class 'NoneType'>
            # motion_video: <class 'NoneType'>
            # animate_pose_video: <class 'NoneType'>
            # animate_face_video: <class 'NoneType'>
            # animate_inpaint_video: <class 'NoneType'>
            # animate_mask_video: <class 'NoneType'>
            # num_views: <class 'int'>
            # noise: torch.Size([1, 16, 25, 60, 104])
            # latents: torch.Size([1, 16, 25, 60, 104])
            # vace_context: torch.Size([1, 96, 25, 60, 104])
            # prompt: <class 'str'>
            # tea_cache_l1_thresh: <class 'NoneType'>
            # tea_cache_model_id: <class 'str'>
            # num_inference_steps: <class 'int'>
            # context: torch.Size([1, 512, 4096])
            # negative_prompt: <class 'str'>
            # tea_cache_l1_thresh: <class 'NoneType'>
            # tea_cache_model_id: <class 'str'>
            # num_inference_steps: <class 'int'>
            # context: torch.Size([1, 512, 4096])
            # 0%|                                                                       | 0/5 [00:00<?, ?it/s]
            # x shape: torch.Size([1, 16, 25, 60, 104])

            # Initialize f for reference image handling
            f = 0
            if _check_multiview_reference_mode(multiview_reference_mode, "temporal_concat") and multiview_reference_image is not None:
                # print("[DEBUG] using temporal_concat mode")
                f = len(multiview_reference_image) if isinstance(multiview_reference_image, list) else 1
            elif vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
                if vace_reference_image is not None and isinstance(vace_reference_image, list):
                    f = len(vace_reference_image)
                else:
                    f = 1

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            model_output_posi = self.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=timestep,
            )
            # Handle segmentation/depth head output
            if isinstance(model_output_posi, tuple):
                if len(model_output_posi) == 3:
                    noise_pred_posi, mask_pred_posi, depth_pred_posi = model_output_posi
                else:
                    raise ValueError(f"Model output should have 3 elements, but got {len(model_output_posi)}")
            else:
                noise_pred_posi = model_output_posi
                mask_pred_posi = None
                depth_pred_posi = None
            
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                    if mask_pred_posi is not None:
                        mask_pred_posi, mask_pred_nega = mask_pred_posi.chunk(2, dim=0)
                    else:
                        mask_pred_nega = None
                    if depth_pred_posi is not None:
                        depth_pred_posi, depth_pred_nega = depth_pred_posi.chunk(2, dim=0)
                    else:
                        depth_pred_nega = None
                else:
                    model_output_nega = self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=timestep,
                    )
                    if isinstance(model_output_nega, tuple):
                        if len(model_output_nega) == 3:
                            noise_pred_nega, mask_pred_nega, depth_pred_nega = model_output_nega
                        else:
                            raise ValueError(f"Model output should have 3 elements, but got {len(model_output_nega)}")
                    else:
                        noise_pred_nega = model_output_nega
                        mask_pred_nega = None
                        depth_pred_nega = None
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                if mask_pred_posi is not None and mask_pred_nega is not None:
                    mask_pred = mask_pred_nega + cfg_scale * (mask_pred_posi - mask_pred_nega)
                else:
                    mask_pred = mask_pred_posi if mask_pred_posi is not None else mask_pred_nega
                if depth_pred_posi is not None and depth_pred_nega is not None:
                    depth_pred = depth_pred_nega + cfg_scale * (depth_pred_posi - depth_pred_nega)
                else:
                    depth_pred = depth_pred_posi if depth_pred_posi is not None else depth_pred_nega
            else:
                noise_pred = noise_pred_posi
                mask_pred = mask_pred_posi
                depth_pred = depth_pred_posi

            # Store final mask and depth predictions for later visualization
            # Only store from the last timestep to save memory
            if progress_id == len(self.scheduler.timesteps) - 1:
                self._last_mask_pred = mask_pred.clone() if mask_pred is not None else None
                self._last_depth_pred = depth_pred.clone() if depth_pred is not None else None
            else:
                # Clear previous predictions to save memory
                if hasattr(self, '_last_mask_pred'):
                    del self._last_mask_pred
                if hasattr(self, '_last_depth_pred'):
                    del self._last_depth_pred

            if _check_multiview_reference_mode(multiview_reference_mode, "temporal_concat") and multiview_reference_image is not None:
                f = len(multiview_reference_image) if isinstance(multiview_reference_image, list) else 1
            elif vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
                if vace_reference_image is not None and isinstance(vace_reference_image, list):
                    f = len(vace_reference_image)
                else:
                    f = 1
                    
            #########################################################################################
            # Scheduler process
            # 1. add high frequency mask to the latents
            # 2. add noise to the latents
            # 3. add the latents to the scheduler
            # 4. decode the latents
            # 5. save the video
            # 6. return the video
            # --------------------------------------------------------------
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
            
            # Save timestep video if callback is provided
            if save_timesteps_callback is not None and progress_id % 5 == 0:
                # Prepare latents for decoding (handle vace_reference_image case)
                
                if video is not None:
                    video = self.vae_output_to_video(video)
                    save_timesteps_callback(progress_id, video)
                    continue
                
                decode_latents = inputs_shared["latents"].clone()
                # Reconstruct xt-1 by subtracting the noise predicted at the current step, then inspect it.
                    # decode_latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], decode_latents)
                decode_latents, _ = self.scheduler.step_x0(noise_pred, self.scheduler.timesteps[progress_id], decode_latents)
                if _check_multiview_reference_mode(multiview_reference_mode, "temporal_concat") and multiview_reference_image is not None:
                    f = len(multiview_reference_image) if isinstance(multiview_reference_image, list) else 1
                    decode_latents = decode_latents[:, :, f:]
                elif vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
                    if vace_reference_image is not None and isinstance(vace_reference_image, list):
                        f = len(vace_reference_image)
                    else:
                        f = 1
                    decode_latents = decode_latents[:, :, f:]
                
                # Decode current timestep
                self.load_models_to_device(['vae'])
                timestep_video = self.vae.decode(decode_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                timestep_video = self.vae_output_to_video(timestep_video)
                self.load_models_to_device(self.in_iteration_models)
                
                # Call callback with timestep video
                save_timesteps_callback(progress_id, timestep_video)
        
        # print("inputs_shared[latents].shape:", inputs_shared["latents"].shape)
            
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        
        # Handle reference frames: skip them when decoding
        # In multiview case, f is per-view, so after concatenation we need to skip f frames at the start
        # and then f frames every num_frames_per_view frames
        if is_multiview and num_views > 1:
            # For multiview, reference frames are at the beginning of each view's latents
            # After concatenation: [ref0, ref1, ..., ref_f-1, frame0_view0, ..., frame0_view1, ..., ref0_view1, ...]
            # We need to extract only the actual video frames (skip reference frames)
            B, C, total_t, H, W = inputs_shared["latents"].shape
            t_per_view = total_t // num_views
            
            # Extract video frames (skip reference frames) for each view
            decode_latents_list = []
            for view_idx in range(num_views):
                view_latents = inputs_shared["latents"][:, :, view_idx*t_per_view+f:(view_idx+1)*t_per_view]
                decode_latents_list.append(view_latents)
            
            # VAE decode expects a list of tensors, one per item to decode
            # Decode each view separately and collect results
            decoded_videos = []
            for view_latents in decode_latents_list:
                view_video = self.vae.decode(view_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                decoded_videos.append(view_video)
            
            # Stack decoded videos: [V, B, C, T', H', W']
            video_tensor = torch.stack(decoded_videos, dim=0)  # [V, B, C, T', H', W']
            # Rearrange to [B, C, V*T', H', W'] for vae_output_to_video
            video_tensor = rearrange(video_tensor, "v b c t h w -> b c (v t) h w")  # [B, C, V*T', H', W']
            video = self.vae_output_to_video(video_tensor)
            
            # Split video list into views
            num_frames_per_view_decoded = len(video) // num_views
            multiview_video = []
            for view_idx in range(num_views):
                start_idx = view_idx * num_frames_per_view_decoded
                end_idx = (view_idx + 1) * num_frames_per_view_decoded
                view_video = video[start_idx:end_idx]
                multiview_video.append(view_video)
            
            return multiview_video
        else:
            # print("inputs_shared[latents].shape:", inputs_shared["latents"].shape)
            
            decode_latents = inputs_shared["latents"][:, :, f:]  # Standard single-view case
            # print("decode_latents shape:", decode_latents.shape)
            video = self.vae.decode(decode_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video = self.vae_output_to_video(video)
            self.load_models_to_device([])
            return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image", "multiview_reference_image", "multiview_reference_mode"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image, multiview_reference_image, multiview_reference_mode):
        if multiview_reference_mode is None:
            multiview_reference_mode = "temporal_concat"
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        if _check_multiview_reference_mode(multiview_reference_mode, "temporal_concat") and multiview_reference_image is not None:
            f = len(multiview_reference_image) if isinstance(multiview_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        if _check_multiview_reference_mode(multiview_reference_mode, "temporal_concat") and multiview_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image", "multiview_reference_image", "multiview_reference_mode", "multiview_reference_weight"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image, multiview_reference_image, multiview_reference_mode, multiview_reference_weight):
        if multiview_reference_mode is None:
            multiview_reference_mode = "temporal_concat"
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if _check_multiview_reference_mode(multiview_reference_mode, "temporal_concat") and multiview_reference_image is not None:
            if not isinstance(multiview_reference_image, list):
                multiview_reference_image = [multiview_reference_image]
            multiview_reference_latents_list = []
            for ref_img in multiview_reference_image:
                ref_tensor = pipe.preprocess_video([ref_img]).squeeze(0)  # (C, 1, H, W)
                ref_latent = pipe.vae.encode([ref_tensor], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                multiview_reference_latents_list.append(ref_latent.squeeze(0))  # (C, 1, H, W)
            multiview_reference_latents = torch.concat(multiview_reference_latents_list, dim=1)  # (C, T, H, W)
            multiview_reference_latents = multiview_reference_latents.unsqueeze(0)
            # Cross-modal consistency: down-weight reference when text–image consistency is low
            w = multiview_reference_weight if multiview_reference_weight is not None else 1.0
            multiview_reference_latents = multiview_reference_latents * w
            input_latents = torch.concat([multiview_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}


class WanVideoUnit_MultiviewConsistencyCheck(PipelineUnit):
    """
    Cross-modal consistency check: CLIP text–image similarity for multiview reference.
    When 3D-rendered reference images are corrupted, consistency is low; we output
    a weight in [0, 1] to down-weight the reference path and rely more on text.
    """
    def __init__(self):
        super().__init__(input_params=("prompt", "multiview_reference_image", "multiview_reference_weight"))

    def process(
        self,
        pipe: WanVideoPipeline,
        prompt,
        multiview_reference_image,
        multiview_reference_weight,
    ):
        if multiview_reference_weight is not None:
            return {}
        if not prompt or not multiview_reference_image or len(multiview_reference_image) == 0:
            print("no prompt or no multiview_reference_image, return 1.0 for multiview_reference_weight", flush=True)
            return {"multiview_reference_weight": 1.0}
        if not hasattr(pipe, "_clip_consistency_checker") or pipe._clip_consistency_checker is None:
            # Prefer CLIP_CONSISTENCY_MODEL_PATH; else local release layout.
            clip_model_path = os.getenv("CLIP_CONSISTENCY_MODEL_PATH") or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models",
                "LongCLIP-GmP-ViT-L-14",
            )
            pipe._clip_consistency_checker = CLIPConsistencyChecker(
                model_name=clip_model_path,
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )
        score = pipe._clip_consistency_checker.compute_consistency(
            prompt, multiview_reference_image, device=pipe.device, reduce="mean"
        )
        # print(f"Using CLIP consistency checker, multiview_reference_weight: {score}", flush=True)
        return {"multiview_reference_weight": float(score)}


class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", 
            "multiview_reference_image", 
            "normal_maps", 
            "position_maps", 
            # "ptv3_feature_maps", 
            "vace_scale",
            "height", "width", "num_frames", "tiled", "tile_size", "tile_stride", "multiview_reference_mode", "multiview_reference_weight", "multiview_zero_conv_scale", "multiview_ipadapter_scale"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, 
        multiview_reference_image, 
        normal_maps, position_maps, 
        # ptv3_feature_maps, 
        vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride,
        multiview_reference_mode,
        multiview_reference_weight,
        multiview_zero_conv_scale,
        multiview_ipadapter_scale
    ):
        if multiview_reference_mode is None:
            multiview_reference_mode = "temporal_concat"
        if multiview_zero_conv_scale is None:
            multiview_zero_conv_scale = 1.0
        ref_weight = multiview_reference_weight if multiview_reference_weight is not None else 1.0
        # if vace_reference_image is not None:
        #     print("vace_reference_image added")
        # else:
        #     print("vace_reference_image not added")
        # if multiview_reference_image is not None:
        #     print("multiview_reference_image added")
        # else:
        #     print("multiview_reference_image not added")
        # if normal_maps is not None:
        #     print("normal_maps added")
        # else:
        #     print("normal_maps not added")
        # if position_maps is not None:
        #     print("position_maps added")
        # else:
        #     print("position_maps not added")
        # Support temporal_concat mode (can be combined with ipadapter)
        if _check_multiview_reference_mode(multiview_reference_mode, "temporal_concat") and multiview_reference_image is not None:
            vace_reference_image = multiview_reference_image
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None or multiview_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                # Use the VAE's expected input channels
                vace_video_channels = getattr(pipe.vae, 'in_channels', 3)
                vace_video = torch.zeros((1, vace_video_channels, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')

            f = 0
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                # Process each frame separately to avoid temporal downsampling
                vace_reference_latents_list = []
                for ref_img in vace_reference_image:
                    # Preprocess each image individually as a single-frame video
                    ref_tensor = pipe.preprocess_video([ref_img]).squeeze(0)  # (C, 1, H, W)
                    # Encode each frame separately - returns (1, C, T, H, W) where T=1
                    ref_latent = pipe.vae.encode([ref_tensor], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                    # ref_latent shape: (1, C, 1, H, W) - squeeze batch dim and keep temporal
                    vace_reference_latents_list.append(ref_latent.squeeze(0))  # (C, 1, H, W)
                
                # Concatenate all reference latents along temporal dimension
                # Each is (C, 1, H, W), so concat along dim=1 (temporal) gives (C, T, H, W)
                vace_reference_latents = torch.concat(vace_reference_latents_list, dim=1)  # (C, T, H, W)
                f = vace_reference_latents.shape[1]  # Number of reference frames
                
                # Add batch dimension: (1, C, T, H, W)
                vace_reference_latents = vace_reference_latents.unsqueeze(0)
                
                # Concatenate with zeros for the second channel (inactive/reactive split)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                # Split along temporal dimension for concatenation with video latents
                vace_reference_latents = [vace_reference_latents[:, :, j:j+1] for j in range(f)]
                # Cross-modal consistency: down-weight reference when text–image consistency is low
                vace_reference_latents = [x * ref_weight for x in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            # print("vace_context.shape after adding vace_reference_image: ", vace_context.shape)

            # Multiview reference images: ControlNet-style zero layer injection (per-layer, model-side)
            # temporal_concat+zero_conv: both direct concat and control branch for richer conditioning
            vace_control_context = None
            if _check_multiview_reference_mode(multiview_reference_mode, "zero_conv") and multiview_reference_image is not None and len(multiview_reference_image) > 0:
                if not isinstance(multiview_reference_image, list):
                    multiview_reference_image = [multiview_reference_image]
                multiview_latents = []
                for view_img in multiview_reference_image:
                    view_tensor = pipe.preprocess_video([view_img]).squeeze(0)
                    view_latent = pipe.vae.encode([view_tensor], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                    multiview_latents.append(view_latent.squeeze(0))  # (C, 1, H, W)
                # (C*V, 1, H, W) -> (1, C*V, 1, H, W) -> repeat along T
                multiview_latents = torch.concat(multiview_latents, dim=0).unsqueeze(0)
                vace_control_context = multiview_latents.repeat(1, 1, vace_context.shape[2], 1, 1)
            
            # Encode and concatenate normal maps (similar to Hunyuan3D-paint)
            if normal_maps is not None and len(normal_maps) > 0:
                normal_latents = []
                for normal_map in normal_maps:
                    normal_tensor = pipe.preprocess_video([normal_map]).squeeze(0)
                    normal_latent = pipe.vae.encode([normal_tensor], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                    normal_latents.append(normal_latent)
                normal_latents = torch.concat(normal_latents, dim=1)
                # Copy to match vace_context frame length
                normal_latents = torch.concat([normal_latents] * (vace_context.shape[2] // normal_latents.shape[2]), dim=2)
                # print("normal_latents.shape before adding normal: ", normal_latents.shape)
                # print("vace_context.shape before adding normal: ", vace_context.shape)
                vace_context = torch.concat((vace_context, normal_latents), dim=1)
            
            # Encode and concatenate position maps (for VACE context, similar to Hunyuan3D-paint)
            if position_maps is not None and len(position_maps) > 0:
                position_latents = []
                for position_map in position_maps:
                    position_tensor = pipe.preprocess_video([position_map]).squeeze(0)
                    position_latent = pipe.vae.encode([position_tensor], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                    position_latents.append(position_latent)
                position_latents = torch.concat(position_latents, dim=1)
                # Copy to match vace_context frame length
                position_latents = torch.concat([position_latents] * (vace_context.shape[2] // position_latents.shape[2]), dim=2)
                # print("position_latents.shape before adding position: ", position_latents.shape)
                # print("vace_context.shape before adding position: ", vace_context.shape)
                vace_context = torch.concat((vace_context, position_latents), dim=1)

            new_channels = vace_context.shape[1]
            # Automatically update VACE model dimension if needed
            if pipe.vace is not None and pipe.vace.vace_in_dim != new_channels:
                # print(f"Updating VACE model vace_in_dim from {pipe.vace.vace_in_dim} to {new_channels}")
                pipe.update_vace_in_dim(new_channels)
            if pipe.vace2 is not None and pipe.vace2.vace_in_dim != new_channels:
                # print(f"Updating VACE2 model vace_in_dim from {pipe.vace2.vace_in_dim} to {new_channels}")
                pipe.update_vace_in_dim(new_channels, vace_model=pipe.vace2)
            
            # Structure of vace_context:
            #  frame_num
            # ------------------------
            # |vace_reference |inactivate | 16
            # |zeros          |reactivate | 16
            # |zeros          |mask       | 64
            # ------------------------
            # num_ref_patches for ref_gating (approach 1): ref_frames * (H//2) * (W//2)
            num_ref_patches = 0
            if _check_multiview_reference_mode(multiview_reference_mode, "ref_gating") and vace_reference_image is not None:
                num_ref_frames = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
                t, ht, wt = vace_context.shape[2], vace_context.shape[3], vace_context.shape[4]
                num_ref_patches = num_ref_frames * (ht // 2) * (wt // 2)
            return {
                "vace_context": vace_context,
                "vace_scale": vace_scale,
                "vace_control_context": vace_control_context,
                "vace_control_scale": multiview_zero_conv_scale * ref_weight,
                "num_ref_patches": num_ref_patches,
            }
        else:
            return {"vace_context": None, "vace_scale": vace_scale, "vace_control_context": None, "vace_control_scale": multiview_zero_conv_scale * ref_weight, "num_ref_patches": 0}


class WanVideoUnit_MultiviewIPAdapter(PipelineUnit):
    """
    Pipeline unit for IP-Adapter style multiview reference image injection.
    
    Encodes multiview reference images using CLIP/image encoder and generates
    K, V for cross-attention injection in transformer blocks.
    """
    def __init__(self):
        super().__init__(
            input_params=("multiview_reference_image", "multiview_reference_mode", "multiview_reference_weight", "multiview_ipadapter_scale",
                         "num_views", "view_indices"),
            onload_model_names=("image_encoder", "multiview_ipadapter", "multiview_feature_bank_adapter")
        )
    
    def process(
        self,
        pipe: WanVideoPipeline,
        multiview_reference_image,
        multiview_reference_mode,
        multiview_reference_weight,
        multiview_ipadapter_scale,
        num_views,
        view_indices,
    ):
        """
        Process multiview reference images for IP-Adapter or Feature Bank injection.
        
        - ipadapter: CLIP CLS token per view -> compressed projection -> K,V
        - feature_bank: Full image encoder output (all 257 tokens per view) -> Feature Bank -> K,V
          Q from latent z_t attends to ALL viewpoints; allows borrowing from adjacent views.
        
        Returns:
            Dictionary with "multiview_ipadapter_kv" if mode is enabled
        """
        if multiview_reference_image is None or len(multiview_reference_image) == 0:
            return {}
        
        use_feature_bank = _check_multiview_reference_mode(multiview_reference_mode, "feature_bank")
        use_ipadapter = _check_multiview_reference_mode(multiview_reference_mode, "ipadapter")
        if not use_feature_bank and not use_ipadapter:
            return {}
        
        if not isinstance(multiview_reference_image, list):
            multiview_reference_image = [multiview_reference_image]
        
        pipe.load_models_to_device(["image_encoder"])
        w = multiview_reference_weight if multiview_reference_weight is not None else 1.0
        scale = (multiview_ipadapter_scale if multiview_ipadapter_scale is not None else 1.0) * w
        
        # ---------- Feature Bank mode: full encoder output, all views concatenated ----------
        if use_feature_bank:
            if pipe.multiview_feature_bank_adapter is None:
                print("[WARNING] multiview_feature_bank_adapter is None, skipping Feature Bank injection")
                return {}
            feature_bank_list = []
            for ref_img in multiview_reference_image:
                processed = pipe.preprocess_image(ref_img).to(device=pipe.device, dtype=pipe.torch_dtype)
                if hasattr(pipe.image_encoder, "encode_image"):
                    image_emb = pipe.image_encoder.encode_image([processed])  # (1, 257, 1280)
                    if image_emb.dim() == 3:
                        # Keep ALL tokens (Feature Bank): (1, 257, 1280)
                        feature_bank_list.append(image_emb.squeeze(0))  # (257, 1280)
                    else:
                        image_emb = image_emb.view(image_emb.shape[0], -1).unsqueeze(1).expand(-1, 257, -1)
                        feature_bank_list.append(image_emb.squeeze(0))
                else:
                    print("[WARNING] image_encoder does not have encode_image, skipping Feature Bank")
                    return {}
            # Concatenate: (num_views * 257, 1280) -> (1, num_views * 257, 1280)
            feature_bank = torch.cat(feature_bank_list, dim=0).unsqueeze(0)
            pipe.load_models_to_device(["multiview_feature_bank_adapter"])
            ip_kv_dict = pipe.multiview_feature_bank_adapter(feature_bank, scale=scale)
            return {"multiview_ipadapter_kv": ip_kv_dict}
        
        # ---------- IP-Adapter mode: CLS token per view ----------
        if pipe.multiview_ipadapter is None:
            print("[WARNING] multiview_ipadapter is None, skipping IP-Adapter injection")
            return {}
        multiview_image_embeds = []
        for ref_img in multiview_reference_image:
            processed = pipe.preprocess_image(ref_img).to(device=pipe.device, dtype=pipe.torch_dtype)
            if hasattr(pipe.image_encoder, "encode_image"):
                image_emb = pipe.image_encoder.encode_image([processed])  # (1, 257, 1280)
                if image_emb.dim() == 3:
                    image_emb = image_emb[:, 0, :]  # CLS token
                elif image_emb.dim() == 2:
                    pass
                else:
                    image_emb = image_emb.view(image_emb.shape[0], -1)
                multiview_image_embeds.append(image_emb.squeeze(0))
            else:
                print("[WARNING] image_encoder does not have encode_image method, skipping IP-Adapter")
                return {}
        multiview_image_embeds = torch.stack(multiview_image_embeds, dim=0).unsqueeze(0)  # (1, num_views, embed_dim)
        if view_indices is None:
            if num_views is not None and num_views > 1:
                view_indices = torch.arange(num_views, device=multiview_image_embeds.device).unsqueeze(0)
            else:
                view_indices = torch.zeros((1, len(multiview_reference_image)), device=multiview_image_embeds.device, dtype=torch.long)
        pipe.load_models_to_device(["multiview_ipadapter"])
        ip_kv_dict = pipe.multiview_ipadapter(multiview_image_embeds, view_indices=view_indices, scale=scale)
        return {"multiview_ipadapter_kv": ip_kv_dict}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",)
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio", None), inputs_shared.pop("audio_embeds", None), inputs_shared.get("audio_sample_rate", 16000)
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video", None), inputs_shared.pop("s2v_pose_latents", None), inputs_shared.pop("motion_video", None)

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class WanVideoPostUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"))

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoPostUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoPostUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoPostUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    vace_control_context = None,
    vace_control_scale: float = 1.0,
    num_ref_patches: Optional[int] = None,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    position_maps: Optional[torch.Tensor] = None,  # [B, T, H, W, 3] position maps for 3D-aware RoPE
    normal_maps: Optional[torch.Tensor] = None,  # [B, T, H, W, 3] normal maps (optional)
    # Multiview parameters (inspired by cosmos-predict2.5)
    num_views: int = 1,  # Number of views for multiview generation
    view_indices: Optional[torch.Tensor] = None,  # View indices for cross-view attention (B, V)
    multiview_ipadapter_kv: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,  # IP-Adapter K, V dict
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            vace_control_context=vace_control_context,
            vace_control_scale=vace_control_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # print(f"x shape: {x.shape}") 
    # auto-multiview: x shape: torch.Size([1, 16, 88, 16, 8])
    # standard vace: x shape: torch.Size([1, 16, 25, 60, 104])
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Camera control
    x = dit.patchify(x, control_camera_latents_input)
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    # print("f, h, w:", f, h, w, flush=True) # f, h, w: 25 60 104
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            control_context=vace_control_context,
            control_scale=vace_control_scale,
            num_ref_patches=num_ref_patches,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    # Collect features from all transformer blocks for segmentation
    diffusion_features = []
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
        # Note: When using tea_cache, we don't collect features as the blocks are not executed
        # Segmentation will be skipped in this case
    else:
        def create_custom_forward(module):
            def custom_forward(x, context, t_mod, freqs, ip_kv=None):
                return module(x, context, t_mod, freqs, ip_kv=ip_kv)
            return custom_forward
        
        # Collect features from all transformer blocks for segmentation
        diffusion_features = []
        for block_id, block in enumerate(dit.blocks):
            # Get IP-Adapter K, V for this block if available
            ip_kv = None
            if multiview_ipadapter_kv is not None and block_id in multiview_ipadapter_kv:
                ip_kv = multiview_ipadapter_kv[block_id]
            
            # Block
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, ip_kv,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs, ip_kv,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs, ip_kv=ip_kv)
            
            # Collect features for segmentation/depth head
            if dit.perception_head is not None or (hasattr(dit, 'depth_head') and dit.depth_head is not None):
                diffusion_features.append(x)
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
    
    # Do Latent Segmentation
    mask_pred = None
    if dit.perception_head is not None and len(diffusion_features) > 0:
        features = []
        start_layer = getattr(dit, 'start_layer', 0)
        end_layer = getattr(dit, 'end_layer', len(dit.blocks) - 1)
        for i in range(start_layer, end_layer + 1):
            if i < len(diffusion_features):
                # Reshape from [B, T*H*W, C] to [B*T, C, H, W]
                spatial_feature = rearrange(
                    diffusion_features[i],
                    "B (T H W) C -> (B T) C H W",
                    T=f,
                    H=h,
                    W=w,
                )
                features.append(spatial_feature)
        
        if len(features) > 0:
            mask_pred = dit.perception_head(features)
            # print("mask_pred shape:", mask_pred.shape, flush=True) # torch.Size([1, 16, 25, 60, 104])
            # Reshape back to [B, T, C, H, W]
            mask_pred = rearrange(
                mask_pred,
                "(B T) C H W -> B T C H W",
                T=f,
                H=mask_pred.shape[2],
                W=mask_pred.shape[3],
            )
    
    # Do Latent Depth Prediction
    depth_pred = None
    if hasattr(dit, 'depth_head') and dit.depth_head is not None and len(diffusion_features) > 0:
        features = []
        start_layer = getattr(dit, 'start_layer', 0)
        end_layer = getattr(dit, 'end_layer', len(dit.blocks) - 1)
        for i in range(start_layer, end_layer + 1):
            if i < len(diffusion_features):
                # Reshape from [B, T*H*W, C] to [B*T, C, H, W]
                spatial_feature = rearrange(
                    diffusion_features[i],
                    "B (T H W) C -> (B T) C H W",
                    T=f,
                    H=h,
                    W=w,
                )
                features.append(spatial_feature)
        
        if len(features) > 0:
            depth_pred = dit.depth_head(features)
            # print("depth_pred shape:", depth_pred.shape, flush=True) # torch.Size([1, 16, 25, 60, 104])
            # Reshape back to [B, T, C, H, W]
            depth_pred = rearrange(
                depth_pred,
                "(B T) C H W -> B T C H W",
                T=f,
                H=depth_pred.shape[2],
                W=depth_pred.shape[3],
            )
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))

    return x, mask_pred, depth_pred


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
