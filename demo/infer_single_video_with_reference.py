"""Single-video multiview inference with reference images."""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from diffsynth import VideoData, load_state_dict, save_video
from diffsynth.models.wan_video_segmentation import SemanticFPNHead
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA_PATH = str(_REPO_ROOT / "models/checkpoints/diffsynth_multiview_lora.safetensors")
DEFAULT_FULL_CHECKPOINT_PATH = str(_REPO_ROOT / "models/checkpoints/diffsynth_multiview_full.safetensors")
DEFAULT_CLIP_PATH = str(_REPO_ROOT / "models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)

DEFAULT_VACE_LAYERS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-video multiview inference with reference images.")
    parser.add_argument("--video_name", default="maple", help="Video name used for default paths.")
    args, _ = parser.parse_known_args()

    parser.add_argument("--video_path", default=f"./demo/src/{args.video_name}.mp4", help="Input video or image folder.")
    parser.add_argument(
        "--reference_image_path",
        default=f"./demo/ref/{args.video_name}",
        help="Reference image file or directory.",
    )
    parser.add_argument("--mask_video_path", default=f"./demo/mask/{args.video_name}.mp4", help="Mask video or image folder.")
    parser.add_argument("--output_path", default=None, help="Output mp4 path. Defaults to demo/out or demo/out_fullft.")
    parser.add_argument("--prompt", default=f"./demo/prompt/{args.video_name}.txt", help="Prompt text or path to a .txt file.")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")

    parser.add_argument(
        "--base_model_path",
        default=os.getenv("WAN2_1_VACE_1_3B_MODEL_PATH", str(_REPO_ROOT / "models/Wan-AI/Wan2.1-VACE-1.3B")),
        help="Base model root.",
    )
    parser.add_argument("--lora_model_path", default=DEFAULT_LORA_PATH, help="LoRA checkpoint path.")
    parser.add_argument("--full_checkpoint_path", default=DEFAULT_FULL_CHECKPOINT_PATH, help="Full fine-tuning checkpoint path.")
    parser.add_argument("--checkpoint_type", choices=("lora", "full"), default="lora", help="Checkpoint type to load.")
    parser.add_argument("--no_load_lora", action="store_true", help="Skip LoRA loading (base model only).")

    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=81, help="Frame count, adjusted to 4n+1 (min 17).")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=10)

    parser.add_argument("--multiview_reference_mode", default="temporal_concat+feature_bank")
    parser.add_argument("--multiview_ipadapter_scale", type=float, default=0.2)
    parser.add_argument(
        "--disable_multiview_consistency_check",
        action="store_true",
        help="Disable CLIP text-image consistency weighting for multiview references.",
    )
    parser.add_argument(
        "--clip_consistency_model_path",
        default=os.getenv("CLIP_CONSISTENCY_MODEL_PATH", str(_REPO_ROOT / "models/LongCLIP-GmP-ViT-L-14")),
        help="LongCLIP path for multiview consistency check.",
    )
    parser.add_argument("--num_reference_views", type=int, default=4, help="Max number of reference views to use.")
    parser.add_argument(
        "--infer_depth_seg",
        action="store_true",
        help="Load depth/seg heads from checkpoint and save predicted depth/segmentation videos.",
    )
    return parser


def resolve_media_path(path: str, label: str) -> str:
    path_obj = Path(path)
    if path_obj.exists():
        return str(path_obj)
    if path.endswith(".mp4"):
        alt = path[:-4]
        if Path(alt).exists():
            return alt
    raise FileNotFoundError(f"{label} does not exist: {path}")


def resolve_target_num_frames(requested_frames: int) -> int:
    n = max(17, requested_frames)
    if n % 4 != 1:
        n -= (n - 1) % 4
        print(f"[WARN] num_frames adjusted to {n} to satisfy 4n+1.")
    return n


def collect_frames_padded(video_data: VideoData, target_len: int, label: str):
    n = len(video_data)
    if n < 1:
        raise ValueError(f"{label} has no frames.")
    if n >= target_len:
        return [video_data[i] for i in range(target_len)]
    frames = [video_data[i] for i in range(n)]
    last = frames[-1]
    pad_count = target_len - n
    frames.extend([last.copy() for _ in range(pad_count)])
    print(f"[INFO] {label}: {n} -> {target_len} frames (last frame repeated {pad_count}x).")
    return frames


def load_video_data_from_path(path: str, height: int, width: int, arg_name: str) -> VideoData:
    path_obj = Path(path)
    if path_obj.is_dir():
        return VideoData(image_folder=path, height=height, width=width)
    if path_obj.is_file():
        return VideoData(video_file=path, height=height, width=width)
    raise FileNotFoundError(f"{arg_name} does not exist: {path}")


def load_multiview_reference_images(reference_path: str):
    path_obj = Path(reference_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Reference path does not exist: {reference_path}")

    if path_obj.is_dir():
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        image_paths = sorted(
            {
                p
                for pattern in patterns
                for p in path_obj.glob(pattern)
            }
            | {
                p
                for pattern in patterns
                for p in path_obj.glob(pattern.upper())
            }
        )
    else:
        image_paths = [path_obj]

    if not image_paths:
        raise ValueError(f"No images found under reference path: {reference_path}")
    return [Image.open(p).convert("RGB") for p in image_paths]


def resolve_prompt(prompt_arg: str) -> str:
    prompt_path = Path(prompt_arg)
    if prompt_path.is_file():
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Prompt file is empty: {prompt_arg}")
        return prompt_text
    return prompt_arg


def _get_vace_layers(pipe):
    if not hasattr(pipe, "vace") or pipe.vace is None:
        return DEFAULT_VACE_LAYERS
    if hasattr(pipe.vace, "vace_layers_mapping"):
        return tuple(pipe.vace.vace_layers_mapping.keys())
    if hasattr(pipe.vace, "vace_layers"):
        return pipe.vace.vace_layers
    return DEFAULT_VACE_LAYERS


def _get_attention_dims(dit_model):
    if hasattr(dit_model, "blocks") and len(dit_model.blocks) > 0:
        first_block = dit_model.blocks[0]
        if hasattr(first_block, "cross_attn") and hasattr(first_block.cross_attn, "num_heads"):
            num_attention_heads = first_block.cross_attn.num_heads
        else:
            num_attention_heads = getattr(dit_model, "num_heads", 24)
    else:
        num_attention_heads = getattr(dit_model, "num_heads", 24)
    dim = getattr(dit_model, "dim", 1536)
    return num_attention_heads, dim // num_attention_heads, dim


def _extract_state_dict(checkpoint_state_dict, strip_prefixes, keep_prefixes=()):
    state_dict = {}
    for key, value in checkpoint_state_dict.items():
        for prefix in strip_prefixes:
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = value
                break
        else:
            for prefix in keep_prefixes:
                if key.startswith(prefix):
                    state_dict[key] = value
                    break
    return state_dict


def init_head_from_checkpoint(model, checkpoint_state_dict, head_name, key_prefixes, device, dtype):
    if model is None:
        return False
    if not hasattr(model, head_name) or getattr(model, head_name) is None:
        num_layers = len(model.blocks) if hasattr(model, "blocks") else 30
        if not hasattr(model, "start_layer"):
            model.start_layer = 0
        if not hasattr(model, "end_layer"):
            model.end_layer = num_layers - 1
        head = SemanticFPNHead(
            in_channels=getattr(model, "dim", 1536),
            out_channels=getattr(model, "in_dim", 16),
            num_tensors=model.end_layer - model.start_layer + 1,
            patch_size=getattr(model, "patch_size", (1, 2, 2)),
        )
        setattr(model, head_name, head.to(dtype=dtype).to(device))

    state_dict = _extract_state_dict(checkpoint_state_dict, key_prefixes)
    if not state_dict:
        print(f"[INFO] No {head_name} weights found in checkpoint.")
        return False
    try:
        getattr(model, head_name).load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded {head_name} ({len(state_dict)} parameters).")
        return True
    except Exception as e:
        print(f"[WARN] Failed to load {head_name}: {e}")
        return False


def _init_adapter(pipe, adapter_cls, attr_name, checkpoint_state_dict, strip_prefixes, keep_prefixes, extra_kwargs, dtype):
    num_attention_heads, attention_head_dim, dim = _get_attention_dims(pipe.dit)
    adapter = adapter_cls(
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        cross_attention_dim=dim,
        clip_embeddings_dim=1280,
        **extra_kwargs,
    ).to(dtype=dtype).to(pipe.device)
    state_dict = _extract_state_dict(checkpoint_state_dict, strip_prefixes, keep_prefixes)
    if state_dict:
        adapter.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded {attr_name} ({len(state_dict)} parameters).")
    else:
        print(f"[WARN] No {attr_name} weights in checkpoint; using random init.")
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad = False
    setattr(pipe, attr_name, adapter)


def init_multiview_adapters(pipe, checkpoint_state_dict, multiview_reference_mode, dtype):
    mode = multiview_reference_mode or ""
    vace_layers = _get_vace_layers(pipe)

    if "ipadapter" in mode and pipe.image_encoder is not None and hasattr(pipe, "dit") and pipe.dit is not None:
        from diffsynth.models.wan_video_multiview_ipadapter import MultiviewIPAdapter

        _init_adapter(
            pipe,
            MultiviewIPAdapter,
            "multiview_ipadapter",
            checkpoint_state_dict,
            ("pipe.multiview_ipadapter.", "multiview_ipadapter."),
            ("image_proj.", "ipadapter_modules."),
            {"num_tokens": 16, "num_views": 4, "vace_layers": vace_layers},
            dtype,
        )
    elif "ipadapter" in mode and pipe.image_encoder is None:
        print("[WARN] image_encoder is None; skipping IP-Adapter initialization.")

    if "feature_bank" in mode and hasattr(pipe, "vace") and pipe.vace is not None and hasattr(pipe, "dit") and pipe.dit is not None:
        from diffsynth.models.wan_video_multiview_ipadapter import MultiviewFeatureBankAdapter

        _init_adapter(
            pipe,
            MultiviewFeatureBankAdapter,
            "multiview_feature_bank_adapter",
            checkpoint_state_dict,
            ("pipe.multiview_feature_bank_adapter.", "multiview_feature_bank_adapter."),
            ("image_proj.", "fb_modules."),
            {"vace_layers": vace_layers},
            dtype,
        )


def apply_mask_to_video(frames, masks):
    masked = []
    for frame, mask in zip(frames, masks):
        frame_np = np.array(frame)
        mask_np = np.array(mask.convert("RGB")) / 255.0
        masked.append(Image.fromarray((frame_np * (1 - mask_np)).astype(np.uint8)))
    return masked


def _derive_sidecar_output_path(output_path: str, suffix: str) -> str:
    p = Path(output_path)
    if p.suffix.lower() == ".mp4":
        return str(p.with_name(p.stem + suffix + p.suffix))
    return output_path + suffix + ".mp4"


def _decode_latent_video_to_pil_frames(pipe, latent_b_t_c_h_w: torch.Tensor, num_ref_frames: int):
    if latent_b_t_c_h_w is None:
        return None
    pred = latent_b_t_c_h_w
    if num_ref_frames > 0 and pred.shape[1] > num_ref_frames:
        pred = pred[:, num_ref_frames:, :, :, :]
    pred = rearrange(pred, "b t c h w -> b c t h w")

    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        decoded = pipe.vae.decode(
            pred,
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
    pipe.load_models_to_device([])

    frames = []
    _, _, t, _, _ = decoded.shape
    for t_idx in range(t):
        frame_tensor = decoded[0, :, t_idx, :, :].cpu().float()
        frame_tensor = (frame_tensor / 2 + 0.5).clamp(0, 1)
        frame_array = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame_array))
    return frames


def load_checkpoint(pipe, args):
    checkpoint_path = args.lora_model_path if args.checkpoint_type == "lora" else args.full_checkpoint_path
    checkpoint_state_dict = {}

    if args.checkpoint_type == "full":
        print(f"[INFO] Loading full checkpoint: {checkpoint_path}")
        checkpoint_state_dict = load_state_dict(checkpoint_path)
        vace_keys = set(pipe.vace.state_dict().keys())
        vace_state = {}
        for key, value in checkpoint_state_dict.items():
            new_key = key
            for prefix in ("pipe.vace.", "vace."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            if new_key in vace_keys or new_key.startswith(
                ("vace_", "vace_blocks", "vace_patch_embedding", "blocks.", "patch_embedding.", "ref_")
            ):
                vace_state[new_key] = value
        result = pipe.vace.load_state_dict(vace_state, strict=False)
        print(f"[INFO] Loaded VACE weights: {len(vace_state)} keys; missing={len(result[0])}, unexpected={len(result[1])}")
    elif not args.no_load_lora:
        print(f"[INFO] Loading LoRA: {checkpoint_path}")
        pipe.load_lora(pipe.vace, checkpoint_path, alpha=1)
        checkpoint_state_dict = load_state_dict(checkpoint_path)

    return checkpoint_state_dict


def init_depth_seg_heads(pipe, checkpoint_state_dict):
    head_specs = [
        (pipe.dit if hasattr(pipe, "dit") else None, "perception_head", ("pipe.dit.perception_head.", "dit.perception_head.", "perception_head.")),
        (pipe.dit if hasattr(pipe, "dit") else None, "depth_head", ("pipe.dit.depth_head.", "dit.depth_head.", "depth_head.")),
        (pipe.dit2 if hasattr(pipe, "dit2") else None, "perception_head", ("vace.dit2.perception_head.", "dit2.perception_head.")),
        (pipe.dit2 if hasattr(pipe, "dit2") else None, "depth_head", ("vace.dit2.depth_head.", "dit2.depth_head.")),
    ]
    for model, head_name, prefixes in head_specs:
        init_head_from_checkpoint(model, checkpoint_state_dict, head_name, prefixes, pipe.device, torch.bfloat16)


def validate_args(args):
    if not args.base_model_path:
        raise ValueError("Set --base_model_path or WAN2_1_VACE_1_3B_MODEL_PATH.")
    if not Path(args.base_model_path).exists():
        raise FileNotFoundError(f"Base model path does not exist: {args.base_model_path}")

    args.video_path = resolve_media_path(args.video_path, "video_path")
    if not Path(args.reference_image_path).exists():
        raise FileNotFoundError(f"Reference image path does not exist: {args.reference_image_path}")
    args.mask_video_path = resolve_media_path(args.mask_video_path, "mask_video_path")

    checkpoint_path = args.lora_model_path if args.checkpoint_type == "lora" else args.full_checkpoint_path
    if args.checkpoint_type == "lora" and not args.no_load_lora and not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"LoRA path does not exist: {checkpoint_path}")
    if args.checkpoint_type == "full" and not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Full checkpoint path does not exist: {checkpoint_path}")

    use_consistency_check = not args.disable_multiview_consistency_check
    if use_consistency_check:
        os.environ["CLIP_CONSISTENCY_MODEL_PATH"] = args.clip_consistency_model_path
        if not Path(args.clip_consistency_model_path).exists():
            raise FileNotFoundError(
                f"CLIP consistency model not found: {args.clip_consistency_model_path}. "
                "Download zer0int/LongCLIP-GmP-ViT-L-14 or pass --disable_multiview_consistency_check."
            )
    return use_consistency_check


def main():
    args = build_parser().parse_args()
    if args.output_path is None:
        folder = "./demo/out_fullft" if args.checkpoint_type == "full" else "./demo/out"
        args.output_path = f"{folder}/{args.video_name}_{args.seed}.mp4"

    use_consistency_check = validate_args(args)
    prompt = resolve_prompt(args.prompt)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=f"{args.base_model_path}/diffusion_pytorch_model.safetensors", offload_device="cpu"),
            ModelConfig(path=f"{args.base_model_path}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path=f"{args.base_model_path}/Wan2.1_VAE.pth", offload_device="cpu"),
            ModelConfig(path=DEFAULT_CLIP_PATH, offload_device="cpu"),
        ],
        tokenizer_config=ModelConfig(path=f"{args.base_model_path}/google/umt5-xxl/"),
    )

    checkpoint_state_dict = load_checkpoint(pipe, args)
    if args.infer_depth_seg:
        init_depth_seg_heads(pipe, checkpoint_state_dict)
    init_multiview_adapters(pipe, checkpoint_state_dict, args.multiview_reference_mode, torch.bfloat16)
    pipe.enable_vram_management()

    actual_num_frames = resolve_target_num_frames(args.num_frames)
    video_data = load_video_data_from_path(args.video_path, args.height, args.width, "video_path")
    mask_video_data = load_video_data_from_path(args.mask_video_path, args.height, args.width, "mask_video_path")
    video_frames = collect_frames_padded(video_data, actual_num_frames, "video_path")
    mask_frames = collect_frames_padded(mask_video_data, actual_num_frames, "mask_video_path")
    video_frames = apply_mask_to_video(video_frames, mask_frames)

    multiview_reference_image = load_multiview_reference_images(args.reference_image_path)
    if args.num_reference_views > 0:
        multiview_reference_image = multiview_reference_image[: args.num_reference_views]

    size = (args.width, args.height)
    video_frames = [frame.convert("RGB").resize(size) for frame in video_frames]
    mask_frames = [mask.convert("RGB").resize(size) for mask in mask_frames]
    multiview_reference_image = [image.convert("RGB").resize(size) for image in multiview_reference_image]

    print(f"[INFO] Running inference: {len(video_frames)} frames, {len(multiview_reference_image)} reference views.")
    generated_video = pipe(
        num_inference_steps=args.num_inference_steps,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        vace_video=video_frames,
        vace_video_mask=mask_frames,
        vace_reference_image=None,
        multiview_reference_image=multiview_reference_image,
        normal_maps=None,
        position_maps=None,
        trajectory_maps=None,
        trajectory_scale=None,
        num_frames=actual_num_frames,
        seed=args.seed,
        tiled=True,
        multiview_reference_mode=args.multiview_reference_mode,
        multiview_ipadapter_scale=args.multiview_ipadapter_scale,
        use_multiview_consistency_check=use_consistency_check,
    )
    generated_video = generated_video[:actual_num_frames]
    save_video(generated_video, args.output_path, fps=args.fps, quality=args.quality)
    print(f"[DONE] Saved generated video to: {args.output_path}")

    if not args.infer_depth_seg:
        return

    num_ref_frames = len(multiview_reference_image)
    for attr, suffix in (("_last_mask_pred", "_predicted_segmentation"), ("_last_depth_pred", "_predicted_depth")):
        latent = getattr(pipe, attr, None)
        if latent is None:
            continue
        frames = _decode_latent_video_to_pil_frames(pipe, latent, num_ref_frames=num_ref_frames)
        if not frames:
            continue
        out_path = _derive_sidecar_output_path(args.output_path, suffix)
        save_video(frames, out_path, fps=args.fps, quality=args.quality)
        print(f"[DONE] Saved {suffix.lstrip('_')} video to: {out_path}")
        delattr(pipe, attr)


if __name__ == "__main__":
    main()
