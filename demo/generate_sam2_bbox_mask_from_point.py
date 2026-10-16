#!/usr/bin/env python3
"""
使用单点 prompt 调用 SAM2 对视频进行分割，并导出：
1) 二值分割 mask 视频（黑白）
2) bbox mask 视频（黑白，默认填充矩形）
3) 每帧 bbox 坐标 json

示例：
python demo/generate_sam2_bbox_mask_from_point.py \
  --video demo/src/road3.mp4 \
  --point "320,240" \
  --output-mask-video demo/mask/road3_sam_mask.mp4 \
  --output-bbox-mask-video demo/mask/road3_bbox_mask.mp4 \
  --output-bbox-json demo/mask/road3_bbox.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# Add project and SAM2 to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SAM2_ROOT = os.path.join(PROJECT_ROOT, "third-party", "sam2")
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


def parse_point_str(point_str: str) -> Tuple[float, float]:
    s = point_str.strip().replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid --point format: {point_str}. Use 'x,y' or 'x y'.")
    return float(parts[0]), float(parts[1])


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return bbox as (x_min, y_min, x_max, y_max), or None for empty mask."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def create_predictor(
    sam_checkpoint: Optional[str],
    sam_config: Optional[str],
    device: str,
):
    if sam_checkpoint and sam_config:
        return build_sam2_video_predictor(sam_config, sam_checkpoint, device=device)

    try:
        return SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
    except Exception:
        env_ckpt = os.getenv("SAM2_CHECKPOINT")
        env_cfg = os.getenv("SAM2_CONFIG")
        if not env_ckpt or not env_cfg:
            raise RuntimeError(
                "Cannot load SAM2 model. Provide both --sam-checkpoint and --sam-config, "
                "or set SAM2_CHECKPOINT and SAM2_CONFIG env vars."
            )
        return build_sam2_video_predictor(env_cfg, env_ckpt, device=device)


def write_mask_video(
    masks: Sequence[np.ndarray],
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> None:
    ensure_parent(output_path)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    try:
        for m in masks:
            frame = np.where(m > 0, 255, 0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def build_bbox_mask(
    bbox: Optional[Tuple[int, int, int, int]],
    height: int,
    width: int,
    mode: str,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint8)
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    if mode == "filled":
        out[y1 : y2 + 1, x1 : x2 + 1] = 255
    else:
        out[y1, x1 : x2 + 1] = 255
        out[y2, x1 : x2 + 1] = 255
        out[y1 : y2 + 1, x1] = 255
        out[y1 : y2 + 1, x2] = 255
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use SAM2 single-point prompt to generate mask/bbox mask videos."
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument(
        "--point",
        type=str,
        required=True,
        help="Point prompt on first frame, format: 'x,y' or 'x y'",
    )
    parser.add_argument(
        "--point-normalized",
        action="store_true",
        help="Interpret --point as normalized coordinates in [0,1]",
    )
    parser.add_argument(
        "--output-mask-video",
        type=str,
        required=True,
        help="Output binary mask video path",
    )
    parser.add_argument(
        "--output-bbox-mask-video",
        type=str,
        required=True,
        help="Output bbox mask video path",
    )
    parser.add_argument(
        "--output-bbox-json",
        type=str,
        default=None,
        help="Optional output bbox json path",
    )
    parser.add_argument(
        "--bbox-mode",
        type=str,
        default="filled",
        choices=["filled", "outline"],
        help="bbox mask style for output video",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="third-party/sam2/checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint path (optional when using HF pretrained)",
    )
    parser.add_argument(
        "--sam-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path (used with --sam-checkpoint or SAM2_CONFIG)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read video metadata from: {video_path}")
    if fps <= 0:
        fps = 25.0

    x, y = parse_point_str(args.point)
    if args.point_normalized:
        x = x * width
        y = y * height
    x = float(np.clip(x, 0, width - 1))
    y = float(np.clip(y, 0, height - 1))

    predictor = create_predictor(args.sam_checkpoint, args.sam_config, args.device)
    inference_state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )

    prompt_points = np.array([[x, y]], dtype=np.float32)
    prompt_labels = np.array([1], dtype=np.int32)  # foreground point
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=prompt_points,
        labels=prompt_labels,
    )

    masks_per_frame: List[np.ndarray] = [np.zeros((height, width), dtype=np.uint8) for _ in range(frame_count)]
    bboxes: List[Optional[Tuple[int, int, int, int]]] = [None for _ in range(frame_count)]

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if out_frame_idx < 0 or out_frame_idx >= frame_count:
            continue
        logits = out_mask_logits
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        logits = np.asarray(logits)
        if logits.ndim == 4:
            logits = logits[:, 0, :, :]
        if logits.ndim == 2:
            logits = logits[None, ...]

        frame_mask = np.zeros((height, width), dtype=np.uint8)
        for i, obj_id in enumerate(out_obj_ids):
            if int(obj_id) != 1:
                continue
            obj_mask = (logits[i] > 0.0).astype(np.uint8)
            if obj_mask.shape != frame_mask.shape:
                obj_mask = cv2.resize(
                    obj_mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
            frame_mask = np.maximum(frame_mask, obj_mask)

        masks_per_frame[out_frame_idx] = frame_mask
        bboxes[out_frame_idx] = mask_to_bbox(frame_mask)

    bbox_masks = [
        build_bbox_mask(bbox, height=height, width=width, mode=args.bbox_mode)
        for bbox in bboxes
    ]

    write_mask_video(masks_per_frame, args.output_mask_video, fps, width, height)
    write_mask_video(bbox_masks, args.output_bbox_mask_video, fps, width, height)

    if args.output_bbox_json:
        ensure_parent(args.output_bbox_json)
        payload = {
            "video": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "point_prompt": {
                "x": x,
                "y": y,
                "is_normalized_input": bool(args.point_normalized),
            },
            "bboxes_xyxy": [
                None if b is None else [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                for b in bboxes
            ],
        }
        with open(args.output_bbox_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[Done] mask video: {args.output_mask_video}")
    print(f"[Done] bbox mask video: {args.output_bbox_mask_video}")
    if args.output_bbox_json:
        print(f"[Done] bbox json: {args.output_bbox_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
