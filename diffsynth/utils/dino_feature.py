import os
from typing import Optional, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def _resolve_hf_model_path(model_name: str) -> str:
    if not model_name or not os.path.isdir(model_name):
        return model_name
    snapshots_dir = os.path.join(model_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_name
    for name in ("pytorch_model.bin", "model.safetensors", "config.json"):
        if os.path.isfile(os.path.join(model_name, name)):
            return model_name
    try:
        revisions = [
            revision for revision in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, revision))
        ]
    except OSError:
        return model_name
    if not revisions:
        return model_name
    for revision in revisions:
        resolved = os.path.join(snapshots_dir, revision)
        has_model = (
            os.path.isfile(os.path.join(resolved, "pytorch_model.bin"))
            or os.path.isfile(os.path.join(resolved, "model.safetensors"))
        )
        has_processor = os.path.isfile(os.path.join(resolved, "preprocessor_config.json"))
        if has_model and has_processor:
            return resolved
    return os.path.join(snapshots_dir, revisions[0])


class DINOFeatureLossHelper:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model_name = model_name or os.getenv("DINO_VIEW_CONTROL_MODEL_PATH") or "facebook/dinov2-base"
        self._device = device
        self._model = None
        self._processor = None
        self._image_size = (224, 224)
        self._image_mean = None
        self._image_std = None

    def _to_float_tensor(
        self,
        image: Union[Image.Image, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        elif isinstance(image, torch.Tensor):
            if image.ndim == 2:
                image = image.unsqueeze(0).unsqueeze(0)
            elif image.ndim == 3:
                if image.shape[0] in (1, 3):
                    image = image.unsqueeze(0)
                elif image.shape[-1] in (1, 3):
                    image = image.permute(2, 0, 1).unsqueeze(0)
                else:
                    image = image.unsqueeze(0)
            elif image.ndim == 5 and image.shape[2] == 1:
                image = image[:, :, 0, :, :]
            if image.ndim != 4:
                raise ValueError(f"Expected image tensor with 2, 3, 4 or 5 dims, got {tuple(image.shape)}")
            if image.shape[1] not in (1, 3) and image.shape[-1] in (1, 3):
                image = image.permute(0, 3, 1, 2)
            image = image.to(dtype=torch.float32)
            if image.min() < 0.0:
                image = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
            else:
                image = image.clamp(0.0, 1.0)
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        if device is not None:
            image = image.to(device=device, dtype=torch.float32)
        return image

    def _extract_first_frame_mask(
        self,
        mask_or_box: Optional[Union[Image.Image, torch.Tensor, list]],
    ) -> Optional[torch.Tensor]:
        if mask_or_box is None:
            return None
        if isinstance(mask_or_box, list):
            if len(mask_or_box) == 0:
                return None
            mask_or_box = mask_or_box[0]

        if isinstance(mask_or_box, Image.Image):
            mask_arr = np.array(mask_or_box.convert("L"), dtype=np.float32) / 255.0
            return torch.from_numpy(mask_arr)

        if not isinstance(mask_or_box, torch.Tensor):
            return None

        mask = mask_or_box.detach().float().cpu()
        if mask.ndim == 5:
            # Prefer [B, C, T, H, W]
            if mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim == 4:
                if mask.shape[1] > 1:
                    mask = mask[:, 0]
                else:
                    mask = mask[:, 0]
        if mask.ndim == 4:
            # Could be [C, T, H, W] or [B, C, H, W]
            if mask.shape[0] == 1 and mask.shape[1] in (1, 3):
                mask = mask[0]
            elif mask.shape[0] in (1, 3) and mask.shape[1] > 1:
                mask = mask[:, 0]
            else:
                mask = mask[0]
        if mask.ndim == 3:
            if mask.shape[0] in (1, 3):
                mask = mask.mean(dim=0)
            elif mask.shape[-1] in (1, 3):
                mask = mask.mean(dim=-1)
            else:
                mask = mask[0]
        if mask.ndim != 2:
            return None
        return mask

    def _compute_bbox_from_mask(
        self,
        mask: Optional[torch.Tensor],
        width: int,
        height: int,
        threshold: float = 0.1,
    ):
        if mask is None:
            return None
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="nearest",
            ).squeeze(0).squeeze(0)
        positive = mask > threshold
        if not positive.any():
            return None
        ys, xs = torch.where(positive)
        x_min = int(xs.min().item())
        x_max = int(xs.max().item()) + 1
        y_min = int(ys.min().item())
        y_max = int(ys.max().item()) + 1
        return x_min, y_min, x_max, y_max

    def _compute_bbox_from_image_content(
        self,
        image: Union[Image.Image, torch.Tensor],
        threshold: float = 0.03,
    ):
        image_tensor = self._to_float_tensor(image).cpu()
        image_gray = image_tensor[0].mean(dim=0)
        positive = image_gray > threshold
        if not positive.any():
            return None
        ys, xs = torch.where(positive)
        x_min = int(xs.min().item())
        x_max = int(xs.max().item()) + 1
        y_min = int(ys.min().item())
        y_max = int(ys.max().item()) + 1
        return x_min, y_min, x_max, y_max

    def _expand_bbox(self, bbox, width: int, height: int, expansion_ratio: float = 0.15):
        if bbox is None:
            return 0, 0, width, height
        x_min, y_min, x_max, y_max = bbox
        box_w = max(1, x_max - x_min)
        box_h = max(1, y_max - y_min)
        pad_w = int(round(box_w * expansion_ratio))
        pad_h = int(round(box_h * expansion_ratio))
        x_min = max(0, x_min - pad_w)
        y_min = max(0, y_min - pad_h)
        x_max = min(width, x_max + pad_w)
        y_max = min(height, y_max + pad_h)
        if x_max <= x_min:
            x_max = min(width, x_min + 1)
        if y_max <= y_min:
            y_max = min(height, y_min + 1)
        return x_min, y_min, x_max, y_max

    def _crop_tensor_by_bbox(self, image: Union[Image.Image, torch.Tensor], bbox):
        image_tensor = self._to_float_tensor(image)
        _, _, height, width = image_tensor.shape
        x_min, y_min, x_max, y_max = self._expand_bbox(bbox, width=width, height=height)
        return image_tensor[:, :, y_min:y_max, x_min:x_max]

    def _normalized_bbox(self, bbox, width: int, height: int):
        if bbox is None:
            return None
        x_min, y_min, x_max, y_max = bbox
        return (
            x_min / max(width, 1),
            y_min / max(height, 1),
            x_max / max(width, 1),
            y_max / max(height, 1),
        )

    def _denormalize_bbox(self, bbox, width: int, height: int):
        if bbox is None:
            return None
        x_min, y_min, x_max, y_max = bbox
        return (
            int(round(x_min * width)),
            int(round(y_min * height)),
            int(round(x_max * width)),
            int(round(y_max * height)),
        )

    def _ensure_loaded(self, device: Optional[Union[str, torch.device]] = None):
        if self._model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINO viewpoint loss requires transformers. Install with: pip install transformers"
            ) from exc

        resolved_path = _resolve_hf_model_path(self.model_name)
        if os.path.isfile(resolved_path):
            ext = os.path.splitext(resolved_path)[1].lower()
            if ext in {".pth", ".pt", ".bin", ".ckpt"}:
                raise ValueError(
                    "Unsupported DINO model path for multiview viewpoint loss: "
                    f"`{resolved_path}` is a raw checkpoint file. "
                    "This loss expects a Hugging Face Dinov2 model name or local model directory, "
                    "for example `facebook/dinov2-base` or a downloaded HF snapshot directory."
                )
        print(f"Loading DINO model from {resolved_path}")
        self._processor = AutoImageProcessor.from_pretrained(resolved_path)
        self._model = AutoModel.from_pretrained(resolved_path)
        self._model.eval()
        self._model.requires_grad_(False)

        model_device = torch.device(device or self._device or "cpu")
        self._model = self._model.to(model_device)

        crop_size = getattr(self._processor, "crop_size", None)
        size = getattr(self._processor, "size", None)
        height = width = 224
        if isinstance(crop_size, dict):
            height = crop_size.get("height", crop_size.get("shortest_edge", height))
            width = crop_size.get("width", crop_size.get("shortest_edge", width))
        elif isinstance(size, dict):
            edge = size.get("shortest_edge", size.get("height", height))
            height = size.get("height", edge)
            width = size.get("width", edge)
        elif isinstance(size, int):
            height = width = size
        self._image_size = (int(height), int(width))

        image_mean = getattr(self._processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(self._processor, "image_std", [0.229, 0.224, 0.225])
        self._image_mean = torch.tensor(image_mean, dtype=torch.float32, device=model_device).view(1, 3, 1, 1)
        self._image_std = torch.tensor(image_std, dtype=torch.float32, device=model_device).view(1, 3, 1, 1)

    def _prepare_image_tensor(
        self,
        image: Union[Image.Image, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        image = self._to_float_tensor(image, device=device)
        image = F.interpolate(
            image,
            size=self._image_size,
            mode="bicubic",
            align_corners=False,
        )
        image = (image - self._image_mean) / self._image_std
        return image

    def encode_image(
        self,
        image: Union[Image.Image, torch.Tensor],
        requires_grad: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        self._ensure_loaded(device)
        model_device = next(self._model.parameters()).device
        pixel_values = self._prepare_image_tensor(image, model_device)

        with torch.set_grad_enabled(requires_grad):
            outputs = self._model(pixel_values=pixel_values)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                feature = outputs.last_hidden_state[:, 0]
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feature = outputs.pooler_output
            else:
                raise ValueError("DINO model output does not contain last_hidden_state or pooler_output")
            feature = F.normalize(feature.float(), dim=-1)
        return feature

    def compute_feature_loss(
        self,
        predicted_image: torch.Tensor,
        reference_image: Union[Image.Image, torch.Tensor],
        foreground_mask_or_box: Optional[Union[Image.Image, torch.Tensor, list]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        predicted_for_loss = predicted_image
        reference_for_loss = reference_image

        

        if foreground_mask_or_box is not None:
            predicted_tensor = self._to_float_tensor(predicted_image)
            _, _, pred_h, pred_w = predicted_tensor.shape
            first_frame_mask = self._extract_first_frame_mask(foreground_mask_or_box)
            predicted_bbox = self._compute_bbox_from_mask(first_frame_mask, width=pred_w, height=pred_h)
            if predicted_bbox is not None:
                predicted_for_loss = self._crop_tensor_by_bbox(predicted_image, predicted_bbox)
                reference_bbox = self._compute_bbox_from_image_content(reference_image)
                if reference_bbox is None:
                    reference_tensor = self._to_float_tensor(reference_image)
                    _, _, ref_h, ref_w = reference_tensor.shape
                    normalized_bbox = self._normalized_bbox(predicted_bbox, width=pred_w, height=pred_h)
                    reference_bbox = self._denormalize_bbox(normalized_bbox, width=ref_w, height=ref_h)
                reference_for_loss = self._crop_tensor_by_bbox(reference_image, reference_bbox)

        # Optional debug dump for cropped inputs used by DINO loss.
        # Detach before converting to numpy so training gradients stay intact.
        if True: # os.getenv("DINO_VIEW_CONTROL_SAVE_DEBUG", "").lower() in {"1", "true", "yes"}:
            import cv2

            os.makedirs("visualization", exist_ok=True)
            predicted_for_loss_np = (
                predicted_for_loss.detach().clamp(0.0, 1.0).float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            ).astype(np.uint8)
            reference_for_loss_np = (
                self._to_float_tensor(reference_for_loss).detach().clamp(0.0, 1.0).float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            ).astype(np.uint8)
            cv2.imwrite("visualization/predicted_for_loss.png", predicted_for_loss_np[0][:, :, ::-1])
            cv2.imwrite("visualization/reference_for_loss.png", reference_for_loss_np[0][:, :, ::-1])
            # _extract_first_frame_mask returns 2D [H, W], not NCHW — visualize as single-channel uint8
            if foreground_mask_or_box is not None:
                m = self._extract_first_frame_mask(foreground_mask_or_box)
                if m is not None:
                    mask_u8 = (m.detach().clamp(0.0, 1.0).float().cpu().numpy() * 255.0).astype(np.uint8)
                    cv2.imwrite("visualization/foreground_mask_or_box.png", mask_u8)

        predicted_feature = self.encode_image(predicted_for_loss, requires_grad=True, device=device)
        with torch.no_grad():
            reference_feature = self.encode_image(reference_for_loss, requires_grad=False, device=device)
        return F.mse_loss(predicted_feature, reference_feature)
