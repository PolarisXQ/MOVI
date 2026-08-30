"""
Cross-modal consistency check: CLIP-based text–image consistency score.

When 3D-rendered multiview reference images are corrupted (e.g. mesh collapse),
their CLIP embeddings drift from the text description. This module computes a
consistency score in [0, 1]; low scores can be used to down-weight the reference
path and rely more on text conditioning.
"""

import os
from typing import List, Optional, Union

import torch


def _resolve_clip_model_path(model_name: str) -> str:
    """
    If model_name is a HuggingFace cache root (has snapshots/ subdir but no
    pytorch_model.bin / model.safetensors in the root), resolve to a snapshot
    revision path that contains both model weights and preprocessor_config.json
    so CLIPModel and CLIPProcessor can both load.
    """
    if not os.path.isdir(model_name):
        return model_name
    snapshots_dir = os.path.join(model_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_name
    # Check if root already has model files (e.g. manually copied)
    for name in ("pytorch_model.bin", "model.safetensors", "config.json"):
        if os.path.isfile(os.path.join(model_name, name)):
            return model_name
    # Prefer a snapshot that has both model and preprocessor (required by CLIPProcessor)
    try:
        revs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    except OSError:
        return model_name
    if not revs:
        return model_name
    for rev in revs:
        resolved = os.path.join(snapshots_dir, rev)
        has_model = (
            os.path.isfile(os.path.join(resolved, "pytorch_model.bin"))
            or os.path.isfile(os.path.join(resolved, "model.safetensors"))
        )
        has_processor = os.path.isfile(os.path.join(resolved, "preprocessor_config.json"))
        if has_model and has_processor:
            return resolved
    # Fallback to first revision (e.g. model-only snapshot)
    return os.path.join(snapshots_dir, revs[0])


def compute_clip_text_image_consistency(
    text_embeddings: torch.Tensor,  # (1, D) or (B, D)
    image_embeddings: torch.Tensor,  # (N, D) or (B, N, D)
    temperature: float = 0.07,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Compute consistency score(s) between text and image CLIP embeddings.

    Args:
        text_embeddings: Text features, shape (1, D) or (B, D).
        image_embeddings: Image features, shape (N, D) or (B, N, D) for N images.
        temperature: Softmax temperature for scaling logits (default 0.07 as in CLIP).
        reduce: "mean" to average over images, "min" to use worst view.

    Returns:
        Consistency score in [0, 1]. Shape (1,) or (B,) depending on input.
    """
    if text_embeddings.dim() == 2 and image_embeddings.dim() == 2:
        # (1, D) @ (N, D).T -> (1, N)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        logits = (text_embeddings @ image_embeddings.T) / temperature
        if reduce == "mean":
            score = logits.squeeze(0).mean()
        elif reduce == "min":
            score = logits.squeeze(0).min()
        else:
            raise ValueError(f"Invalid reduce: {reduce}")
        # Map logits to [0, 1] via sigmoid (positive logit -> high score)
        score = torch.sigmoid(score).clamp(0.0, 1.0)
        return score
    elif text_embeddings.dim() == 2 and image_embeddings.dim() == 3:
        B, N, D = image_embeddings.shape
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        # (1, D) @ (B, N, D).T -> (1, B, N) -> (B, N)
        logits = (text_embeddings.unsqueeze(0) @ image_embeddings.permute(0, 2, 1)).squeeze(0) / temperature
        if reduce == "mean":
            score = logits.mean(dim=1)
        else:
            score = logits.min(dim=1).values
        score = torch.sigmoid(score).clamp(0.0, 1.0)
        return score
    else:
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        logits = (text_embeddings @ image_embeddings.T) / temperature
        if reduce == "mean":
            score = logits.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        else:
            score = logits.diagonal(dim1=-2, dim2=-1).min(dim=-1).values
        score = torch.sigmoid(score).clamp(0.0, 1.0)
        return score

class CLIPConsistencyChecker:
    """
    Uses a CLIP model to encode text and images and compute consistency score.
    Default is Long-CLIP (248 tokens) so long prompts are not truncated.
    Use a smaller model (e.g. openai/clip-vit-base-patch32, 77 tokens) via
    CLIP_CONSISTENCY_MODEL_PATH if you prefer.
    """

    def __init__(
        self,
        model_name: str = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if not model_name:
            model_name = os.getenv("CLIP_CONSISTENCY_MODEL_PATH") or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models",
                "LongCLIP-GmP-ViT-L-14",
            )
        self.model_name = model_name
        self._device = device
        self._dtype = dtype
        self._model = None
        self._processor = None

    def _ensure_loaded(self, device: Optional[torch.device] = None):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "CLIP consistency check requires transformers. "
                "Install with: pip install transformers"
            )
        path = _resolve_clip_model_path(self.model_name)
        self._model = CLIPModel.from_pretrained(path)
        self._processor = CLIPProcessor.from_pretrained(path)
        dev = device or self._device or next(self._model.parameters()).device
        self._model = self._model.to(dev)
        if self._dtype is not None:
            self._model = self._model.to(self._dtype)
        self._model.eval()

    @torch.no_grad()
    def compute_consistency(
        self,
        prompt: str,
        images: List,
        device: Optional[Union[str, torch.device]] = None,
        reduce: str = "mean",
        temperature: float = 0.07,
    ) -> float:
        """
        Compute text–image consistency. Uses the model's max_position_embeddings
        (e.g. 248 for Long-CLIP), so long prompts are not truncated.
        """
        if not prompt or not images:
            return 1.0
        self._ensure_loaded(device)
        dev = next(self._model.parameters()).device
        max_length = getattr(self._model.config, "max_position_embeddings", 77)
        inputs = self._processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if self._dtype is not None:
            inputs = {k: v.to(self._dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in inputs.items()}
        outputs = self._model(**inputs)
        text_emb = outputs.text_embeds
        image_emb = outputs.image_embeds
        score = compute_clip_text_image_consistency(
            text_emb, image_emb, temperature=temperature, reduce=reduce
        )
        return score.item()
