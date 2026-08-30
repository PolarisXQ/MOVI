"""
Multiview IP-Adapter for Wan Video Model.

Injects multiview reference images through cross-attention, similar to IP-Adapter.
This provides semantic-level feature injection for better multiview consistency.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from einops import rearrange


class MultiviewIPAdapterImageProj(nn.Module):
    """
    Project multiview reference images to cross-attention features.
    
    Similar to IP-Adapter's image projection, but handles multiple views.
    """
    
    def __init__(
        self,
        cross_attention_dim: int = 1536,
        clip_embeddings_dim: int = 1280,
        num_tokens: int = 16,  # Number of tokens per view
        num_views: int = 4,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.num_views = num_views
        
        # Project CLIP embeddings to cross-attention dimension
        # Each view gets num_tokens tokens
        self.proj = nn.Linear(
            clip_embeddings_dim,
            num_tokens * cross_attention_dim
        )
        self.norm = nn.LayerNorm(cross_attention_dim)
        
        # Optional: View-specific projection for better view discrimination
        self.view_proj = nn.ModuleList([
            nn.Linear(cross_attention_dim, cross_attention_dim)
            for _ in range(num_views)
        ])
    
    def forward(
        self,
        image_embeds: torch.Tensor,  # (B, num_views, clip_embeddings_dim)
        view_indices: Optional[torch.Tensor] = None,  # (B, num_views)
    ) -> torch.Tensor:
        """
        Project image embeddings to cross-attention features.
        
        Args:
            image_embeds: CLIP embeddings of multiview images
                Shape: (B, num_views, clip_embeddings_dim)
            view_indices: Optional view indices for view-specific projection
        
        Returns:
            Projected features: (B, num_views * num_tokens, cross_attention_dim)
        """
        B, V, D = image_embeds.shape
        
        # Project each view
        # (B, V, D) -> (B*V, D) -> (B*V, num_tokens * cross_attention_dim)
        image_embeds_flat = image_embeds.view(B * V, D)
        projected = self.proj(image_embeds_flat)  # (B*V, num_tokens * cross_attention_dim)
        
        # Reshape and normalize
        # (B*V, num_tokens * cross_attention_dim) -> (B*V, num_tokens, cross_attention_dim)
        projected = projected.view(B * V, self.num_tokens, self.cross_attention_dim)
        projected = self.norm(projected)
        
        # Optional: Apply view-specific projection
        if view_indices is not None and self.view_proj is not None:
            view_projected = []
            for b in range(B):
                for v_idx in range(V):
                    view_idx = view_indices[b, v_idx].item() if view_indices.dim() > 1 else view_indices[v_idx].item()
                    view_idx = view_idx % len(self.view_proj)
                    view_feat = self.view_proj[view_idx](projected[b * V + v_idx])
                    view_projected.append(view_feat)
            projected = torch.stack(view_projected, dim=0)
            projected = projected.view(B, V, self.num_tokens, self.cross_attention_dim)
        else:
            projected = projected.view(B, V, self.num_tokens, self.cross_attention_dim)
        
        # Flatten views: (B, V, num_tokens, cross_attention_dim) -> (B, V*num_tokens, cross_attention_dim)
        projected = projected.view(B, V * self.num_tokens, self.cross_attention_dim)
        
        return projected


class MultiviewIPAdapterModule(nn.Module):
    """
    IP-Adapter module for a single transformer block.
    Converts projected image features to K, V for cross-attention.
    """
    
    def __init__(
        self,
        num_attention_heads: int = 24,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 1536,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        
        # Project to K, V
        self.to_k = nn.Linear(cross_attention_dim, num_attention_heads * attention_head_dim)
        self.to_v = nn.Linear(cross_attention_dim, num_attention_heads * attention_head_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert hidden states to K, V for cross-attention.
        
        Args:
            hidden_states: (B, num_tokens, cross_attention_dim)
        
        Returns:
            ip_k: (B, num_attention_heads, num_tokens, attention_head_dim)
            ip_v: (B, num_attention_heads, num_tokens, attention_head_dim)
        """
        ip_k = self.to_k(hidden_states)
        ip_v = self.to_v(hidden_states)
        
        # Reshape for multi-head attention
        ip_k = ip_k.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_attention_heads,
            self.attention_head_dim
        ).transpose(1, 2)  # (B, num_heads, num_tokens, head_dim)
        
        ip_v = ip_v.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_attention_heads,
            self.attention_head_dim
        ).transpose(1, 2)  # (B, num_heads, num_tokens, head_dim)
        
        return ip_k, ip_v


class MultiviewIPAdapter(nn.Module):
    """
    Multiview IP-Adapter for Wan Video Model.
    
    Injects multiview reference images through cross-attention in transformer blocks.
    """
    
    def __init__(
        self,
        num_attention_heads: int = 24,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 1536,
        clip_embeddings_dim: int = 1280,
        num_tokens: int = 16,
        num_views: int = 4,
        vace_layers: Tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}
        
        # Image projection
        self.image_proj = MultiviewIPAdapterImageProj(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            num_tokens=num_tokens,
            num_views=num_views,
        )
        
        # IP-Adapter modules for each VACE layer
        self.ipadapter_modules = nn.ModuleList([
            MultiviewIPAdapterModule(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(len(self.vace_layers))
        ])
    
    def forward(
        self,
        image_embeds: torch.Tensor,  # (B, num_views, clip_embeddings_dim)
        view_indices: Optional[torch.Tensor] = None,  # (B, num_views)
        scale: float = 1.0,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Forward pass: project images and generate K, V for each block.
        
        Args:
            image_embeds: CLIP embeddings of multiview reference images
            view_indices: View indices for view-specific processing
            scale: Scale factor for IP-Adapter injection
        
        Returns:
            Dictionary mapping block_id to {ip_k, ip_v, scale}
        """
        # Project images to cross-attention features
        hidden_states = self.image_proj(image_embeds, view_indices)  # (B, V*num_tokens, cross_attention_dim)
        
        # Generate K, V for each block
        ip_kv_dict = {}
        for block_id in self.vace_layers:
            ipadapter_id = self.vace_layers_mapping[block_id]
            ip_k, ip_v = self.ipadapter_modules[ipadapter_id](hidden_states)
            ip_kv_dict[block_id] = {
                "ip_k": ip_k,
                "ip_v": ip_v,
                "scale": scale
            }
        
        return ip_kv_dict


def create_multiview_ipadapter_from_config(
    config: Dict,
    vace_layers: Optional[Tuple[int, ...]] = None,
) -> MultiviewIPAdapter:
    """
    Create MultiviewIPAdapter from configuration.
    
    Args:
        config: Configuration dictionary
        vace_layers: Optional VACE layer indices
    
    Returns:
        MultiviewIPAdapter instance
    """
    if vace_layers is None:
        vace_layers = config.get(
            "vace_layers",
            (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)
        )
    
    return MultiviewIPAdapter(
        num_attention_heads=config.get("num_attention_heads", 24),
        attention_head_dim=config.get("attention_head_dim", 64),
        cross_attention_dim=config.get("cross_attention_dim", 1536),
        clip_embeddings_dim=config.get("clip_embeddings_dim", 1280),
        num_tokens=config.get("num_tokens", 16),
        num_views=config.get("num_views", 4),
        vace_layers=vace_layers,
    )


# =============================================================================
# Feature Bank Adapter: 3D Reference Cross-Attention Branch
# =============================================================================
# Encodes ALL multiview images through image encoder, forms a Feature Bank F_ref.
# Q: latent z_t (video generation state); K, V: from Feature Bank.
# Each frame's query attends to ALL viewpoints - allows "borrowing" from adjacent
# high-quality views when one view has artifacts (attention-based retrieval).
# =============================================================================


class FeatureBankImageProj(nn.Module):
    """
    Project full image encoder features (all tokens per view) to cross-attention dim.
    
    Unlike MultiviewIPAdapterImageProj which compresses each view to num_tokens,
    this keeps ALL patch tokens from each view, forming a rich Feature Bank.
    """
    
    def __init__(
        self,
        cross_attention_dim: int = 1536,
        clip_embeddings_dim: int = 1280,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        # Per-token projection: each patch token (1280) -> cross_attention_dim (1536)
        self.proj = nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
    
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project image embeddings to cross-attention features.
        
        Args:
            image_embeds: Full image encoder output
                Shape: (B, num_views * num_tokens_per_view, clip_embeddings_dim)
                e.g. (B, 4*257, 1280) for 4 views with 257 tokens each
        
        Returns:
            Projected features: (B, num_tokens, cross_attention_dim)
        """
        projected = self.proj(image_embeds)
        return self.norm(projected)


class MultiviewFeatureBankAdapter(nn.Module):
    """
    Feature Bank Adapter for Wan Video Model.
    
    Encodes ALL multiview reference images through image encoder, extracts full
    feature vectors per view, and forms a Feature Bank F_ref. This bank contains
    comprehensive geometric and texture information from all viewpoints.
    
    A new Cross-Attention branch (reusing the existing ip_kv injection mechanism):
    - Query (Q): Latent state z_t during video generation (frame t features)
    - Key (K) & Value (V): From Feature Bank F_ref
    
    Each frame's query attends to ALL viewpoints in the bank. When frame t should
    show the object's side but f_side has artifacts, attention can "borrow" from
    adjacent views (f_side±15°) or front view (f_front) for ID info.
    """
    
    def __init__(
        self,
        num_attention_heads: int = 24,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 1536,
        clip_embeddings_dim: int = 1280,
        vace_layers: Tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}
        
        self.image_proj = FeatureBankImageProj(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
        )
        
        self.fb_modules = nn.ModuleList([
            MultiviewIPAdapterModule(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(len(self.vace_layers))
        ])
    
    def forward(
        self,
        image_embeds: torch.Tensor,  # (B, num_views * num_tokens_per_view, clip_embeddings_dim)
        scale: float = 1.0,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Forward: project Feature Bank and generate K, V for each block.
        
        Args:
            image_embeds: Full encoder output, all views concatenated
                Shape: (B, num_views * 257, 1280) for 4 views with ViT 257 tokens
            scale: Scale factor for cross-attention injection
        
        Returns:
            Dictionary mapping block_id to {ip_k, ip_v, scale}
        """
        hidden_states = self.image_proj(image_embeds)  # (B, N, cross_attention_dim)
        
        ip_kv_dict = {}
        for block_id in self.vace_layers:
            mod_id = self.vace_layers_mapping[block_id]
            ip_k, ip_v = self.fb_modules[mod_id](hidden_states)
            ip_kv_dict[block_id] = {
                "ip_k": ip_k,
                "ip_v": ip_v,
                "scale": scale
            }
        return ip_kv_dict


def create_multiview_feature_bank_adapter_from_config(
    config: Dict,
    vace_layers: Optional[Tuple[int, ...]] = None,
) -> MultiviewFeatureBankAdapter:
    """Create MultiviewFeatureBankAdapter from configuration."""
    if vace_layers is None:
        vace_layers = config.get(
            "vace_layers",
            (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)
        )
    return MultiviewFeatureBankAdapter(
        num_attention_heads=config.get("num_attention_heads", 24),
        attention_head_dim=config.get("attention_head_dim", 64),
        cross_attention_dim=config.get("cross_attention_dim", 1536),
        clip_embeddings_dim=config.get("clip_embeddings_dim", 1280),
        vace_layers=vace_layers,
    )
