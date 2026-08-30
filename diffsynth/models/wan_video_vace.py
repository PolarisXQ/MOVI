import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional
from .wan_video_dit import DiTBlock
from .utils import hash_state_dict_keys


class ReferenceGatingModule(nn.Module):
    """
    Learned gating for reference patch features (方案1.1).
    gate = sigmoid(Linear(concat(x, mean(r))))
    r_modulated = gate * r

    Gate is initialized "closed" (sigmoid input large negative) so that at the start
    of training the ref patches contribute almost nothing, preserving pretrained
    temporal_concat behavior; training can then learn to open the gate where helpful.
    """
    def __init__(self, dim: int, eps: float = 1e-6, gate_init_bias: float = -5.0):
        super().__init__()
        self.dim = dim
        self.gate_init_bias = gate_init_bias
        # Input: concat(x_i, mean_ref) -> 2*dim, Output: scalar gate per position
        self.gate_linear = nn.Linear(dim * 2, dim)
        self.gate_out = nn.Linear(dim, 1)
        self._init_gate_closed()

    def _init_gate_closed(self):
        """Initialize gate so that sigmoid(gate_out(...)) ≈ 0 at start (ref patches muted)."""
        nn.init.zeros_(self.gate_out.weight)
        if self.gate_out.bias is not None:
            nn.init.constant_(self.gate_out.bias, self.gate_init_bias)
        # gate_out(x) = 0 * x + bias = bias -> sigmoid(bias) ≈ 0 when bias = -5

    def forward(
        self,
        c: torch.Tensor,  # (B, L, dim) - full vace context
        x: torch.Tensor,  # (B, L, dim) - main denoising features
        num_ref_patches: int,
    ) -> torch.Tensor:
        if num_ref_patches <= 0:
            return c
        B, L, D = c.shape
        c_ref = c[:, :num_ref_patches]  # (B, num_ref, dim)
        x_ref = x[:, :num_ref_patches]  # (B, num_ref, dim)
        mean_ref = c_ref.mean(dim=1, keepdim=True)  # (B, 1, dim)
        mean_ref = mean_ref.expand(-1, num_ref_patches, -1)  # (B, num_ref, dim)
        gate_in = torch.cat([x_ref, mean_ref], dim=-1)  # (B, num_ref, 2*dim)
        gate = torch.sigmoid(self.gate_out(torch.nn.functional.gelu(self.gate_linear(gate_in))))  # (B, num_ref, 1)
        c_modulated = c.clone()
        c_modulated[:, :num_ref_patches] = c_ref * gate
        return c_modulated


class ViewAwareReferenceAttention(nn.Module):
    """
    View-aware cross-attention over references (方案1.2).
    query: current (video) patch features
    key/value: reference patch features
    output: weighted ref info, residual added to video patches

    o_proj is zero-initialized so that at the start of training the residual is 0
    (no change to video patches), preserving pretrained behavior.
    """
    def __init__(self, dim: int, num_heads: int = 8, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self._init_output_zero()

    def _init_output_zero(self):
        """Zero-init o_proj so residual added to video patches is 0 at start."""
        nn.init.zeros_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        c: torch.Tensor,  # (B, L, dim)
        num_ref_patches: int,
    ) -> torch.Tensor:
        if num_ref_patches <= 0:
            return c
        B, L, D = c.shape
        num_video_patches = L - num_ref_patches
        if num_video_patches <= 0:
            return c
        c_ref = c[:, :num_ref_patches]   # (B, num_ref, dim)
        c_video = c[:, num_ref_patches:]  # (B, num_video, dim)
        q = self.q_proj(c_video).view(B, num_video_patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(c_ref).view(B, num_ref_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(c_ref).view(B, num_ref_patches, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, num_video_patches, D)
        out = self.o_proj(out)
        c_out = c.clone()
        c_out[:, num_ref_patches:] = c_video + out
        return c_out


class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    allow_missing_keys = True
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
        use_ref_gating: bool = False,
        use_ref_attention: bool = False,
        use_multiscale_vace: bool = True,
        multiscale_spatial_scales=(1, 2, 4),
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}
        self.dim = dim
        self.use_ref_gating = use_ref_gating
        self.use_ref_attention = use_ref_attention
        self.use_multiscale_vace = bool(use_multiscale_vace)
        self.multiscale_spatial_scales = tuple(sorted({int(s) for s in multiscale_spatial_scales if int(s) >= 1}))
        if len(self.multiscale_spatial_scales) == 0:
            self.multiscale_spatial_scales = (1,)

        # Reference gating and view-aware attention (approach 1).
        self.ref_gating_module = ReferenceGatingModule(dim, eps) if use_ref_gating else None
        self.ref_attention_module = ViewAwareReferenceAttention(dim, num_heads=min(8, num_heads), eps=eps) if use_ref_attention else None
        self._num_heads = num_heads

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])
        # ControlNet blocks (initialized lazily to avoid extra memory unless used)
        self.control_blocks = None

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.control_patch_embedding = None
        # FPN-like multi-scale fusion in VACE context encoding (zero-init: compatible with old ckpt).
        self.vace_multiscale_fusion = nn.ModuleList()
        self.control_multiscale_fusion = nn.ModuleList()
        if self.use_multiscale_vace:
            for _ in self.multiscale_spatial_scales[1:]:
                vace_fuser = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
                control_fuser = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
                nn.init.zeros_(vace_fuser.weight)
                nn.init.zeros_(vace_fuser.bias)
                nn.init.zeros_(control_fuser.weight)
                nn.init.zeros_(control_fuser.bias)
                self.vace_multiscale_fusion.append(vace_fuser)
                self.control_multiscale_fusion.append(control_fuser)

        # Cache patch embedding config for control branch creation.
        # NOTE: When VRAM management is enabled, `vace_patch_embedding` may be wrapped
        # by `AutoWrappedModule` and won't expose Conv3d attributes (out_channels/weight/etc).
        # These cached values are safe to use to construct the control patch embedding.
        self._vace_patch_cfg = {
            "out_channels": dim,
            "kernel_size": patch_size,
            "stride": patch_size,
        }

        # ControlNet-style zero layers (one per VACE layer)
        self.controlnet_zero_linears = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim) for _ in self.vace_layers]
        )
        for layer in self.controlnet_zero_linears:
            torch.nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        
        # Track whether training information has been printed to avoid duplicate output.
        self._training_info_printed = False

    def add_ref_gating_modules(self, use_gating: bool = True, use_attention: bool = True):
        """Inject ref gating/attention modules on demand (e.g., when loading pretrained VACE)."""
        dim = getattr(self, "dim", self.vace_blocks[0].after_proj.in_features)
        num_heads = getattr(self, "_num_heads", 12)
        if use_gating and self.ref_gating_module is None:
            self.ref_gating_module = ReferenceGatingModule(dim)
            self.use_ref_gating = True
        if use_attention and self.ref_attention_module is None:
            self.ref_attention_module = ViewAwareReferenceAttention(
                dim, num_heads=min(8, num_heads)
            )
            self.use_ref_attention = True

    def init_missing_params(self):
        # Ensure zero-layer params are initialized after meta -> real tensor materialization
        for layer in self.controlnet_zero_linears:
            if layer.weight.is_meta:
                layer.weight = torch.nn.Parameter(layer.weight.to_empty(device=layer.weight.device))
            torch.nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                if layer.bias.is_meta:
                    layer.bias = torch.nn.Parameter(layer.bias.to_empty(device=layer.bias.device))
                torch.nn.init.zeros_(layer.bias)

    def _ensure_control_blocks(self):
        if self.control_blocks is None:
            # Deep-copy base blocks as ControlNet branch blocks (standard ControlNet pattern)
            self.control_blocks = torch.nn.ModuleList([copy.deepcopy(b) for b in self.vace_blocks])
            # Put in eval by default; training script can set requires_grad / train mode as needed
            self.control_blocks.eval()

    @staticmethod
    def _unwrap_vram_managed_module(m: torch.nn.Module) -> torch.nn.Module:
        """
        VRAM management may wrap modules (e.g. AutoWrappedModule) and hide attributes like
        `.weight`, `.bias`, `.out_channels`. The wrapped module is stored in `.module`.
        """
        return getattr(m, "module", m)

    def _get_control_patch_embedding(self, in_channels, device, dtype):
        rebuild = (
            self.control_patch_embedding is None or
            self.control_patch_embedding.in_channels != in_channels
        )
        if rebuild:
            vace_patch = self._unwrap_vram_managed_module(self.vace_patch_embedding)
            self.control_patch_embedding = torch.nn.Conv3d(
                in_channels,
                # Prefer reading from the real Conv3d; fallback to cached cfg.
                getattr(vace_patch, "out_channels", self._vace_patch_cfg["out_channels"]),
                kernel_size=getattr(vace_patch, "kernel_size", self._vace_patch_cfg["kernel_size"]),
                stride=getattr(vace_patch, "stride", self._vace_patch_cfg["stride"]),
                padding=getattr(vace_patch, "padding", 0),
                dilation=getattr(vace_patch, "dilation", 1),
                groups=getattr(vace_patch, "groups", 1),
                bias=getattr(vace_patch, "bias", None) is not None,
            )
            # Initialize control branch with informative non-zero weights.
            # Use Kaiming init, then copy overlap from main VACE patch embedding.
            torch.nn.init.kaiming_uniform_(self.control_patch_embedding.weight, a=2.23606797749979)  # sqrt(5)
            if self.control_patch_embedding.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.control_patch_embedding.weight)
                bound = 1 / fan_in ** 0.5
                torch.nn.init.uniform_(self.control_patch_embedding.bias, -bound, bound)
            with torch.no_grad():
                # Copy overlap weights/bias from the real Conv3d if accessible
                src_w = getattr(vace_patch, "weight", None)
                if src_w is not None:
                    copy_in = min(src_w.shape[1], self.control_patch_embedding.weight.shape[1])
                    self.control_patch_embedding.weight[:, :copy_in].copy_(src_w[:, :copy_in])
                src_b = getattr(vace_patch, "bias", None)
                if self.control_patch_embedding.bias is not None and src_b is not None:
                    self.control_patch_embedding.bias.copy_(src_b)
        self.control_patch_embedding = self.control_patch_embedding.to(device=device, dtype=dtype)
        return self.control_patch_embedding

    def _encode_vace_context(self, vace_context, x, patch_embedding):
        is_control_branch = patch_embedding is self.control_patch_embedding
        fusion_layers = (
            self.control_multiscale_fusion if is_control_branch else self.vace_multiscale_fusion
        )
        c = []
        for u in vace_context:
            u = u.unsqueeze(0)  # [1, C, T, H, W]
            base_feature = patch_embedding(u)
            if self.use_multiscale_vace and len(self.multiscale_spatial_scales) > 1:
                fused_feature = base_feature
                for scale_idx, spatial_scale in enumerate(self.multiscale_spatial_scales[1:]):
                    pooled_u = F.avg_pool3d(
                        u,
                        kernel_size=(1, spatial_scale, spatial_scale),
                        stride=(1, spatial_scale, spatial_scale),
                        ceil_mode=True,
                    )
                    scale_feature = patch_embedding(pooled_u)
                    scale_feature = F.interpolate(
                        scale_feature,
                        size=base_feature.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                    fused_feature = fused_feature + fusion_layers[scale_idx](scale_feature)
                c.append(fused_feature)
            else:
                c.append(base_feature)
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        return c

    def _print_training_info(self, control_train_mode: str):
        """打印当前训练模式和哪些参数会回传梯度"""
        is_zero_layer_mode = (control_train_mode == "zero_only")
        print(f"[VaceWanModel] Control训练模式: {control_train_mode}")
        print(f"[VaceWanModel] 是否进入zero-layer模式: {is_zero_layer_mode}")
        
        # Check the requires_grad state of each component.
        trainable_parts = []
        
        # Main VACE branch (always trained).
        main_vace_params = sum(1 for p in self.vace_blocks.parameters() if p.requires_grad)
        if main_vace_params > 0:
            trainable_parts.append(f"主VACE分支 (vace_blocks): {main_vace_params} 个参数")
        
        # Zero layers (always trained).
        zero_layer_params = sum(1 for p in self.controlnet_zero_linears.parameters() if p.requires_grad)
        if zero_layer_params > 0:
            trainable_parts.append(f"Zero-layers (controlnet_zero_linears): {zero_layer_params} 个参数")
        
        # Control branch.
        if self.control_blocks is not None:
            control_block_params = sum(1 for p in self.control_blocks.parameters() if p.requires_grad)
            if control_block_params > 0:
                trainable_parts.append(f"Control分支blocks (control_blocks): {control_block_params} 个参数")
            else:
                trainable_parts.append(f"Control分支blocks (control_blocks): 冻结 (requires_grad=False)")
        
        # Control patch embedding
        if self.control_patch_embedding is not None:
            control_patch_params = sum(1 for p in self.control_patch_embedding.parameters() if p.requires_grad)
            if control_patch_params > 0:
                trainable_parts.append(f"Control patch embedding: {control_patch_params} 个参数")
            else:
                trainable_parts.append(f"Control patch embedding: 冻结 (requires_grad=False)")
        
        print(f"[VaceWanModel] 会回传梯度的部分:")
        for part in trainable_parts:
            print(f"  - {part}")
        
        if is_zero_layer_mode:
            print(f"[VaceWanModel] ⚠️ Zero-layer模式: Control分支在no_grad()中运行，只有zero-layers会回传梯度")
        else:
            print(f"[VaceWanModel] ✓ ControlNet模式: Control分支完整训练")

    def _run_vace_blocks(
        self,
        blocks,
        c,
        x,
        context,
        t_mod,
        freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block in blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c, x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                c = block(c, x, context, t_mod, freqs)
        return torch.unbind(c)[:-1]

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        control_context=None,
        control_scale: float = 1.0,
        control_train_mode: str = "zero_only",  # "zero_only" (no_grad control branch) or "controlnet"
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        num_ref_patches: Optional[int] = None,
    ):
        c = self._encode_vace_context(vace_context, x, self.vace_patch_embedding)
        # Approach 1: reference gating and view-aware attention.
        if num_ref_patches is not None and num_ref_patches > 0:
            if self.ref_gating_module is not None:
                self.ref_gating_module.to(dtype=c.dtype, device=c.device)
                c = self.ref_gating_module(c, x, num_ref_patches)
            if self.ref_attention_module is not None:
                self.ref_attention_module.to(dtype=c.dtype, device=c.device)
                c = self.ref_attention_module(c, num_ref_patches)
        hints = self._run_vace_blocks(
            self.vace_blocks, c, x, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
        if control_context is None or control_scale == 0:
            return hints
        else:   
            self._ensure_control_blocks()
            
            # Print training information only on the first call.
            if not self._training_info_printed:
                self._print_training_info(control_train_mode)
                self._training_info_printed = True
        
        control_patch_embedding = self._get_control_patch_embedding(
            in_channels=control_context.shape[1],
            device=control_context.device,
            dtype=control_context.dtype,
        )
        control_c = self._encode_vace_context(control_context, x, control_patch_embedding)
        if control_train_mode == "controlnet":
            control_hints = self._run_vace_blocks(
                self.control_blocks, control_c, x, context, t_mod, freqs,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        else:
            # Default: keep control branch forward but avoid storing activations to reduce CPU/GPU memory.
            with torch.no_grad():
                control_hints = self._run_vace_blocks(
                    self.control_blocks, control_c, x, context, t_mod, freqs,
                    use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False,
                )
        
        fused_hints = []
        for idx, (hint, control_hint) in enumerate(zip(hints, control_hints)):
            fused_hints.append(hint + self.controlnet_zero_linears[idx](control_hint) * control_scale)
        return fused_hints
    
    @staticmethod
    def state_dict_converter():
        return VaceWanModelDictConverter()
    
    
class VaceWanModelDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("vace")}
        if hash_state_dict_keys(state_dict_) == '3b2726384e4f64837bdf216eea3f310d': # vace 14B
            config = {
                "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35),
                "vace_in_dim": 96,
                "patch_size": (1, 2, 2),
                "has_image_input": False,
                "dim": 5120,
                "num_heads": 40,
                "ffn_dim": 13824,
                "eps": 1e-06,                
            }
        else:
            config = {}
        return state_dict_, config
