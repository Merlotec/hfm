"""
Hierarchical Fluid Model (HFM)

Token vocabulary
----------------
patches  : [B, P, P, d_patch]       P = img_size // patch_px = 64
feat[l]  : [B, S_l, S_l, d_feat_l]  S_l ∈ {32, 64, 128, 256}

Context tokens [B, K, d_ctx] from ContextEncoder are injected at every feat
level via cross-attention, replacing the old incremental sys/resid mechanism.

Layer schedule (hierarchical bottleneck)
-----------------------------------------
  layers 0 .. n_fine_layers-1       : all levels + patches  (encode)
  layers n_fine_layers .. -n_fine-1 : coarse levels only    (process)
  layers -n_fine_layers .. end      : all levels + patches  (decode)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional, Tuple

from .config import HFMConfig
from .attention import (
    LocalSelfAttention,
    WindowSelfAttention,
    CrossAttention,
    WindowedCrossAttention,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Patch encoder / decoder
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """[B, C, H, W] → [B, P, P, embed_dim] via strided Conv2d."""

    def __init__(self, in_channels: int = 4, patch_px: int = 4, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_px, stride=patch_px, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), 'b c h w -> b h w c')


class PatchDecoder(nn.Module):
    """[B, P, P, d_patch] → [B, C, H, W] via adjoint (weight-tied) ConvTranspose2d."""

    def __init__(self, encoder: PatchEmbed):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(
            rearrange(x, 'b h w c -> b c h w'),
            self.encoder.proj.weight,
            stride=self.encoder.proj.stride,
        )


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class LearnedPos2D(nn.Module):
    def __init__(self, h: int, w: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, h, w, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


# ---------------------------------------------------------------------------
# Full self-attention helper (sequences or spatial grids)
# ---------------------------------------------------------------------------

class _FullSelfAttn(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.ndim == 4
        if spatial:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        else:
            B, _, C = x.shape
            H = W = 0

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        nh, hd = self.n_heads, self.head_dim
        L = x.shape[1]
        q = q.reshape(B, L, nh, hd).transpose(1, 2)
        k = k.reshape(B, L, nh, hd).transpose(1, 2)
        v = v.reshape(B, L, nh, hd).transpose(1, 2)

        out = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        out = self.drop(self.proj(torch.matmul(out, v).transpose(1, 2).reshape(B, L, C)))

        return out.reshape(B, H, W, C) if spatial else out


# ---------------------------------------------------------------------------
# Per-level feature transformer block
# ---------------------------------------------------------------------------

class FeatureLevelBlock(nn.Module):
    """
    One transformer depth-step for a single level of the feature hierarchy.

    Attention pattern:
      1. self-attn within level
      2. cross-attn ← coarser level  (hierarchy)
      3. cross-attn ← context tokens [B, K, d_ctx]  (PDE dynamics)
      4. cross-attn ← patches        (spatial detail)
    FFN
    """

    def __init__(self, level: int, d_feat: int, d_ctx: int, d_patch: int,
                 n_heads: int, size: int, window_size: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.level    = level
        use_full      = (level == 0) or (size <= window_size)

        self.self_attn = (
            _FullSelfAttn(d_feat, n_heads, dropout=dropout) if use_full
            else WindowSelfAttention(d_feat, n_heads, window_size=window_size,
                                     dropout=dropout)
        )
        self.norm1 = nn.LayerNorm(d_feat)

        if level > 0:
            self.cross_coarse = (
                CrossAttention(d_feat, d_feat, n_heads, dropout=dropout) if use_full
                else WindowedCrossAttention(d_feat, d_feat, n_heads,
                                            window_size=window_size, dropout=dropout)
            )
            self.norm_coarse = nn.LayerNorm(d_feat)

        # Context cross-attn: feat tokens attend to flat [B, K, d_ctx] tokens.
        # Always full attention since K (n_ctx_tokens) is small.
        self.cross_ctx = CrossAttention(d_feat, d_ctx, n_heads, dropout=dropout)
        self.norm_ctx  = nn.LayerNorm(d_feat)

        self.cross_patch = (
            CrossAttention(d_feat, d_patch, n_heads, dropout=dropout) if use_full
            else WindowedCrossAttention(d_feat, d_patch, n_heads,
                                        window_size=window_size, dropout=dropout)
        )
        self.norm_patch = nn.LayerNorm(d_feat)

        self.ffn      = FeedForward(d_feat, mlp_ratio, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_feat)

    def forward(
        self,
        feat: torch.Tensor,
        context: torch.Tensor,
        patches: torch.Tensor,
        coarser_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = feat
        x = x + self.self_attn(self.norm1(x))
        if self.level > 0 and coarser_feat is not None:
            x = x + self.cross_coarse(self.norm_coarse(x), coarser_feat)
        x = x + self.cross_ctx(self.norm_ctx(x), context)
        x = x + self.cross_patch(self.norm_patch(x), patches)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Patch transformer block
# ---------------------------------------------------------------------------

class PatchBlock(nn.Module):
    """
    Transformer block for patch tokens.

    Attention:
      1. local self-attn (radius r)
      2. cross-attn ← finest feature level
    FFN
    """

    def __init__(self, d_patch: int, d_feat_finest: int, n_heads: int,
                 radius: int = 3, window_size: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.local_self = LocalSelfAttention(d_patch, n_heads, radius=radius,
                                             dropout=dropout)
        self.norm1     = nn.LayerNorm(d_patch)
        self.cross_feat = WindowedCrossAttention(d_patch, d_feat_finest, n_heads,
                                                  window_size=window_size,
                                                  dropout=dropout)
        self.norm2     = nn.LayerNorm(d_patch)
        self.ffn       = FeedForward(d_patch, mlp_ratio, dropout=dropout)
        self.norm_ffn  = nn.LayerNorm(d_patch)

    def forward(self, patches: torch.Tensor, feat_finest: torch.Tensor) -> torch.Tensor:
        x = patches + self.local_self(self.norm1(patches))
        x = x + self.cross_feat(self.norm2(x), feat_finest)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# One full transformer layer
# ---------------------------------------------------------------------------

class HFMLayer(nn.Module):
    """Updates patches and feature hierarchy, conditioned on context tokens."""

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        self.cfg = cfg
        n_levels = cfg.n_levels

        self.feat_blocks = nn.ModuleList([
            FeatureLevelBlock(
                level=l,
                d_feat=cfg.d_feat[l],
                d_ctx=cfg.d_ctx,
                d_patch=cfg.d_patch,
                n_heads=cfg.n_heads_feat[l],
                size=cfg.feat_sizes[l],
                window_size=cfg.feat_window_size,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
            )
            for l in range(n_levels)
        ])

        self.patch_block = PatchBlock(
            d_patch=cfg.d_patch,
            d_feat_finest=cfg.d_feat[-1],
            n_heads=cfg.n_heads_patch,
            radius=cfg.patch_local_radius,
            window_size=cfg.feat_window_size,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

        # Upsample projections: feat[l-1] → feat[l] for coarser context
        self.feat_up_projs = nn.ModuleList([
            nn.Linear(cfg.d_feat[l - 1], cfg.d_feat[l])
            for l in range(1, n_levels)
        ])

    @staticmethod
    def _resize(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if t.shape[1] == h and t.shape[2] == w:
            return t
        return F.interpolate(
            t.permute(0, 3, 1, 2).float(), size=(h, w), mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)

    def _coarser_feat(self, feat: List[torch.Tensor], level: int) -> Optional[torch.Tensor]:
        if level == 0:
            return None
        s = self.cfg.feat_sizes[level]
        return self.feat_up_projs[level - 1](self._resize(feat[level - 1], s, s))

    def _patches_for_level(self, patches: torch.Tensor, level: int) -> torch.Tensor:
        s = self.cfg.feat_sizes[level]
        return self._resize(patches, s, s)

    def forward(
        self,
        patches: torch.Tensor,
        feat: List[torch.Tensor],
        context: torch.Tensor,
        n_coarse_levels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        n_coarse_levels: when set, only update feat levels 0..n-1 and pass
            patches through unchanged (they act as skip connections).
        """
        n_active = n_coarse_levels if n_coarse_levels is not None else self.cfg.n_levels

        new_feat: List[torch.Tensor] = list(feat)
        for l in range(n_active):
            coarser = self._coarser_feat(new_feat, l)
            new_feat[l] = self.feat_blocks[l](
                feat[l], context, self._patches_for_level(patches, l), coarser
            )

        if n_coarse_levels is None:
            new_patches = self.patch_block(patches, new_feat[-1])
        else:
            new_patches = patches

        return new_patches, new_feat


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HFM(nn.Module):
    """
    Hierarchical Fluid Model.

    forward(x, context, pixel_mask) → predicted next frame [B, C, H, W]

    Parameters
    ----------
    x          : [B, in_channels, H, W]  current frame
    context    : [B, K, d_ctx]           summary tokens from ContextEncoder
    pixel_mask : [1, 1, H, W] bool, optional
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(cfg.in_channels, cfg.patch_px, cfg.d_patch)
        self.patch_pos   = LearnedPos2D(cfg.n_patch, cfg.n_patch, cfg.d_patch)
        self.mask_embed  = PatchEmbed(1, cfg.patch_px, cfg.d_patch)

        self.feat_init = nn.ParameterList([
            nn.Parameter(torch.zeros(1, s, s, d))
            for s, d in zip(cfg.feat_sizes, cfg.d_feat)
        ])
        self.feat_pos = nn.ModuleList([
            LearnedPos2D(s, s, d)
            for s, d in zip(cfg.feat_sizes, cfg.d_feat)
        ])

        self.layers  = nn.ModuleList([HFMLayer(cfg) for _ in range(cfg.n_layers)])
        self.decoder = PatchDecoder(self.patch_embed)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        w = self.patch_embed.proj.weight
        nn.init.orthogonal_(w.reshape(w.shape[0], -1))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]

        # Patch-level hole mask (1 = active patch, 0 = fully-hole patch)
        patch_mask: Optional[torch.Tensor] = None
        if pixel_mask is not None:
            pm = F.max_pool2d(pixel_mask.float(),
                              kernel_size=self.cfg.patch_px,
                              stride=self.cfg.patch_px)
            patch_mask = pm.permute(0, 2, 3, 1)   # [1, P, P, 1]

        # Encode input frame + geometry bias
        patches = self.patch_pos(self.patch_embed(x))
        if pixel_mask is not None:
            patches = patches + self.mask_embed(pixel_mask.float())
        if patch_mask is not None:
            patches = patches * patch_mask

        # Initialise feature hierarchy
        feat = [
            self.feat_pos[l](self.feat_init[l].expand(B, -1, -1, -1))
            for l in range(self.cfg.n_levels)
        ]

        # Transformer layers with hierarchical bottleneck schedule
        n_layers = len(self.layers)
        n_fine   = self.cfg.n_fine_layers
        coarse_n = self.cfg.n_coarse_levels if self.cfg.n_coarse_levels > 0 else None

        for i, layer in enumerate(self.layers):
            is_full  = (i < n_fine or i >= n_layers - n_fine or coarse_n is None)
            n_coarse = None if is_full else coarse_n

            if self.cfg.gradient_checkpointing and self.training:
                patches, feat = _checkpointed_layer(layer, patches, feat, context, n_coarse)
            else:
                patches, feat = layer(patches, feat, context, n_coarse_levels=n_coarse)

            if patch_mask is not None:
                patches = patches * patch_mask

        return self.decoder(patches)


def _checkpointed_layer(
    layer: HFMLayer,
    patches: torch.Tensor,
    feat: List[torch.Tensor],
    context: torch.Tensor,
    n_coarse: Optional[int],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    from torch.utils.checkpoint import checkpoint
    n_f = len(feat)
    nc_val = n_coarse if n_coarse is not None else -1

    def fn(p, ctx, nc_t, *feat_args):
        nc = None if nc_t.item() < 0 else int(nc_t.item())
        new_p, new_f = layer(p, list(feat_args), ctx, n_coarse_levels=nc)
        return (new_p,) + tuple(new_f)

    nc_tensor = patches.new_zeros(1).fill_(nc_val)
    out = checkpoint(fn, patches, context, nc_tensor, *feat, use_reentrant=False)
    assert out is not None
    return out[0], list(out[1:1 + n_f])
