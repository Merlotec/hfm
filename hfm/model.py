"""
Hierarchical Fluid Model (HFM)

Token vocabulary
----------------
patches  : [B, P, P, d_patch]       P = img_size // patch_px = 64
feat[l]  : [B, S_l, S_l, d_feat_l]  S_l ∈ {32,64,128,256}
sys[l]   : [B, S_l, S_l, d_sys_l]   same spatial grid, lower dim
resid    : [B, P, P, d_resid]        same grid as patches

Attention topology per transformer block
-----------------------------------------
patches  → local self-attn (radius r)  +  cross-attn to feat[l] covering same region
resid    → local self-attn (radius r)  +  cross-attn to sys[l] covering same region
feat[l]  → full self-attn at coarsest level (l=0: 32×32=1024 tokens)
           window self-attn at finer levels  +  cross-attn from/to feat[l-1]
           +  cross-attn to sys (all levels flattened)
           +  cross-attn to patches in spatial region
sys[l]   → same structure as feat[l], but values are only written when
           the residual stream is active (use a gated update)
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

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


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
    """4-channel image → [B, P, P, d_patch] grid of patch embeddings.

    No bias, no LayerNorm — both would break the left-inverse property with
    the paired PatchDecoder.  Per-token normalisation happens inside each
    transformer layer via PreNorm.
    """

    def __init__(self, in_channels: int = 4, patch_px: int = 4, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_px, stride=patch_px, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                                # [B, d_patch, P, P]
        return rearrange(x, 'b c h w -> b h w c')       # [B, P, P, d_patch]


class PatchDecoder(nn.Module):
    """[B, P, P, d_patch] → [B, in_channels, img_size, img_size].

    Uses the *transposed* convolution of the paired PatchEmbed (weight tying).
    Because Conv2d and ConvTranspose2d share the same weight tensor shape
    [d_patch, in_channels, kH, kW], passing the encoder weight directly to
    F.conv_transpose2d makes this the exact adjoint operation.

    Consequence: encode(decode(z)) == z when the weight matrix is orthogonal.
    The network is free to learn orthogonality; no explicit constraint is needed
    because the only consistent fixed point of the weight-tied pair is W^T W = I.
    """

    def __init__(self, encoder: "PatchEmbed"):
        super().__init__()
        self.encoder = encoder   # shared weight — no new parameters here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, P, d_patch]
        x = rearrange(x, 'b h w c -> b c h w')         # [B, d_patch, P, P]
        stride = self.encoder.proj.stride
        return F.conv_transpose2d(x, self.encoder.proj.weight, stride=stride)


# ---------------------------------------------------------------------------
# Feature / system hierarchy initialisation
# ---------------------------------------------------------------------------

class HierarchyInit(nn.Module):
    """
    Learnable initial feature/system embeddings at each pyramid level.
    Returns a list of [1, S, S, D] tensors (one per level).
    """

    def __init__(self, sizes: Tuple[int, ...], dims: Tuple[int, ...]):
        super().__init__()
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(1, s, s, d))
            for s, d in zip(sizes, dims)
        ])

    def forward(self, batch_size: int) -> List[torch.Tensor]:
        return [e.expand(batch_size, -1, -1, -1) for e in self.embeddings]


# ---------------------------------------------------------------------------
# Positional encoding (learnable per level)
# ---------------------------------------------------------------------------

class LearnedPos2D(nn.Module):
    def __init__(self, h: int, w: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, h, w, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


# ---------------------------------------------------------------------------
# Per-level transformer block for feature/system hierarchy
# ---------------------------------------------------------------------------

class FeatureLevelBlock(nn.Module):
    """
    One transformer depth-step for a single level of the feature hierarchy.

    Attention pattern for feature map tokens:
      1. self-attn within level (full if l==0, window otherwise)
      2. cross-attn ← coarser level   (hierarchical context)
      3. cross-attn ← sys same level  (system context)
      4. cross-attn ← patches         (spatial refinement)
    FFN
    """

    def __init__(self, level: int, d_feat: int, d_sys: int, d_patch: int,
                 n_heads: int, size: int, window_size: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.level = level
        self.size = size
        self.window_size = window_size
        use_full = (level == 0) or (size <= window_size)

        self.self_attn = (
            _FullSelfAttn(d_feat, n_heads, dropout=dropout) if use_full
            else WindowSelfAttention(d_feat, n_heads, window_size=window_size,
                                     dropout=dropout)
        )
        self.norm1 = nn.LayerNorm(d_feat)

        # Cross-attn from coarser level (only for l > 0)
        if level > 0:
            # Coarser-level tokens are upsampled to this level's size, then windowed
            self.cross_coarse = (
                CrossAttention(d_feat, d_feat, n_heads, dropout=dropout) if use_full
                else WindowedCrossAttention(d_feat, d_feat, n_heads,
                                            window_size=window_size, dropout=dropout)
            )
            self.norm_coarse = nn.LayerNorm(d_feat)

        # Cross-attn from sys (same spatial size → windowed when large)
        self.cross_sys = (
            CrossAttention(d_feat, d_sys, n_heads, dropout=dropout) if use_full
            else WindowedCrossAttention(d_feat, d_sys, n_heads,
                                        window_size=window_size, dropout=dropout)
        )
        self.norm_sys = nn.LayerNorm(d_feat)

        # Cross-attn from patches: patches are at 64×64, feature may be finer
        # → always use windowed so we never build a 65536×4096 attention map
        self.cross_patch = (
            CrossAttention(d_feat, d_patch, n_heads, dropout=dropout) if use_full
            else WindowedCrossAttention(d_feat, d_patch, n_heads,
                                        window_size=window_size, dropout=dropout)
        )
        self.norm_patch = nn.LayerNorm(d_feat)

        self.ffn = FeedForward(d_feat, mlp_ratio, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_feat)

    def forward(self, feat: torch.Tensor,
                sys_same: torch.Tensor,
                patches: torch.Tensor,
                coarser_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # feat / sys_same / patches / coarser_feat: all [B, H, W, D]
        x = feat
        x = x + self.self_attn(self.norm1(x))

        if self.level > 0 and coarser_feat is not None:
            x = x + self.cross_coarse(self.norm_coarse(x), coarser_feat)

        x = x + self.cross_sys(self.norm_sys(x), sys_same)
        x = x + self.cross_patch(self.norm_patch(x), patches)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SystemLevelBlock(nn.Module):
    """
    Transformer block for one level of the system embedding.

    Attention pattern:
      1. self-attn within level
      2. cross-attn ← feat same level
      3. cross-attn ← coarser sys  (hierarchy)
      4. cross-attn ← residual patches  (gated: only when residual active)
    FFN
    """

    def __init__(self, level: int, d_sys: int, d_feat: int, d_resid: int,
                 n_heads: int, size: int, window_size: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.level = level
        use_full = (level == 0) or (size <= window_size)

        self.self_attn = (
            _FullSelfAttn(d_sys, n_heads, dropout=dropout) if use_full
            else WindowSelfAttention(d_sys, n_heads, window_size=window_size,
                                     dropout=dropout)
        )
        self.norm1 = nn.LayerNorm(d_sys)

        self.cross_feat = (
            CrossAttention(d_sys, d_feat, n_heads, dropout=dropout) if use_full
            else WindowedCrossAttention(d_sys, d_feat, n_heads,
                                        window_size=window_size, dropout=dropout)
        )
        self.norm_feat = nn.LayerNorm(d_sys)

        if level > 0:
            self.cross_coarse = (
                CrossAttention(d_sys, d_sys, n_heads, dropout=dropout) if use_full
                else WindowedCrossAttention(d_sys, d_sys, n_heads,
                                            window_size=window_size, dropout=dropout)
            )
            self.norm_coarse = nn.LayerNorm(d_sys)

        # Gated residual cross-attn (residual always at patch resolution = fine)
        self.cross_resid = (
            CrossAttention(d_sys, d_resid, n_heads, dropout=dropout) if use_full
            else WindowedCrossAttention(d_sys, d_resid, n_heads,
                                        window_size=window_size, dropout=dropout)
        )
        self.norm_resid = nn.LayerNorm(d_sys)
        self.resid_gate = nn.Parameter(torch.zeros(1))

        self.ffn = FeedForward(d_sys, mlp_ratio, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_sys)

    def forward(self, sys: torch.Tensor,
                feat_same: torch.Tensor,
                residual: Optional[torch.Tensor],
                coarser_sys: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = sys
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_feat(self.norm_feat(x), feat_same)

        if self.level > 0 and coarser_sys is not None:
            x = x + self.cross_coarse(self.norm_coarse(x), coarser_sys)

        if residual is not None:
            gate = torch.sigmoid(self.resid_gate)
            x = x + gate * self.cross_resid(self.norm_resid(x), residual)

        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full self-attention helper (sequences, no spatial structure required)
# ---------------------------------------------------------------------------

class _FullSelfAttn(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.ndim == 4
        H = W = C = 0
        if spatial:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        else:
            B, _, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        nh, hd = self.n_heads, self.head_dim
        L = x.shape[1]
        q = q.reshape(B, L, nh, hd).transpose(1, 2)
        k = k.reshape(B, L, nh, hd).transpose(1, 2)
        v = v.reshape(B, L, nh, hd).transpose(1, 2)

        scale = math.sqrt(hd)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.drop(self.proj(out))

        if spatial:
            out = out.reshape(B, H, W, C)
        return out


# ---------------------------------------------------------------------------
# Patch transformer block
# ---------------------------------------------------------------------------

class PatchBlock(nn.Module):
    """
    Transformer block for image-level patch tokens.

    Attention:
      1. local self-attn (radius r)
      2. cross-attn ← feature hierarchy (tokens covering this patch's region)
    FFN
    """

    def __init__(self, d_patch: int, d_feat_finest: int, n_heads: int,
                 radius: int = 3, window_size: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.local_self = LocalSelfAttention(d_patch, n_heads, radius=radius,
                                             dropout=dropout)
        self.norm1 = nn.LayerNorm(d_patch)
        # Windowed cross-attn: patches are 64×64, feat_finest may be 256×256
        self.cross_feat = WindowedCrossAttention(d_patch, d_feat_finest, n_heads,
                                                  window_size=window_size,
                                                  dropout=dropout)
        self.norm2 = nn.LayerNorm(d_patch)
        self.ffn = FeedForward(d_patch, mlp_ratio, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_patch)

    def forward(self, patches: torch.Tensor,
                feat_finest: torch.Tensor) -> torch.Tensor:
        x = patches + self.local_self(self.norm1(patches))
        x = x + self.cross_feat(self.norm2(x), feat_finest)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Residual-patch transformer block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Transformer block for residual patch tokens.

    Attention:
      1. local self-attn among residual patches
      2. cross-attn ← system embedding (finest level)
    FFN
    """

    def __init__(self, d_resid: int, d_sys_finest: int, n_heads: int,
                 radius: int = 3, window_size: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.local_self = LocalSelfAttention(d_resid, n_heads, radius=radius,
                                             dropout=dropout)
        self.norm1 = nn.LayerNorm(d_resid)
        # Windowed cross-attn: resid patches are 64×64, sys_finest may be 256×256
        self.cross_sys = WindowedCrossAttention(d_resid, d_sys_finest, n_heads,
                                                window_size=window_size,
                                                dropout=dropout)
        self.norm2 = nn.LayerNorm(d_resid)
        self.ffn = FeedForward(d_resid, mlp_ratio, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_resid)

    def forward(self, resid: torch.Tensor,
                sys_finest: torch.Tensor) -> torch.Tensor:
        x = resid + self.local_self(self.norm1(resid))
        x = x + self.cross_sys(self.norm2(x), sys_finest)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# One full transformer layer (all token groups updated in parallel)
# ---------------------------------------------------------------------------

class HFMLayer(nn.Module):
    """
    A single depth step that updates all token groups simultaneously.

    The residual stream affects system embeddings via the gated cross-attn
    inside SystemLevelBlock. When residual is None the gate is inactive.
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        n_levels = cfg.n_levels

        # Feature blocks per level
        self.feat_blocks = nn.ModuleList([
            FeatureLevelBlock(
                level=l,
                d_feat=cfg.d_feat[l],
                d_sys=cfg.d_sys[l],
                d_patch=cfg.d_patch,
                n_heads=cfg.n_heads_feat[l],
                size=cfg.feat_sizes[l],
                window_size=cfg.feat_window_size,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
            )
            for l in range(n_levels)
        ])

        # System blocks per level
        self.sys_blocks = nn.ModuleList([
            SystemLevelBlock(
                level=l,
                d_sys=cfg.d_sys[l],
                d_feat=cfg.d_feat[l],
                d_resid=cfg.d_resid,
                n_heads=cfg.n_heads_sys[l],
                size=cfg.feat_sizes[l],
                window_size=cfg.feat_window_size,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
            )
            for l in range(n_levels)
        ])

        # Patch block (attends to finest feature level)
        self.patch_block = PatchBlock(
            d_patch=cfg.d_patch,
            d_feat_finest=cfg.d_feat[-1],
            n_heads=cfg.n_heads_patch,
            radius=cfg.patch_local_radius,
            window_size=cfg.feat_window_size,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

        # Residual block (attends to finest sys level)
        self.resid_block = ResidualBlock(
            d_resid=cfg.d_resid,
            d_sys_finest=cfg.d_sys[-1],
            n_heads=cfg.n_heads_patch,
            radius=cfg.patch_local_radius,
            window_size=cfg.feat_window_size,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

        # Downsample projections: feat[l] → feat[l-1] spatial resolution
        # Used so each level can receive context from adjacent coarser level.
        # We downsample spatially via adaptive average pool and project dim.
        self.feat_down_projs = nn.ModuleList([
            nn.Linear(cfg.d_feat[l], cfg.d_feat[l - 1])
            for l in range(1, n_levels)
        ])
        self.sys_down_projs = nn.ModuleList([
            nn.Linear(cfg.d_sys[l], cfg.d_sys[l - 1])
            for l in range(1, n_levels)
        ])

        # Upsample projections: feat[l-1] → feat[l] spatial resolution
        self.feat_up_projs = nn.ModuleList([
            nn.Linear(cfg.d_feat[l - 1], cfg.d_feat[l])
            for l in range(1, n_levels)
        ])
        self.sys_up_projs = nn.ModuleList([
            nn.Linear(cfg.d_sys[l - 1], cfg.d_sys[l])
            for l in range(1, n_levels)
        ])

        self.cfg = cfg

    # ---- helpers ----

    @staticmethod
    def _resize(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Bilinear resize [B, H, W, C] → [B, target_h, target_w, C]."""
        _, h, w, _ = t.shape
        if h == target_h and w == target_w:
            return t
        t = t.permute(0, 3, 1, 2)
        t = F.interpolate(t.float(), size=(target_h, target_w),
                          mode='bilinear', align_corners=False)
        return t.permute(0, 2, 3, 1)

    def _coarser_feat(self, feat: List[torch.Tensor],
                      level: int) -> Optional[torch.Tensor]:
        """Return feat[level-1] upsampled to feat[level]'s spatial size."""
        if level == 0:
            return None
        s = self.cfg.feat_sizes[level]
        upsampled = self._resize(feat[level - 1], s, s)
        return self.feat_up_projs[level - 1](upsampled)

    def _coarser_sys(self, sys: List[torch.Tensor],
                     level: int) -> Optional[torch.Tensor]:
        if level == 0:
            return None
        s = self.cfg.feat_sizes[level]
        upsampled = self._resize(sys[level - 1], s, s)
        return self.sys_up_projs[level - 1](upsampled)

    def _patches_for_level(self, patches: torch.Tensor,
                           level: int) -> torch.Tensor:
        """Downsample/pool patches to feature level spatial size."""
        s = self.cfg.feat_sizes[level]
        return self._resize(patches, s, s)

    def _resid_for_sys_level(self, resid: torch.Tensor,
                              level: int) -> torch.Tensor:
        s = self.cfg.feat_sizes[level]
        return self._resize(resid, s, s)

    # ---- forward ----

    def forward(
        self,
        patches: torch.Tensor,
        feat: List[torch.Tensor],
        sys: List[torch.Tensor],
        resid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
        """
        All inputs are spatial grids [B, H, W, D].
        Returns updated (patches, feat_list, sys_list, resid_or_none).
        """
        # ---- update feature levels (coarse→fine so coarser context is fresh) ----
        new_feat: List[torch.Tensor] = []
        for l in range(self.cfg.n_levels):
            coarser = self._coarser_feat(feat, l)
            patches_at_l = self._patches_for_level(patches, l)
            new_feat.append(self.feat_blocks[l](
                feat[l], sys[l], patches_at_l, coarser
            ))

        # ---- update system levels ----
        new_sys: List[torch.Tensor] = []
        for l in range(self.cfg.n_levels):
            coarser = self._coarser_sys(sys, l)
            resid_at_l = (self._resid_for_sys_level(resid, l)
                          if resid is not None else None)
            new_sys.append(self.sys_blocks[l](
                sys[l], feat[l], resid_at_l, coarser
            ))

        # ---- update patches (see finest feature level) ----
        new_patches = self.patch_block(patches, feat[-1])

        # ---- update residual (if present) ----
        new_resid: Optional[torch.Tensor] = None
        if resid is not None:
            new_resid = self.resid_block(resid, sys[-1])

        return new_patches, new_feat, new_sys, new_resid


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HFM(nn.Module):
    """
    Hierarchical Fluid Model.

    forward(x, sys_emb, resid) → (prediction, updated_sys_emb)

    Parameters
    ----------
    x        : [B, in_channels, img_size, img_size]
    sys_emb  : list of [B, S_l, S_l, d_sys_l] tensors, one per level.
               Pass None to use zeroed initial embeddings.
    resid    : [B, in_channels, img_size, img_size] or None
               Prediction error from the previous step.
    freeze_sys: If True the system embedding tensors are detached before
               being used as queries so gradients do not flow through them
               within this forward pass (but the tensors themselves still
               carry grad from earlier passes that built them).
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.patch_embed = PatchEmbed(cfg.in_channels, cfg.patch_px, cfg.d_patch)
        self.resid_embed = PatchEmbed(cfg.in_channels, cfg.patch_px, cfg.d_resid)

        # Positional encodings
        P = cfg.n_patch
        self.patch_pos = LearnedPos2D(P, P, cfg.d_patch)
        self.resid_pos = LearnedPos2D(P, P, cfg.d_resid)
        self.feat_pos = nn.ModuleList([
            LearnedPos2D(s, s, d)
            for s, d in zip(cfg.feat_sizes, cfg.d_feat)
        ])
        self.sys_pos = nn.ModuleList([
            LearnedPos2D(s, s, d)
            for s, d in zip(cfg.feat_sizes, cfg.d_sys)
        ])

        # Learnable initial feature embeddings (shared across batch, summed in)
        self.feat_init = nn.ParameterList([
            nn.Parameter(torch.zeros(1, s, s, d))
            for s, d in zip(cfg.feat_sizes, cfg.d_feat)
        ])

        # Transformer layers
        self.layers = nn.ModuleList([HFMLayer(cfg) for _ in range(cfg.n_layers)])

        # Decoder: shares weights with patch_embed (adjoint / ConvTranspose2d)
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

        # Patch and residual encoder weights are square (d_patch == in_ch * kH * kW),
        # so orthogonal init makes encode(decode(z)) == z exactly from step 0.
        for embed in (self.patch_embed, self.resid_embed):
            w = embed.proj.weight                        # [d_patch, in_ch, kH, kW]
            nn.init.orthogonal_(w.reshape(w.shape[0], -1))  # treat as 2-D matrix

    def init_sys_emb(self, batch_size: int,
                     device: torch.device) -> List[torch.Tensor]:
        """Return zero-initialised system embeddings for a new sequence."""
        return [
            torch.zeros(batch_size, s, s, d, device=device)
            for s, d in zip(self.cfg.feat_sizes, self.cfg.d_sys)
        ]

    @staticmethod
    def _checkpointed_layer(
        layer: nn.Module,
        patches: torch.Tensor,
        feat: List[torch.Tensor],
        sys: List[torch.Tensor],
        resid_tokens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
        """Run one HFMLayer with activation checkpointing.

        torch.utils.checkpoint requires flat tensor I/O, so we pack/unpack
        the list arguments.  A sentinel zero tensor is used in place of None
        residual so the function signature stays uniform.
        """
        from torch.utils.checkpoint import checkpoint
        n_f = len(feat)
        n_s = len(sys)
        has_resid = resid_tokens is not None
        r_in = resid_tokens if has_resid else patches.new_zeros(*patches.shape[:-1], 1)

        def fn(p, *args):
            f = list(args[:n_f])
            s = list(args[n_f:n_f + n_s])
            r = args[-1] if has_resid else None
            new_p, new_f, new_s, new_r = layer(p, f, s, r)
            r_out = new_r if new_r is not None else p.new_zeros(*p.shape[:-1], 1)
            return (new_p,) + tuple(new_f) + tuple(new_s) + (r_out,)

        out = checkpoint(fn, patches, *feat, *sys, r_in, use_reentrant=False)
        new_patches = out[0]
        new_feat    = list(out[1:1 + n_f])
        new_sys     = list(out[1 + n_f:1 + n_f + n_s])
        new_resid   = out[-1] if has_resid else None
        return new_patches, new_feat, new_sys, new_resid

    def forward(
        self,
        x: torch.Tensor,
        sys_emb: Optional[List[torch.Tensor]] = None,
        resid: Optional[torch.Tensor] = None,
        freeze_sys: bool = False,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        pixel_mask : (1, 1, H, W) bool — True for fluid pixels.
            A patch is kept if it contains at least one fluid pixel.
            Hole patch tokens are forced to zero after positional encoding
            and after every transformer layer so they cannot contaminate
            fluid tokens through attention.
        """
        B = x.shape[0]
        device = x.device

        # ---- derive patch-level hole mask ----
        # patch_mask[b, p, q, :] = 0 when the entire p×q patch is a hole.
        patch_mask: Optional[torch.Tensor] = None
        if pixel_mask is not None:
            pm = F.max_pool2d(pixel_mask.float(),
                              kernel_size=self.cfg.patch_px,
                              stride=self.cfg.patch_px)   # [1, 1, P, P]
            patch_mask = pm.permute(0, 2, 3, 1)           # [1, P, P, 1] broadcast over d

        # ---- encode inputs ----
        patches = self.patch_pos(self.patch_embed(x))       # [B, P, P, d_patch]
        if patch_mask is not None:
            patches = patches * patch_mask

        resid_tokens = None
        if resid is not None:
            resid_tokens = self.resid_pos(self.resid_embed(resid))
            if patch_mask is not None:
                resid_tokens = resid_tokens * patch_mask

        # ---- initialise feature hierarchy ----
        feat = [
            self.feat_pos[l](self.feat_init[l].expand(B, -1, -1, -1))
            for l in range(self.cfg.n_levels)
        ]

        # ---- initialise / apply system embeddings ----
        if sys_emb is None:
            sys_emb = self.init_sys_emb(B, device)

        if freeze_sys:
            # Detach so no gradient flows *through* sys within this forward.
            # Gradients still flow back to the ops that produced these tensors.
            sys = [s.detach() for s in sys_emb]
        else:
            sys = [self.sys_pos[l](sys_emb[l]) for l in range(self.cfg.n_levels)]

        # ---- transformer layers ----
        for layer in self.layers:
            if self.cfg.gradient_checkpointing and self.training:
                patches, feat, sys, resid_tokens = HFM._checkpointed_layer(
                    layer, patches, feat, sys, resid_tokens
                )
            else:
                patches, feat, sys, resid_tokens = layer(
                    patches, feat, sys, resid_tokens
                )
            # Re-zero hole patches after each layer: LayerNorm maps zero→beta
            # (the learned bias), which would otherwise contaminate neighbouring
            # fluid tokens through attention in the next layer.
            if patch_mask is not None:
                patches = patches * patch_mask
                if resid_tokens is not None:
                    resid_tokens = resid_tokens * patch_mask

        # ---- decode patches → prediction ----
        pred = self.decoder(patches)                        # [B, C, H, W]

        return pred, sys
