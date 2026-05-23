"""
Fluid Model (HFM) — flat global-attention variant.

Token vocabulary
----------------
patches : [B, P², d_patch]           P = img_size // patch_px  (16)
global  : [B, n_global, d_patch]     learnable capacity tokens  (16)

All P²+n_global tokens are processed together with full self-attention and
2-D RoPE applied to patch positions.  Global tokens carry no spatial bias
(identity rotation).  Context tokens [B, K, d_ctx] from ContextEncoder are
injected at every layer via cross-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional

from .config import HFMConfig
from .attention import CrossAttention


# ---------------------------------------------------------------------------
# Shared building blocks
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
# Patch encoder
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """[B, C, H, W] → [B, P, P, embed_dim] via strided Conv2d."""

    def __init__(self, in_channels: int = 4, patch_px: int = 16, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_px, stride=patch_px, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), 'b c h w -> b h w c')


class SkipEncoder(nn.Module):
    """Single-scale skip features from the input frame for the decoder's post-conv."""

    def __init__(self, in_channels: int, skip_ch: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, skip_ch, 3, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self.stem(x)]


# ---------------------------------------------------------------------------
# Overlapping patch decoder
# ---------------------------------------------------------------------------

class OverlappingPatchDecoder(nn.Module):
    """
    [B, P_h, P_w, d] → [B, C, H, W] via 25% overlapping patches + tent-weighted averaging.

    Each patch token predicts a (3p/2 × 3p/2) region, extending p/4 pixels into
    each neighbour.  Overlapping contributions are blended with a bilinear tent
    kernel; the normalisation denominator is pre-computed.  A residual post-fold
    CNN provides sub-patch spatial refinement.
    """

    def __init__(self, d: int, out_channels: int, img_size: int, patch_size: int,
                 skip_ch: int = 0):
        super().__init__()
        assert patch_size % 4 == 0, "patch_size must be divisible by 4 for 25% overlap"
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.out_channels = out_channels
        self.skip_ch      = skip_ch
        kernel            = patch_size + patch_size // 2   # 3p/2 = 24 for p=16
        self.kernel       = kernel
        n                 = img_size // patch_size

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, out_channels * kernel * kernel),
        )

        mid = max(out_channels * 8, 64)
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_ch, mid, 3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, 3, padding=1, padding_mode='replicate'),
        )

        coords = torch.arange(kernel).float() + 0.5
        centre = kernel / 2.0
        w1d    = (centre - (coords - centre).abs()).clamp(min=0)
        wk_2d  = w1d.unsqueeze(1) * w1d.unsqueeze(0)
        self.register_buffer('weight_kernel', wk_2d)

        wk_flat  = wk_2d.flatten()
        norm_in  = wk_flat.unsqueeze(0).unsqueeze(-1).expand(1, -1, n * n)
        norm_map = F.fold(
            norm_in.contiguous(),
            output_size=(img_size, img_size),
            kernel_size=kernel,
            stride=patch_size,
            padding=patch_size // 4,
        )
        self.register_buffer('norm_map', norm_map)

    weight_kernel: torch.Tensor
    norm_map: torch.Tensor

    def forward(
        self,
        patches: torch.Tensor,
        skip_feats: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, ph, pw, d = patches.shape
        P      = ph * pw
        C      = self.out_channels
        kernel = self.kernel
        p      = self.patch_size

        preds   = self.head(patches.reshape(B, P, d))
        wk_flat = self.weight_kernel.flatten()
        preds   = preds.reshape(B, P, C, kernel * kernel) * wk_flat
        preds   = preds.permute(0, 2, 3, 1).reshape(B, C * kernel * kernel, P)

        output = F.fold(
            preds.contiguous(),
            output_size=(self.img_size, self.img_size),
            kernel_size=kernel,
            stride=p,
            padding=p // 4,
        )
        output = output / self.norm_map.clamp(min=1e-6)

        post_in = (
            torch.cat([output, skip_feats[0]], dim=1)
            if self.skip_ch > 0 and skip_feats
            else output
        )
        return output + self.post_conv(post_in)


# ---------------------------------------------------------------------------
# Learned 2-D positional bias (additive, for initial patch encoding)
# ---------------------------------------------------------------------------

class LearnedPos2D(nn.Module):
    def __init__(self, h: int, w: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, h, w, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


# ---------------------------------------------------------------------------
# 2-D Rotary Position Embedding helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap halves with sign flip: [..., a, b] → [..., -b, a]."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, n_heads, N, hd];  cos/sin: [1, 1, N, hd]."""
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Global self-attention with 2-D RoPE
# ---------------------------------------------------------------------------

class GlobalSelfAttn(nn.Module):
    """Full self-attention with rotary position bias."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x.shape
        nh, hd  = self.n_heads, self.head_dim

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, nh, hd).transpose(1, 2)   # [B, nh, N, hd]
        k = k.reshape(B, N, nh, hd).transpose(1, 2)
        v = v.reshape(B, N, nh, hd).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1
        )
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.drop(self.proj(out))


# ---------------------------------------------------------------------------
# One transformer layer
# ---------------------------------------------------------------------------

class HFMLayer(nn.Module):
    """Global self-attention → context cross-attention → FFN."""

    def __init__(self, d: int, n_heads: int, d_ctx: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1     = nn.LayerNorm(d)
        self.self_attn = GlobalSelfAttn(d, n_heads, dropout=dropout)
        self.norm2     = nn.LayerNorm(d)
        self.cross_ctx = CrossAttention(d, d_ctx, n_heads, dropout=dropout)
        self.norm3     = nn.LayerNorm(d)
        self.ffn       = FeedForward(d, mlp_ratio, dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = tokens + self.self_attn(self.norm1(tokens), cos, sin)
        x = x + self.cross_ctx(self.norm2(x), context)
        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HFM(nn.Module):
    """
    Fluid Model — global attention over patch + global tokens.

    forward(x, context, pixel_mask) → [B, C, H, W]

    x          : [B, in_channels, H, W]   current frame
    context    : [B, K, d_ctx]            summary from ContextEncoder
    pixel_mask : [1, 1, H, W] bool        geometry mask, optional
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        self.cfg = cfg
        P  = cfg.n_patch            # grid side (16 for 256px / 16px patches)
        hd = cfg.d_patch // cfg.n_heads

        self.patch_embed = PatchEmbed(cfg.in_channels, cfg.patch_px, cfg.d_patch)
        self.mask_embed  = PatchEmbed(1, cfg.patch_px, cfg.d_patch)
        self.skip_encoder = SkipEncoder(cfg.in_channels, cfg.skip_ch)

        # Learnable global capacity tokens
        self.global_tokens = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(1, cfg.n_global_tokens, cfg.d_patch), std=0.02
            )
        )

        # Pre-compute 2-D RoPE cos/sin for [patch tokens | global tokens].
        # Patch tokens at grid positions (i//P, i%P); global tokens use identity rotation.
        # Registered as buffers so they move with the model to any device.
        assert hd % 4 == 0, "d_patch // n_heads must be divisible by 4 for 2-D RoPE"
        quarter = hd // 4
        theta   = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float) / quarter))
        rows    = torch.arange(P).repeat_interleave(P).float()   # [P²]
        cols    = torch.arange(P).repeat(P).float()              # [P²]
        rf      = torch.outer(rows, theta)                        # [P², quarter]
        cf      = torch.outer(cols, theta)
        half_c  = torch.cat([rf.cos(), cf.cos()], dim=-1)        # [P², hd//2]
        half_s  = torch.cat([rf.sin(), cf.sin()], dim=-1)
        cos_p   = torch.cat([half_c, half_c], dim=-1)[None, None]  # [1, 1, P², hd]
        sin_p   = torch.cat([half_s, half_s], dim=-1)[None, None]
        ones_g  = torch.ones (1, 1, cfg.n_global_tokens, hd)
        zeros_g = torch.zeros(1, 1, cfg.n_global_tokens, hd)
        self.register_buffer('rope_cos', torch.cat([cos_p, ones_g],  dim=2))
        self.register_buffer('rope_sin', torch.cat([sin_p, zeros_g], dim=2))

        self.layers = nn.ModuleList([
            HFMLayer(cfg.d_patch, cfg.n_heads, cfg.d_ctx, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.decoder = OverlappingPatchDecoder(
            cfg.d_patch, cfg.in_channels, cfg.img_size, cfg.patch_px, cfg.skip_ch
        )

        self._init_weights()

    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

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
        for seq in [self.decoder.head, self.decoder.post_conv]:
            for child in reversed(list(seq.children())):
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    nn.init.zeros_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    break

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B       = x.shape[0]
        P       = self.cfg.n_patch
        n_patch = P * P

        # Perturb input during training for stochastic generation
        if self.training and self.cfg.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.cfg.noise_std

        skip_feats = self.skip_encoder(x)

        # Patch-level geometry mask  [1, P², 1]
        patch_mask: Optional[torch.Tensor] = None
        if pixel_mask is not None:
            pm = F.max_pool2d(pixel_mask.float(),
                              kernel_size=self.cfg.patch_px,
                              stride=self.cfg.patch_px)
            patch_mask = pm.reshape(pm.shape[0], -1, 1)

        # Encode input patches
        patches = self.patch_embed(x)                               # [B, P, P, d]
        if pixel_mask is not None:
            patches = patches + self.mask_embed(pixel_mask.float())
        patches = patches.reshape(B, n_patch, self.cfg.d_patch)     # [B, P², d]
        if patch_mask is not None:
            patches = patches * patch_mask

        # Concatenate global capacity tokens
        tokens = torch.cat(
            [patches, self.global_tokens.expand(B, -1, -1)], dim=1
        )   # [B, P² + n_global, d]

        # Transformer layers — hole patches start at zero but evolve freely via
        # attention, allowing the transformer to inpaint them and receive gradient
        # from the hole-filling loss.
        for layer in self.layers:
            if self.cfg.gradient_checkpointing and self.training:
                tokens = _checkpointed_layer(
                    layer, tokens, context, self.rope_cos, self.rope_sin
                )
            else:
                tokens = layer(tokens, context, self.rope_cos, self.rope_sin)

        # Decode from patch tokens only
        patch_tokens = tokens[:, :n_patch].reshape(B, P, P, self.cfg.d_patch)
        return self.decoder(patch_tokens, skip_feats)


def _checkpointed_layer(
    layer: nn.Module,
    tokens: torch.Tensor,
    context: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint
    out = checkpoint(layer, tokens, context, cos, sin, use_reentrant=False)
    assert out is not None
    return out  # type: ignore[return-value]
