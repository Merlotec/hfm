"""
ContextEncoder: learns the underlying PDE dynamics from a sequence of frames.

Takes T frames and produces K summary tokens that are fed as conditioning
context into the main HFM prediction model.  The K tokens are produced by
learned query vectors (Perceiver-style) that cross-attend to all T·P²
spatiotemporal frame tokens, acting as a differentiable summary of the
governing equations for this sequence.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Optional

from .config import HFMConfig
from .model import PatchEmbed, LearnedPos2D, FeedForward


class _ContextBlock(nn.Module):
    """Standard full self-attention transformer block."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, mlp_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ContextEncoder(nn.Module):
    """
    Encodes T sequential frames → K summary context tokens [B, K, d_ctx].

    Parameters
    ----------
    cfg : HFMConfig

    Forward inputs
    --------------
    frames      : list of T tensors [B, C, H, W]
    pixel_mask  : [1, 1, H, W] bool, optional — shared geometry encoding

    Returns
    -------
    [B, K, d_ctx]
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        P = cfg.img_size // cfg.ctx_patch_px    # patches per side in context encoder

        # Instance-dict lookup, NOT getattr: dataclass defaults are CLASS
        # attributes, so getattr on a cfg pickled before this field existed
        # would silently return the new default (True) and build a 9-channel
        # embed that old 5-channel checkpoints cannot load.  Only a cfg whose
        # own __dict__ carries the field (i.e. constructed by current code)
        # enables the diff channels.
        self.use_diffs = bool(vars(cfg).get('ctx_temporal_diffs', False))
        in_ch = (2 * cfg.in_channels if self.use_diffs else cfg.in_channels) + 1
        self.patch_embed  = PatchEmbed(in_ch, cfg.ctx_patch_px, cfg.d_ctx)
        self.spatial_pos  = LearnedPos2D(P, P, cfg.d_ctx)
        self.temporal_pos = nn.Embedding(64, cfg.d_ctx)   # supports up to 64 input frames

        self.layers = nn.ModuleList([
            _ContextBlock(cfg.d_ctx, cfg.n_ctx_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_ctx_layers)
        ])

        # K learned query tokens; Perceiver-style cross-attn produces the summary
        self.summary_tokens   = nn.Parameter(torch.zeros(1, cfg.n_ctx_tokens, cfg.d_ctx))
        self.summary_cross    = nn.MultiheadAttention(
            cfg.d_ctx, cfg.n_ctx_heads, batch_first=True, dropout=cfg.dropout
        )
        self.summary_norm_q   = nn.LayerNorm(cfg.d_ctx)
        self.summary_norm_kv  = nn.LayerNorm(cfg.d_ctx)
        self.summary_ffn      = FeedForward(cfg.d_ctx, cfg.mlp_ratio, cfg.dropout)
        self.summary_norm_ffn = nn.LayerNorm(cfg.d_ctx)

        self.cfg = cfg
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.summary_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        frames: List[torch.Tensor],
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = frames[0].shape[0]

        if pixel_mask is not None:
            mask_ch = pixel_mask.float().expand(B, 1, frames[0].shape[2], frames[0].shape[3])
        else:
            mask_ch = torch.ones(B, 1, frames[0].shape[2], frames[0].shape[3],
                                 device=frames[0].device, dtype=frames[0].dtype)

        tokens: List[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None
        for t, frame in enumerate(frames):
            if pixel_mask is not None:
                frame = frame * pixel_mask
            if self.use_diffs:
                # Explicit inter-frame motion channel: [f_t, f_t - f_{t-1}].
                # The diff's magnitude/structure scales with the temporal stride,
                # making the timestep readable at the very first conv instead of
                # having to be disentangled from appearance deep in the trunk.
                diff = frame - prev if prev is not None else torch.zeros_like(frame)
                prev = frame
                frame_aug = torch.cat([frame, diff, mask_ch], dim=1)
            else:
                frame_aug = torch.cat([frame, mask_ch], dim=1)
            tok = self.spatial_pos(self.patch_embed(frame_aug))  # [B, P, P, d_ctx]
            tok = rearrange(tok, 'b h w d -> b (h w) d')         # [B, P², d_ctx]
            tok = tok + self.temporal_pos.weight[t]            # broadcast temporal bias
            tokens.append(tok)

        x = torch.cat(tokens, dim=1)                           # [B, T·P², d_ctx]

        for layer in self.layers:
            x = layer(x)

        # Summary: K queries cross-attend to all frame tokens
        q  = self.summary_tokens.expand(B, -1, -1)
        kv = self.summary_norm_kv(x)
        ctx_out, _ = self.summary_cross(
            self.summary_norm_q(q), kv, kv, need_weights=False
        )
        q = q + ctx_out
        q = q + self.summary_ffn(self.summary_norm_ffn(q))

        return q   # [B, K, d_ctx]
