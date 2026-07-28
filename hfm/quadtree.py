"""
Fixed 4-level quadtree fluid model.

The image is tiled by a *complete* quadtree: every level is fully populated, and a
block at level ``l`` covers ``finest_px · 2**l`` pixels.  For a 256px frame with
``finest_px = 4`` and 4 levels::

    level 0 :  4px blocks → 64×64 grid  (4096 tokens)   finest, encoded losslessly
    level 1 :  8px blocks → 32×32 grid  (1024 tokens)
    level 2 : 16px blocks → 16×16 grid  ( 256 tokens)
    level 3 : 32px blocks →  8×8  grid  (  64 tokens)   coarsest

Each level carries its own token dimension (a bell curve peaking at level 1) and its
own attention:

* **Self-attention** is local with 2-D RoPE over a ``(k+1)×(k+1)`` block window, where
  ``k`` is the block's pixel side.  Level 0 sees a 5×5 window, level 1 a 9×9 window;
  by level 2 the window (17×17) already covers the whole 16×16 grid, so levels 2–3 are
  effectively global.

* **Cross-attention** couples adjacent levels bidirectionally — every parent attends to
  its 4 children and every child to its parent (:class:`AdjacentCrossAttn`).  Composed
  across passes this is what lets a parent update a child and vice versa, and what
  reaches all descendants over the course of the schedule.

Update schedule (red / black + pyramidal tail)
-----------------------------------------------
Levels {0, 2} self-attend on one pass, {1, 3} on the next — the two halves run "in
parallel", which also halves the cost of the expensive fine-level self-attention.
Cross-attention runs on every pass between still-live adjacent levels, so a {0,2} pass
still writes into levels 1 and 3.  Each level has a self-attention *budget*
(``ml_passes``, non-increasing); coarse budgets run out first, so late passes touch
only the fine levels — a pyramidal tail that spends no compute refining coarse tokens
that are never decoded.  The output is decoded from level 0 alone, which fully tiles
the image.

The number of times the coarsest level iterates (``ml_passes[-1]``) is what the user
calls the "number of layers".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HFMConfig
from .model import FeedForward, OverlappingPatchDecoder
from .attention import CrossAttention, LocalSelfAttentionRoPE, MaskedSelfAttentionRoPE


# ---------------------------------------------------------------------------
# 2-D RoPE tables
# ---------------------------------------------------------------------------

def build_grid_rope(grid: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard 2-D RoPE cos/sin for a ``grid × grid`` field in row-major order.

    Returns cos/sin of shape ``[1, 1, grid², head_dim]`` — the layout consumed by
    :class:`GlobalSelfAttn` and :class:`LocalSelfAttentionRoPE`.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2-D RoPE"
    quarter = head_dim // 4
    theta = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float) / quarter))
    rows = torch.arange(grid).repeat_interleave(grid).float()   # [grid²]
    cols = torch.arange(grid).repeat(grid).float()
    rf = torch.outer(rows, theta)                               # [grid², quarter]
    cf = torch.outer(cols, theta)
    half_c = torch.cat([rf.cos(), cf.cos()], dim=-1)            # [grid², hd/2]
    half_s = torch.cat([rf.sin(), cf.sin()], dim=-1)
    cos = torch.cat([half_c, half_c], dim=-1)[None, None]       # [1, 1, grid², hd]
    sin = torch.cat([half_s, half_s], dim=-1)[None, None]
    return cos, sin


def build_window_mask(grid: int, radius: int) -> torch.Tensor:
    """
    Additive attention mask (``[1, 1, grid², grid²]``) restricting every token to the
    ``(2·radius+1)²`` Chebyshev window around it: 0 where attention is allowed, -inf
    outside.  Fed to :class:`MaskedSelfAttentionRoPE` so a fused full-attention kernel
    reproduces a local window exactly.  Every token attends to itself, so no row is
    fully masked.
    """
    idx = torch.arange(grid * grid)
    ri, ci = idx // grid, idx % grid
    dr = (ri[:, None] - ri[None, :]).abs()
    dc = (ci[:, None] - ci[None, :]).abs()
    keep = (dr <= radius) & (dc <= radius)                      # [N, N] bool
    mask = torch.zeros(grid * grid, grid * grid)
    mask.masked_fill_(~keep, float('-inf'))
    return mask[None, None]


# Local grids with at most this many cells per side use full masked SDPA (one fused
# kernel) instead of the gather-based local attention; larger grids (level 0's 64×64)
# keep the gather, since a full N² attention there is far more work than the window.
_MASKED_SDPA_MAX_GRID = 32


# ---------------------------------------------------------------------------
# Per-level transformer block
# ---------------------------------------------------------------------------

class LevelBlock(nn.Module):
    """
    One level's update: RoPE self-attention → context cross-attention → FFN.

    The self-attention backend is chosen per level (tokens stay in grid form
    ``[B, G, G, d]``):

    * ``global`` — the ``(k+1)`` window already covers the grid: fused SDPA, no mask.
    * ``masked`` — a local window on a small grid (e.g. level 1's 9×9 over 32×32):
      fused SDPA restricted by a precomputed window mask — same result as a gather but
      ~20× faster at this token count.
    * ``local``  — a small window on a large grid (level 0's 5×5 over 64×64): the
      gather-based :class:`LocalSelfAttentionRoPE`, since a full N² attention would be
      wasteful there.
    """

    def __init__(self, d: int, n_heads: int, d_ctx: int, grid: int, radius: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.grid = grid
        if (2 * radius + 1) >= grid:
            self.mode = 'global'
        elif grid <= _MASKED_SDPA_MAX_GRID:
            self.mode = 'masked'
        else:
            self.mode = 'local'

        self.norm1 = nn.LayerNorm(d)
        if self.mode == 'local':
            self.self_attn = LocalSelfAttentionRoPE(d, n_heads, radius, dropout=dropout)
        else:
            self.self_attn = MaskedSelfAttentionRoPE(d, n_heads, dropout=dropout)
            if self.mode == 'masked':
                self.register_buffer('attn_mask', build_window_mask(grid, radius))

        self.norm2 = nn.LayerNorm(d)
        self.cross_ctx = CrossAttention(d, d_ctx, n_heads, dropout=dropout)
        self.norm3 = nn.LayerNorm(d)
        self.ffn = FeedForward(d, mlp_ratio, dropout=dropout)

    def forward(self, tok: torch.Tensor, context: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = self.norm1(tok)
        if self.mode == 'local':
            a = self.self_attn(n, cos, sin)
        elif self.mode == 'masked':
            a = self.self_attn(n, cos, sin, attn_mask=self.attn_mask)
        else:  # global
            a = self.self_attn(n, cos, sin, attn_mask=None)
        tok = tok + a
        tok = tok + self.cross_ctx(self.norm2(tok), context)   # CrossAttention handles [B,G,G,d]
        tok = tok + self.ffn(self.norm3(tok))
        return tok


# ---------------------------------------------------------------------------
# Bidirectional parent <-> child cross-attention between adjacent levels
# ---------------------------------------------------------------------------

class AdjacentCrossAttn(nn.Module):
    """
    Couple a coarse (parent) level and the next finer (child) level, both ways.

    The child grid is exactly ``2×`` the parent grid, so every parent block owns a
    disjoint 2×2 group of children.

    * **parent ← children**: each parent attends to its 4 children (real attention).
    * **child ← parent**: the parent field is **bilinearly upsampled** to the child grid
      and projected.  This replaces the old single-key "attention" (which, with one key,
      degenerated into copying the parent identically onto all 4 children — a
      block-uniform correction that stamps visible 8/16/32px square artifacts as soon as
      the coarse tokens go off-distribution, e.g. on the model's own rollout output).
      Interpolation makes the correction vary smoothly across block boundaries.

    Both updates are residual, so the two levels update each other within one pass.
    """

    def __init__(self, dp: int, dc: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_p = nn.LayerNorm(dp)
        self.norm_c = nn.LayerNorm(dc)
        self.parent_from_child = CrossAttention(dp, dc, n_heads, dropout=dropout)
        # child ← parent: smooth interpolation + learned projection (not a broadcast).
        self.child_from_parent = nn.Linear(dp, dc, bias=False)

    def forward(self, parent: torch.Tensor, child: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Gp, _, dp = parent.shape
        dc = child.shape[-1]

        np_ = self.norm_p(parent)                                    # [B, Gp, Gp, dp]
        nc_ = self.norm_c(child)                                     # [B, 2Gp, 2Gp, dc]

        # parent ← its 4 children (real attention).
        ncg = (nc_.reshape(B, Gp, 2, Gp, 2, dc)
                  .permute(0, 1, 3, 2, 4, 5)
                  .reshape(B * Gp * Gp, 4, dc))
        pq = np_.reshape(B * Gp * Gp, 1, dp)
        p_upd = self.parent_from_child(pq, ncg).reshape(B, Gp, Gp, dp)

        # child ← parent: bilinear upsample Gp→2Gp so the correction is smooth across
        # block boundaries (no block-uniform stamping), then project dp→dc.
        p_up = F.interpolate(np_.permute(0, 3, 1, 2), scale_factor=2,
                             mode='bilinear', align_corners=False)   # [B, dp, 2Gp, 2Gp]
        c_upd = self.child_from_parent(p_up.permute(0, 2, 3, 1))     # [B, 2Gp, 2Gp, dc]

        return parent + p_upd, child + c_upd


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class QuadtreeHFM(nn.Module):
    """
    Fixed complete-quadtree fluid model.

    forward(x, context, pixel_mask) → [B, C, H, W]   (same contract as flat HFM)
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        self.cfg = cfg
        L = cfg.ml_levels
        self.n_levels = L
        self.dims = list(cfg.ml_dims)
        nh = cfg.n_heads
        assert len(self.dims) == L, "ml_dims must have ml_levels entries"
        assert len(cfg.ml_passes) == L, "ml_passes must have ml_levels entries"

        # Per-level geometry.  Level l blocks are finest_px·2^l pixels; the grid is
        # img_size / block_px on a side; the self-attn window is (block_px + 1) blocks,
        # i.e. radius = block_px // 2.
        self.block_px = [cfg.ml_finest_px * (2 ** l) for l in range(L)]
        self.grids = [cfg.img_size // bp for bp in self.block_px]
        self.radii = [bp // 2 for bp in self.block_px]
        for d in self.dims:
            assert d % nh == 0 and (d // nh) % 4 == 0, \
                f"each ml_dim must be divisible by n_heads and dim//n_heads%4==0 (got {d})"

        self.residual_prediction = getattr(cfg, 'residual_prediction', True)

        # Per-level lossless-capable patch embeds from the mask-augmented frame.
        self.embeds = nn.ModuleList([
            nn.Conv2d(cfg.in_channels + 1, self.dims[l],
                      kernel_size=self.block_px[l], stride=self.block_px[l], bias=False)
            for l in range(L)
        ])

        # Per-level RoPE tables (buffers so they follow the module to any device).
        for l in range(L):
            cos, sin = build_grid_rope(self.grids[l], self.dims[l] // nh)
            self.register_buffer(f'rope_cos_{l}', cos)
            self.register_buffer(f'rope_sin_{l}', sin)

        # Self-attn blocks per level.  Each level owns n_blocks[l] distinct blocks,
        # cycled round-robin across its passes: 1 ties all passes (default, parameter-
        # efficient iterative refinement), ml_passes[l] fully unties.  ml_untie_passes
        # forces the fully-untied case; otherwise ml_blocks_per_level sets it.
        passes = list(cfg.ml_passes)
        if getattr(cfg, 'ml_untie_passes', False):
            nb = list(passes)
        else:
            nb = list(getattr(cfg, 'ml_blocks_per_level', (1,) * L))
        assert len(nb) == L, "ml_blocks_per_level must have ml_levels entries"
        # Never more blocks than passes (extras would be dead weight), never fewer than 1.
        self.n_blocks = [max(1, min(int(nb[l]), passes[l])) for l in range(L)]

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                LevelBlock(self.dims[l], nh, cfg.d_ctx, self.grids[l], self.radii[l],
                           cfg.mlp_ratio, cfg.dropout)
                for _ in range(self.n_blocks[l])
            ])
            for l in range(L)
        ])

        # Bidirectional cross-attn between each adjacent (coarse, fine) pair.
        # cross[l] couples level l (parent) with level l-1 (child).
        self.cross = nn.ModuleList([
            AdjacentCrossAttn(self.dims[l], self.dims[l - 1], nh, cfg.dropout)
            for l in range(1, L)
        ])

        self.schedule = self._build_schedule(passes)

        # Decode from the finest level, which fully tiles the image.  No post-fold conv
        # and no skip encoder: there is no spatial compression to recover from, and the
        # mask reaches the model through the per-level embeds (fed x_aug).  The folded
        # head output is the prediction directly.
        self.decoder = OverlappingPatchDecoder(
            self.dims[0], cfg.in_channels, cfg.img_size, cfg.ml_finest_px,
            skip_ch=0, use_post_conv=False)

        self.last_tree: Dict[str, torch.Tensor] = {}
        self._init_weights()

    # -- schedule -----------------------------------------------------------

    def _build_schedule(self, budgets: List[int]
                        ) -> List[Tuple[List[Tuple[int, int]], List[int]]]:
        """
        Build the red/black + pyramidal pass list.

        Returns a list of ``(self_items, cross_indices)`` where ``self_items`` are
        ``(level, block_idx)`` pairs — ``block_idx`` cycles round-robin over that level's
        ``n_blocks`` as its passes progress (always 0 when the level is fully tied) — and
        ``cross_indices`` index ``self.cross`` (pair (l-1, l) has index l-1).  Levels
        {0,2} run on even-parity passes, {1,3} on odd; a level self-attends only while it
        has budget, and a cross pair runs only while both endpoints still have budget.
        Emission stops once the finest level is exhausted.
        """
        L = self.n_levels
        budgets = list(budgets)
        counts = [0] * L        # times each level has self-attended so far
        schedule: List[Tuple[List[Tuple[int, int]], List[int]]] = []
        g = 0
        guard = 0
        while budgets[0] > 0 and guard < 10_000:
            guard += 1
            alive = [l for l in range(L) if budgets[l] > 0]
            grp = [l for l in range(L) if l % 2 == g]
            self_items: List[Tuple[int, int]] = []
            for l in grp:
                if budgets[l] > 0:
                    self_items.append((l, counts[l] % self.n_blocks[l]))
                    counts[l] += 1
                    budgets[l] -= 1
            cross = [l - 1 for l in range(1, L) if (l - 1) in alive and l in alive]
            if self_items or cross:
                schedule.append((self_items, cross))
            g ^= 1
        return schedule

    # -- init ---------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for emb in self.embeds:
            w = emb.weight
            nn.init.orthogonal_(w.reshape(w.shape[0], -1))
        # Zero the decoder's final projection(s) so the model starts at exact
        # persistence (out = x_base + 0) with residual prediction on.  With no
        # post_conv the head alone carries this.
        seqs = [self.decoder.head]
        if getattr(self.decoder, 'post_conv', None) is not None:
            seqs.append(self.decoder.post_conv)
        for seq in seqs:
            for child in reversed(list(seq.children())):
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    nn.init.zeros_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    break

    # -- pieces -------------------------------------------------------------

    def _rope(self, l: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, f'rope_cos_{l}'), getattr(self, f'rope_sin_{l}')

    def _run_pass(self, tokens: List[torch.Tensor], context: torch.Tensor,
                  self_items: List[Tuple[int, int]], cross: List[int]) -> List[torch.Tensor]:
        # Self-attention on the active levels (the "0&2 / 1&3 in parallel" halves).
        for l, b in self_items:
            cos, sin = self._rope(l)
            tokens[l] = self.blocks[l][b](tokens[l], context, cos, sin)
        # Bidirectional parent<->child coupling on live adjacent pairs.
        for ci in cross:
            parent, child = self.cross[ci](tokens[ci + 1], tokens[ci])
            tokens[ci + 1], tokens[ci] = parent, child
        return tokens

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]

        # Residual base: the CLEAN masked input, taken before training noise so the
        # noise does not land straight in the output.
        x_base = x * pixel_mask if pixel_mask is not None else x

        if self.training and self.cfg.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.cfg.noise_std

        if pixel_mask is not None:
            x = x * pixel_mask
            mask_ch = pixel_mask.float().expand(B, 1, x.shape[2], x.shape[3])
        else:
            mask_ch = torch.ones(B, 1, x.shape[2], x.shape[3],
                                 device=x.device, dtype=x.dtype)

        x_aug = torch.cat([x, mask_ch], dim=1)

        # Embed every level from the frame at its own scale.
        tokens: List[torch.Tensor] = []
        for l in range(self.n_levels):
            emb = self.embeds[l](x_aug)                     # [B, d_l, G, G]
            tokens.append(emb.permute(0, 2, 3, 1).contiguous())   # [B, G, G, d_l]

        for self_items, cross in self.schedule:
            if self.cfg.gradient_checkpointing and self.training:
                tokens = _checkpointed_pass(self._run_pass, tokens, context,
                                            self_items, cross)
            else:
                tokens = self._run_pass(tokens, context, self_items, cross)

        out = self.decoder(tokens[0], pixel_mask=pixel_mask)

        if self.residual_prediction:
            out = x_base + out
            if pixel_mask is not None:
                out = out * pixel_mask
        return out


def _checkpointed_pass(fn, tokens: List[torch.Tensor], context: torch.Tensor,
                       self_items: List[Tuple[int, int]], cross: List[int]) -> List[torch.Tensor]:
    """Gradient-checkpoint one schedule pass.  ``self_items``/``cross`` are captured
    in a closure so only tensors cross the checkpoint boundary."""
    from torch.utils.checkpoint import checkpoint

    def run(*toks):
        return tuple(fn(list(toks), context, self_items, cross))

    out = checkpoint(run, *tokens, use_reentrant=False)
    return list(out)
