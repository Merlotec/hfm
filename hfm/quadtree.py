"""
Dynamic-quadtree fluid model.

Instead of a fixed P×P grid of patches, the field is represented by the leaves
of a quadtree, so different quadrants can carry different resolutions.  Each
refinement round picks k leaves and replaces them with their four children.

Where to spend that resolution is a discrete, non-differentiable choice, so for
this first training stage `choose_splits` simply picks the k leaves **at
random**.  That is deliberate: random trees expose the model to every possible
mix of depths, which is exactly what the scale-invariant machinery below needs
in order to learn the detail axis rather than memorise one particular layout.
Override `choose_splits` to plug in a real criterion later.

Scale invariance
----------------
The whole point is that refinement is a *self-similar* operation, so the model
can be run for more rounds than it was trained with and keep behaving sensibly.
Three things enforce that:

1.  **Scale-invariant sampling.**  Every cell — whatever its physical size — is
    resampled to the same `sample_px × sample_px` crop (from a mip pyramid, so a
    coarse cell sees a properly prefiltered view).  The *same* patch-embed
    weights therefore always see the same kind of picture, just at a different
    scale.

2.  **Shared weights across rounds.**  One layer stack is applied at every
    round.  There are no per-round parameters at all — anything the model needs
    to know about "where it is" on the detail axis has to arrive through the
    scale encoding, which is what forces that axis to be learned as a
    continuous, extrapolatable quantity.

3.  **Relative scale encodings.**  Depth is measured relative to the coarsest
    live cell in the current tree (`d_rel`), and position is measured in units
    of that coarsest cell.  A tree occupying depths {2,3,4} therefore produces
    *identical* encodings to one occupying {0,1,2} — the model literally cannot
    tell absolute depth, only relative depth.  `d_rel` is fed through a
    continuous Fourier featurisation (not a per-level lookup table), so depths
    never seen during training still land somewhere sensible.

Token vocabulary
----------------
cells   : [B, N_r, d_patch]   quadtree leaves, N_r grows each round
global  : [B, n_global, d_patch]   learnable capacity tokens (identity rotation)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HFMConfig
from .model import FeedForward, HFMLayer, SkipEncoder, _checkpointed_layer


# ---------------------------------------------------------------------------
# Scale / position encoding
# ---------------------------------------------------------------------------

class RelativeScaleEncoding(nn.Module):
    """
    Continuous encoding of a cell's depth on the detail axis.

    Takes the *relative* depth d_rel = level - min(level) and maps it through a
    Fourier featurisation to a FiLM (gain, bias) pair.  Because the map is a
    smooth function of a scalar rather than an embedding table, depths outside
    the training range extrapolate instead of hitting an undefined row.
    """

    def __init__(self, dim: int, n_freqs: int = 6):
        super().__init__()
        self.n_freqs = n_freqs
        # Octave-spaced frequencies: the band is self-similar under d → d+1.
        self.register_buffer('freqs', math.pi * 2.0 ** torch.arange(n_freqs).float())
        self.mlp = nn.Sequential(nn.Linear(1 + 2 * n_freqs, dim), nn.GELU())
        self.out = nn.Linear(dim, 2 * dim)
        self.reset_film()

    def reset_film(self):
        """Start as an identity FiLM so early training is unperturbed."""
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    freqs: torch.Tensor

    def forward(self, tokens: torch.Tensor, d_rel: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, d];  d_rel: [B, N] float → [B, N, d]."""
        ang = d_rel.unsqueeze(-1) * self.freqs                      # [B, N, F]
        feat = torch.cat([d_rel.unsqueeze(-1), ang.sin(), ang.cos()], dim=-1)
        gain, bias = self.out(self.mlp(feat)).chunk(2, dim=-1)
        return tokens * (1.0 + gain) + bias


def build_rope(
    u: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    n_global: int,
    n_octaves: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    2-D RoPE for continuous cell centres.

    u, v : [B, N] cell centre coordinates *in units of the coarsest live cell*.
           Renormalising by the coarsest cell is what makes a uniformly refined
           tree indistinguishable from the original one.

    Frequencies are octave-spaced, which is the natural band for a quadtree:
    shifting every cell one level finer shifts the band by exactly one octave,
    so the set of phase differences the model sees is unchanged apart from the
    endpoints.  Global tokens receive identity rotation.

    Returns cos/sin of shape [B, 1, N + n_global, head_dim].
    """
    quarter = head_dim // 4
    j = torch.arange(quarter, device=u.device, dtype=u.dtype)
    omega = math.pi * 2.0 ** (j * (n_octaves / max(quarter - 1, 1)))   # [quarter]

    rf = v.unsqueeze(-1) * omega                                       # [B, N, q]
    cf = u.unsqueeze(-1) * omega
    half_c = torch.cat([rf.cos(), cf.cos()], dim=-1)                   # [B, N, hd/2]
    half_s = torch.cat([rf.sin(), cf.sin()], dim=-1)
    cos = torch.cat([half_c, half_c], dim=-1).unsqueeze(1)             # [B, 1, N, hd]
    sin = torch.cat([half_s, half_s], dim=-1).unsqueeze(1)

    B = u.shape[0]
    ones = torch.ones(B, 1, n_global, head_dim, device=u.device, dtype=u.dtype)
    zeros = torch.zeros_like(ones)
    return torch.cat([cos, ones], dim=2), torch.cat([sin, zeros], dim=2)


# ---------------------------------------------------------------------------
# Cell sampling
# ---------------------------------------------------------------------------

class CellSampler(nn.Module):
    """
    Resample every quadtree cell to a fixed `sample_px × sample_px` crop.

    A cell at depth d covers img_size / (base_grid · 2^d) pixels, so coarse
    cells are heavy downsamples.  Sampling those directly from the full-res
    frame would alias badly and break the self-similarity the model relies on,
    so each depth is sampled from the matching level of a mip pyramid.
    """

    def __init__(self, base_grid: int, sample_px: int):
        super().__init__()
        self.base_grid = base_grid
        self.sample_px = sample_px

    def _mips(self, x: torch.Tensor, max_depth: int) -> List[torch.Tensor]:
        """mip[d] is prefiltered for cells at depth d (coarsest cells → smallest mip)."""
        mips = [x]
        for _ in range(max_depth):
            prev = mips[-1]
            mips.append(prev if prev.shape[-1] <= self.sample_px
                        else F.avg_pool2d(prev, 2))
        # mips[0] is the least-detailed (deepest downsample) view.
        return mips[::-1]

    def forward(
        self,
        x: torch.Tensor,
        level: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
    ) -> torch.Tensor:
        """
        x     : [B, C, H, W]
        level : [B, N] long, row/col: [B, N] long (indices at their own level)
        returns [B, N, C, p, p]
        """
        B, C = x.shape[0], x.shape[1]
        N = level.shape[1]
        p = self.sample_px

        # Cell bounds in [0, 1] image coordinates.
        side = (self.base_grid * 2 ** level).to(x.dtype)               # [B, N]
        x0 = col.to(x.dtype) / side
        y0 = row.to(x.dtype) / side
        step = 1.0 / side

        # Sample points at pixel centres of the p×p crop.
        t = (torch.arange(p, device=x.device, dtype=x.dtype) + 0.5) / p
        gx = x0.unsqueeze(-1) + step.unsqueeze(-1) * t                 # [B, N, p]
        gy = y0.unsqueeze(-1) + step.unsqueeze(-1) * t

        # grid_sample wants normalised [-1, 1] coords, laid out [B, N*p, p, 2].
        gx = (2.0 * gx - 1.0).unsqueeze(2).expand(B, N, p, p)          # varies along W
        gy = (2.0 * gy - 1.0).unsqueeze(3).expand(B, N, p, p)          # varies along H
        grid = torch.stack([gx, gy], dim=-1).reshape(B, N * p, p, 2)

        mips = self._mips(x, int(level.max().item()))
        out = x.new_zeros(B, C, N * p, p)
        for d, mip in enumerate(mips):
            sel = (level == d).to(x.dtype)                             # [B, N]
            if not bool(sel.any()):
                continue
            samp = F.grid_sample(mip, grid, mode='bilinear',
                                 padding_mode='border', align_corners=False)
            w = sel.repeat_interleave(p, dim=1)[:, None, :, None]      # [B,1,N*p,1]
            out = out + samp * w

        return out.reshape(B, C, N, p, p).permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------------------
# Adaptive-resolution decoder
# ---------------------------------------------------------------------------

def _tent(k: int) -> torch.Tensor:
    """Normalised bilinear tent kernel, k×k, peaking at 1 in the centre."""
    coords = torch.arange(k).float() + 0.5
    centre = k / 2.0
    w1d = (centre - (coords - centre).abs()).clamp(min=0)
    wk = w1d.unsqueeze(1) * w1d.unsqueeze(0)
    return wk / wk.max()


class QuadtreeDecoder(nn.Module):
    """
    Splat quadtree leaves back onto the pixel grid.

    Each leaf predicts a fixed `out_px × out_px` tile *in its own local frame*,
    plus a 25% overlap margin blended with a tent kernel — the same operation at
    every scale.  Leaves are folded per depth into that depth's canvas, each
    canvas is resized to full resolution, and the canvases are combined by their
    accumulated tent weights.  Where the tree is deep the fine canvas dominates;
    where it is shallow the coarse one carries the region.
    """

    def __init__(self, d: int, out_channels: int, img_size: int,
                 base_grid: int, out_px: int, skip_ch: int = 0):
        super().__init__()
        assert out_px % 4 == 0, "out_px must be divisible by 4 for 25% overlap"
        self.img_size = img_size
        self.base_grid = base_grid
        self.out_px = out_px
        self.out_channels = out_channels
        self.skip_ch = skip_ch
        self.kernel = out_px + out_px // 2

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, out_channels * self.kernel * self.kernel),
        )

        mid = max(out_channels * 8, 64)
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_ch, mid, 3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, 3, padding=1, padding_mode='replicate'),
        )

        self.register_buffer('weight_kernel', _tent(self.kernel))

    weight_kernel: torch.Tensor

    def _tile_bank(self, preds: torch.Tensor, p_eff: int
                   ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Resize the fixed K×K prediction tiles to the stride actually used at
        this depth.  Deep levels would otherwise fold into a canvas far larger
        than the image; shrinking the tile keeps the canvas at most img_size
        while leaving the prediction itself scale-invariant.
        """
        B, N, C, _ = preds.shape
        K = self.kernel
        if p_eff == self.out_px:
            return preds, self.weight_kernel.flatten(), K

        K_eff = p_eff + p_eff // 2
        tiles = F.interpolate(
            preds.reshape(B * N * C, 1, K, K), size=K_eff,
            mode='bilinear', align_corners=False,
        ).reshape(B, N, C, K_eff * K_eff)
        wk = _tent(K_eff).to(preds.device, preds.dtype).flatten()
        return tiles, wk, K_eff

    def forward(
        self,
        tokens: torch.Tensor,
        level: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        skip_feats: Optional[List[torch.Tensor]] = None,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = tokens.shape
        C, K = self.out_channels, self.kernel

        preds = self.head(tokens).reshape(B, N, C, K * K)              # [B,N,C,K²]

        acc = tokens.new_zeros(B, C, self.img_size, self.img_size)
        acc_w = tokens.new_zeros(B, 1, self.img_size, self.img_size)

        for d in range(int(level.max().item()) + 1):
            sel = (level == d).to(tokens.dtype)                        # [B, N]
            if not bool(sel.any()):
                continue
            G = self.base_grid * 2 ** d

            # Never fold into a canvas larger than the image itself.
            p_eff = min(self.out_px, max(4, (self.img_size // G) // 4 * 4))
            tiles, wk, K_eff = self._tile_bank(preds, p_eff)
            canvas_px = G * p_eff
            idx = (row * G + col).clamp(0, G * G - 1)                  # [B, N]

            # Scatter this depth's leaves into a dense [B, C·K², G²] fold input.
            KK = K_eff * K_eff
            vals = (tiles * wk * sel[:, :, None, None]).reshape(B, N, C * KK)
            dense = tokens.new_zeros(B, G * G, C * KK)
            dense.scatter_add_(1, idx[:, :, None].expand(-1, -1, C * KK), vals)
            dense = dense.permute(0, 2, 1)                             # [B, C·K², G²]

            wvals = (sel[:, :, None] * wk).reshape(B, N, KK)
            wdense = tokens.new_zeros(B, G * G, KK)
            wdense.scatter_add_(1, idx[:, :, None].expand(-1, -1, KK), wvals)
            wdense = wdense.permute(0, 2, 1)

            fold_kw = dict(output_size=(canvas_px, canvas_px), kernel_size=K_eff,
                           stride=p_eff, padding=p_eff // 4)
            canvas = F.fold(dense.contiguous(), **fold_kw)             # [B, C, cp, cp]
            wcanvas = F.fold(wdense.contiguous(), **fold_kw)           # [B, 1, cp, cp]

            if canvas_px != self.img_size:
                mode = 'bilinear'
                canvas = F.interpolate(canvas, size=self.img_size, mode=mode,
                                       align_corners=False)
                wcanvas = F.interpolate(wcanvas, size=self.img_size, mode=mode,
                                        align_corners=False)

            # Finer levels get more say, matching their higher spatial confidence.
            acc = acc + canvas * (2.0 ** d)
            acc_w = acc_w + wcanvas * (2.0 ** d)

        output = acc / acc_w.clamp(min=1e-6)

        if pixel_mask is not None:
            output = output * pixel_mask

        post_in = (
            torch.cat([output, skip_feats[0]], dim=1)
            if self.skip_ch > 0 and skip_feats
            else output
        )
        return output + self.post_conv(post_in)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class QuadtreeHFM(nn.Module):
    """
    Fluid model over an adaptively-refined quadtree.

    forward(x, context, pixel_mask) → [B, C, H, W]   (same contract as HFM)

    The tree realised on each forward pass is stashed on `self.last_tree`
    rather than returned, so existing call sites are unchanged.
    """

    def __init__(self, cfg: HFMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_patch
        hd = d // cfg.n_heads
        assert hd % 4 == 0, "d_patch // n_heads must be divisible by 4 for 2-D RoPE"

        self.base_grid = cfg.qt_base_grid
        self.n_rounds = cfg.qt_rounds
        self.split_k = cfg.qt_split_k

        self.sampler = CellSampler(cfg.qt_base_grid, cfg.qt_sample_px)

        # Scale-invariant cell encoder: one crop → one token, same weights at
        # every depth.  This is the "self-similar operation at a smaller scale".
        self.cell_embed = nn.Conv2d(cfg.in_channels + 1, d,
                                    kernel_size=cfg.qt_sample_px,
                                    stride=cfg.qt_sample_px, bias=False)
        self.scale_enc = RelativeScaleEncoding(d, cfg.qt_scale_freqs)
        self.skip_encoder = SkipEncoder(cfg.in_channels, cfg.skip_ch)

        self.global_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(1, cfg.n_global_tokens, d), std=0.02)
        )

        self.layers = nn.ModuleList([
            HFMLayer(d, cfg.n_heads, cfg.d_ctx, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Blends a freshly sampled child crop into the state inherited from its parent.
        self.child_mix = nn.Sequential(nn.LayerNorm(d), FeedForward(d, 2.0))

        self.decoder = QuadtreeDecoder(
            d, cfg.in_channels, cfg.img_size, cfg.qt_base_grid,
            cfg.qt_out_px, cfg.skip_ch
        )

        self.last_tree: Dict[str, torch.Tensor] = {}
        self._init_weights()

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
        self.scale_enc.reset_film()
        w = self.cell_embed.weight
        nn.init.orthogonal_(w.reshape(w.shape[0], -1))
        for seq in [self.decoder.head, self.decoder.post_conv]:
            for child in reversed(list(seq.children())):
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    nn.init.zeros_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    break

    # -- pieces -------------------------------------------------------------

    def _embed_cells(self, x_aug: torch.Tensor, level: torch.Tensor,
                     row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """Sample each cell's region and encode it. → [B, N, d]"""
        B, N = level.shape
        crops = self.sampler(x_aug, level, row, col)                   # [B,N,C,p,p]
        flat = crops.reshape(B * N, *crops.shape[2:])
        return self.cell_embed(flat).reshape(B, N, self.cfg.d_patch)

    def _encode(self, tokens: torch.Tensor, level: torch.Tensor,
                row: torch.Tensor, col: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """Apply the (shared) layer stack with scale-relative encodings."""
        B, N, d = tokens.shape
        hd = d // self.cfg.n_heads

        # Everything is measured against the coarsest live cell, so an
        # all-over-refined tree is encoded identically to its parent tree.
        lmin = level.min(dim=1, keepdim=True).values                   # [B, 1]
        d_rel = (level - lmin).to(tokens.dtype)                        # [B, N]
        cell = 2.0 ** (-d_rel)                                         # in coarse-cell units
        u = (col.to(tokens.dtype) + 0.5) * cell
        v = (row.to(tokens.dtype) + 0.5) * cell

        tokens = self.scale_enc(tokens, d_rel)
        cos, sin = build_rope(u, v, hd, self.cfg.n_global_tokens, self.cfg.qt_rope_octaves)

        seq = torch.cat([tokens, self.global_tokens.expand(B, -1, -1)], dim=1)
        for layer in self.layers:
            if self.cfg.gradient_checkpointing and self.training:
                seq = _checkpointed_layer(layer, seq, context, cos, sin)
            else:
                seq = layer(seq, context, cos, sin)
        return seq[:, :N]

    def choose_splits(self, tokens: torch.Tensor, level: torch.Tensor,
                      k: int, max_level: int) -> torch.Tensor:
        """
        Pick which k leaves to refine.  Returns indices [B, k].

        Stage 1 (here): uniformly at random among the leaves not yet at max
        depth.  Choosing *where* to spend resolution is a discrete,
        non-differentiable decision, so there is nothing to learn from it yet —
        and random trees are in fact the better training signal for now, since
        they force the scale-invariant machinery to cope with every possible
        mix of depths rather than whichever one a learned policy would collapse
        onto.  Override this method to plug in a real criterion later.
        """
        splittable = level < max_level
        noise = torch.rand(level.shape, device=level.device)
        noise = noise.masked_fill(~splittable, -1.0)
        return noise.topk(k, dim=1).indices

    def _split(
        self,
        tokens: torch.Tensor,
        level: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        x_aug: torch.Tensor,
        max_level: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Replace k leaves by their four children.

        Which leaves get refined is chosen by `choose_splits` — currently
        uniformly at random.  The parent slot becomes child (0,0) and the other
        three are appended, so the token count grows by exactly 3k every round
        and shapes stay static.
        """
        N, d = tokens.shape[1], tokens.shape[2]
        k = min(self.split_k, N)

        top = self.choose_splits(tokens, level, k, max_level)           # [B, k]

        p_tok = tokens.gather(1, top[:, :, None].expand(-1, -1, d))
        p_lvl = level.gather(1, top)
        p_row = row.gather(1, top)
        p_col = col.gather(1, top)

        c_lvl = (p_lvl + 1).repeat(1, 4)                               # [B, 4k]
        offs_r = torch.tensor([0, 0, 1, 1], device=tokens.device)
        offs_c = torch.tensor([0, 1, 0, 1], device=tokens.device)
        c_row = (p_row * 2).repeat(1, 4) + offs_r.repeat_interleave(k)[None]
        c_col = (p_col * 2).repeat(1, 4) + offs_c.repeat_interleave(k)[None]

        fresh = self._embed_cells(x_aug, c_lvl, c_row, c_col)          # [B, 4k, d]
        inherited = p_tok.repeat(1, 4, 1)
        c_tok = inherited + self.child_mix(fresh)

        # Overwrite parents with child 0, append children 1..3.
        idx_d = top[:, :, None].expand(-1, -1, d)
        tokens = tokens.scatter(1, idx_d, c_tok[:, :k])
        level = level.scatter(1, top, c_lvl[:, :k])
        row = row.scatter(1, top, c_row[:, :k])
        col = col.scatter(1, top, c_col[:, :k])

        tokens = torch.cat([tokens, c_tok[:, k:]], dim=1)
        level = torch.cat([level, c_lvl[:, k:]], dim=1)
        row = torch.cat([row, c_row[:, k:]], dim=1)
        col = torch.cat([col, c_col[:, k:]], dim=1)
        return tokens, level, row, col

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
        n_rounds: Optional[int] = None,
    ) -> torch.Tensor:
        """
        `n_rounds` overrides the configured refinement depth — the model is
        scale-invariant by construction, so running more (or fewer) rounds at
        inference than at training time is a supported operation.
        """
        B = x.shape[0]
        rounds = self.n_rounds if n_rounds is None else n_rounds

        if self.training and self.cfg.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.cfg.noise_std

        if pixel_mask is not None:
            x = x * pixel_mask
            mask_ch = pixel_mask.float().expand(B, 1, x.shape[2], x.shape[3])
        else:
            mask_ch = torch.ones(B, 1, x.shape[2], x.shape[3],
                                 device=x.device, dtype=x.dtype)

        skip_feats = self.skip_encoder(x)
        x_aug = torch.cat([x, mask_ch], dim=1)

        # Root: a uniform base_grid × base_grid tiling at depth 0.
        G = self.base_grid
        ar = torch.arange(G, device=x.device)
        row = ar.repeat_interleave(G)[None].expand(B, -1).contiguous()
        col = ar.repeat(G)[None].expand(B, -1).contiguous()
        level = torch.zeros_like(row)

        tokens = self._embed_cells(x_aug, level, row, col)

        for r in range(rounds + 1):
            tokens = self._encode(tokens, level, row, col, context)
            if r < rounds:
                tokens, level, row, col = self._split(
                    tokens, level, row, col, x_aug, rounds
                )

        self.last_tree = {
            'level': level.detach(),
            'row': row.detach(),
            'col': col.detach(),
        }

        return self.decoder(tokens, level, row, col, skip_feats, pixel_mask=pixel_mask)
