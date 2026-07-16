"""
Attention primitives used by the hierarchical fluid model.

Three variants are needed:
  - local_self_attn  : each token attends within a (2r+1)^2 neighbourhood on a H×W grid
  - window_self_attn : non-overlapping windows of size W×W, full attn inside each window
  - cross_attn       : standard multi-head cross-attention (queries from one set,
                       keys/values from another)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scaled_dot(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
    """q/k/v: [..., L, head_dim].  Returns [..., L, head_dim]."""
    scale = math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) / scale
    if mask is not None:
        attn = attn.masked_fill(~mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# Local self-attention on a spatial grid
# ---------------------------------------------------------------------------

class LocalSelfAttention(nn.Module):
    """
    Self-attention where each spatial position attends only within a
    (2·radius+1)^2 window of its neighbours.

    Input:  [B, H, W, C]
    Output: [B, H, W, C]
    """

    def __init__(self, dim: int, n_heads: int, radius: int = 3, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.radius = radius
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        r = self.radius
        win = 2 * r + 1

        # Pad spatially so every position has a full (win × win) neighbourhood
        # x_pad: [B, H+2r, W+2r, C]
        xp = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        xp = F.pad(xp, (r, r, r, r), mode='reflect')
        xp = xp.permute(0, 2, 3, 1)  # [B, H+2r, W+2r, C]

        # For each position collect its neighbourhood tokens
        # Use unfold: [B, H, W, win*win, C]
        rows = [xp[:, i:i+H, :, :] for i in range(win)]
        strips = []
        for row in rows:
            cols = [row[:, :, j:j+W, :] for j in range(win)]
            strips.append(torch.stack(cols, dim=3))          # [B, H, W, C] stacked
        nbr = torch.stack(strips, dim=3)                     # [B, H, win, W, win, C]
        nbr = nbr.reshape(B, H, W, win * win, C)             # [B, H, W, N, C]  N=win²

        # Query from centre token, keys/values from neighbourhood
        q_full = self.qkv(x)                                 # [B, H, W, 3C]
        kv_full = self.qkv(nbr.reshape(B * H * W, win * win, C))
        q_full = q_full.reshape(B * H * W, 1, 3 * C)

        q = q_full[..., :C]                                  # [B*H*W, 1, C]
        k, v = kv_full[..., :C], kv_full[..., C:2*C]        # [B*H*W, N, C]

        # Split heads
        nh, hd = self.n_heads, self.head_dim
        q = q.reshape(B * H * W, 1, nh, hd).transpose(1, 2)   # [BHW, nh, 1, hd]
        k = k.reshape(B * H * W, win*win, nh, hd).transpose(1, 2)
        v = v.reshape(B * H * W, win*win, nh, hd).transpose(1, 2)

        out = _scaled_dot(q, k, v)                            # [BHW, nh, 1, hd]
        out = out.transpose(1, 2).reshape(B * H * W, 1, C)
        out = out.reshape(B, H, W, C)
        return self.drop(self.proj(out))


# ---------------------------------------------------------------------------
# Window self-attention (non-overlapping tiles)
# ---------------------------------------------------------------------------

class WindowSelfAttention(nn.Module):
    """
    Partition [B, H, W, C] into non-overlapping windows of size ws×ws,
    run full self-attention inside each window.

    H and W must be divisible by ws.
    """

    def __init__(self, dim: int, n_heads: int, window_size: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.ws = window_size
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws = self.ws
        assert H % ws == 0 and W % ws == 0, \
            f"H={H}, W={W} must be divisible by window_size={ws}"

        # Partition into windows: [B*nW, ws*ws, C]
        x_w = rearrange(x, 'b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c',
                        wh=ws, ww=ws)

        qkv = self.qkv(x_w)
        q, k, v = qkv.chunk(3, dim=-1)

        nh, hd = self.n_heads, self.head_dim
        Bw = x_w.shape[0]
        N = ws * ws

        q = q.reshape(Bw, N, nh, hd).transpose(1, 2)   # [Bw, nh, N, hd]
        k = k.reshape(Bw, N, nh, hd).transpose(1, 2)
        v = v.reshape(Bw, N, nh, hd).transpose(1, 2)

        out = _scaled_dot(q, k, v)                      # [Bw, nh, N, hd]
        out = out.transpose(1, 2).reshape(Bw, N, C)

        # Reconstruct spatial layout
        out = rearrange(out, '(b nh nw) (wh ww) c -> b (nh wh) (nw ww) c',
                        b=B, nh=H//ws, nw=W//ws, wh=ws, ww=ws)
        return self.drop(self.proj(out))


# ---------------------------------------------------------------------------
# Cross-attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Standard multi-head cross-attention.

    queries: [B, Lq, Cq]   (or [B, H, W, C] — will be flattened/unflattened)
    context: [B, Lk, Ck]   (or [B, Hk, Wk, Ck])
    """

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        assert q_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = q_dim // n_heads
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """queries/context may be [B, L, C] or [B, H, W, C]. Returns same shape as queries."""
        q_4d = queries.ndim == 4
        k_4d = context.ndim == 4

        Hq = Wq = None
        if q_4d:
            B, Hq, Wq, Cq = queries.shape
            queries = queries.reshape(B, Hq * Wq, Cq)
        else:
            B = queries.shape[0]

        if k_4d:
            Bk, Hk, Wk, Ck = context.shape
            context = context.reshape(Bk, Hk * Wk, Ck)

        q = self.q_proj(queries)
        k = self.k_proj(context)
        v = self.v_proj(context)

        nh, hd = self.n_heads, self.head_dim
        Lq, Lk = q.shape[1], k.shape[1]

        q = q.reshape(B, Lq, nh, hd).transpose(1, 2)
        k = k.reshape(B, Lk, nh, hd).transpose(1, 2)
        v = v.reshape(B, Lk, nh, hd).transpose(1, 2)

        out = _scaled_dot(q, k, v)                      # [B, nh, Lq, hd]
        out = out.transpose(1, 2).reshape(B, Lq, nh * hd)
        out = self.drop(self.out_proj(out))

        if q_4d and Hq is not None and Wq is not None:
            out = out.reshape(B, Hq, Wq, nh * hd)
        return out


# ---------------------------------------------------------------------------
# Local cross-attention between a fine-grained grid and a coarse-grained grid
# ---------------------------------------------------------------------------

class LocalCrossAttention(nn.Module):
    """
    Each fine-grid token (H_f × W_f) attends to the single coarse-grid token
    (H_c × W_c) that spatially covers it, plus its (2r+1)^2 coarse neighbours.

    This lets fine tokens absorb global context flowing through the coarse level.
    """

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int,
                 radius: int = 1, dropout: float = 0.0):
        super().__init__()
        assert q_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = q_dim // n_heads
        self.radius = radius
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """
        fine:   [B, H_f, W_f, q_dim]
        coarse: [B, H_c, W_c, kv_dim]  — H_c divides H_f
        Returns [B, H_f, W_f, q_dim]
        """
        B, Hf, Wf, Cq = fine.shape
        _, Hc, Wc, Ck = coarse.shape
        r = self.radius
        win = 2 * r + 1

        # Pad coarse and unfold neighbourhood for each coarse cell
        c = coarse.permute(0, 3, 1, 2)               # [B, Ck, Hc, Wc]
        c = F.pad(c, (r, r, r, r), mode='replicate')
        c = c.permute(0, 2, 3, 1)                    # [B, Hc+2r, Wc+2r, Ck]

        rows = [c[:, i:i+Hc, :, :] for i in range(win)]
        strips = []
        for row in rows:
            strips.append(torch.stack([row[:, :, j:j+Wc, :] for j in range(win)], dim=3))
        nbr_c = torch.stack(strips, dim=3)            # [B, Hc, win, Wc, win, Ck]
        nbr_c = nbr_c.reshape(B, Hc, Wc, win * win, Ck)

        # Each fine token maps to a coarse cell via downscaling
        scale_h = Hf // Hc
        scale_w = Wf // Wc
        # Expand coarse neighbourhood to fine resolution
        # nbr_c_fine: [B, Hf, Wf, win*win, Ck]
        nbr_c_fine = nbr_c.repeat_interleave(scale_h, dim=1) \
                          .repeat_interleave(scale_w, dim=2)

        q = self.q_proj(fine)                         # [B, Hf, Wf, Cq]
        BHW = B * Hf * Wf
        N = win * win
        kv_src = nbr_c_fine.reshape(BHW, N, Ck)

        k = self.k_proj(kv_src)                       # [BHW, N, Cq]
        v = self.v_proj(kv_src)

        nh, hd = self.n_heads, self.head_dim
        q = q.reshape(BHW, 1, nh, hd).transpose(1, 2)
        k = k.reshape(BHW, N, nh, hd).transpose(1, 2)
        v = v.reshape(BHW, N, nh, hd).transpose(1, 2)

        out = _scaled_dot(q, k, v)                    # [BHW, nh, 1, hd]
        out = out.transpose(1, 2).reshape(B, Hf, Wf, Cq)
        return self.drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# Windowed cross-attention between two spatial grids of (possibly different) resolution
# ---------------------------------------------------------------------------

class WindowedCrossAttention(nn.Module):
    """
    Cross-attention between a query grid [B, Hq, Wq, q_dim] and a context
    grid [B, Hk, Wk, kv_dim].

    Both grids are partitioned into spatial windows of size ws × ws.  The
    context grid is pooled / repeated to match the query window count so that
    each query window sees only the context tokens that cover the same region.

    Requires: Hq % ws == 0, Wq % ws == 0.
    The context grid is resized (nearest) so its window count matches.
    """

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int,
                 window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        assert q_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = q_dim // n_heads
        self.ws = window_size
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Hq, Wq, Cq = queries.shape
        _, Hk, Wk, _ = context.shape
        ws = self.ws

        assert Hq % ws == 0 and Wq % ws == 0

        nwH = Hq // ws
        nwW = Wq // ws

        # Resize context to match query windows count * ws so we can partition it
        # into the same number of windows.
        target_Hk = nwH * ws
        target_Wk = nwW * ws
        if Hk != target_Hk or Wk != target_Wk:
            ctx = context.permute(0, 3, 1, 2).float()
            ctx = F.interpolate(ctx, size=(target_Hk, target_Wk),
                                mode='bilinear', align_corners=False)
            context = ctx.permute(0, 2, 3, 1)

        # Partition queries into windows: [B*nwH*nwW, ws*ws, Cq]
        q_win = rearrange(queries, 'b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c',
                          wh=ws, ww=ws)
        k_win = rearrange(context, 'b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c',
                          wh=ws, ww=ws)

        q = self.q_proj(q_win)
        k = self.k_proj(k_win)
        v = self.v_proj(k_win)

        Bw = q_win.shape[0]
        Nq = ws * ws
        Nk = ws * ws
        nh, hd = self.n_heads, self.head_dim

        q = q.reshape(Bw, Nq, nh, hd).transpose(1, 2)
        k = k.reshape(Bw, Nk, nh, hd).transpose(1, 2)
        v = v.reshape(Bw, Nk, nh, hd).transpose(1, 2)

        out = _scaled_dot(q, k, v)
        out = out.transpose(1, 2).reshape(Bw, Nq, Cq)
        out = rearrange(out, '(b nh nw) (wh ww) c -> b (nh wh) (nw ww) c',
                        b=B, nh=nwH, nw=nwW, wh=ws, ww=ws)
        return self.drop(self.out_proj(out))
