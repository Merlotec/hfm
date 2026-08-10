"""
Conditional flow-matching refiner for HFM.

The base model trains on masked MSE+L1, whose optimum is the posterior mean
over chaotic outcomes — high-frequency detail that differs between plausible
futures is averaged away.  This module SAMPLES that missing detail: a small
conditional rectified-flow UNet learns the distribution of the detail residual

    d = x_target - pred_hfm            (z-scored input space, holes zeroed)

conditioned on the frozen base prediction, the input frame, the geometry mask,
and the pooled context vector (which provably carries stride/mesh/BC — the
ContextProbe decodes them from exactly this mean-pooled vector).

Flow-matching objective (rectified flow, noise at t=0 → data at t=1):

    d~  = d / sigma_d                  sigma_d: fixed per-channel scale [C]
    z   ~ N(0, I) * mask
    x_t = (1 - t) * z + t * d~
    u   = d~ - z                       (constant along the straight path)
    L   = mask-weighted MSE( v_theta(x_t, t | cond), u )

MSE only — an L1 term would bias the learned velocity away from the CFM
conditional-expectation optimum.  Sampling is a plain Euler ODE integration
(few steps suffice; the paths are near-straight for small residuals).

The refiner has its own RefinerConfig, deliberately NOT part of HFMConfig:
HFMConfig objects are pickled whole into base checkpoints, and growing that
class already caused a pickle-compat bug once (see the ctx_temporal_diffs
comment in context_encoder.py).  Future shape-changing RefinerConfig fields
must be read with vars(cfg).get(...) for the same reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RefinerConfig:
    in_channels: int = 4          # field channels (rho, u, v, p)
    cond_channels: int = 13       # x_t(4) + pred_hfm(4) + x_in(4) + mask(1)
    base_width: int = 64
    channel_mults: tuple = (1, 2, 3)
    blocks_per_level: int = 2
    groups: int = 32
    t_dim: int = 256
    d_ctx: int = 384              # must match the base cfg's d_ctx
    # 0 = legacy mean-pooled context vector [B, d_ctx].  > 0 = the refiner sees
    # ALL n context tokens [B, n, d_ctx]: each summary token is a learned query
    # slot with a stable identity, so per-token LayerNorm + flatten + MLP is a
    # meaningful (order-stable) encoding, not a bag-of-tokens hack.  Mean-pooling
    # provably suffices for the probe's labels (dt, BCs, physics class), but it
    # discards whatever ELSE the 16 tokens carry -- flow-regime detail the
    # refiner may condition texture on.
    n_ctx_tokens: int = 0
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of t in [0, 1] (scaled by 1000), fp32."""
    t = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)          # [B, dim]


class ConditionEmbed(nn.Module):
    """t-embedding MLP + projected context, summed.

    n_ctx_tokens = 0: context arrives mean-pooled as [B, d_ctx] (legacy).
    n_ctx_tokens = K: context arrives as the full token set [B, K, d_ctx];
    tokens are LayerNormed individually, flattened in their (stable) slot
    order, and projected.  Either way the result is one [B, t_dim] embedding
    consumed by every FiLM layer.
    """

    def __init__(self, t_dim: int, d_ctx: int, n_ctx_tokens: int = 0):
        super().__init__()
        self.n_ctx_tokens = n_ctx_tokens
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        if n_ctx_tokens > 0:
            self.ctx_norm = nn.LayerNorm(d_ctx)          # per token, pre-flatten
            self.ctx_proj = nn.Sequential(
                nn.Linear(d_ctx * n_ctx_tokens, t_dim), nn.SiLU(),
                nn.Linear(t_dim, t_dim),
            )
        else:
            # EXACT legacy layout (LayerNorm inside the Sequential) so existing
            # refiner checkpoints keep loading key-for-key.
            self.ctx_norm = None
            self.ctx_proj = nn.Sequential(
                nn.LayerNorm(d_ctx), nn.Linear(d_ctx, t_dim), nn.SiLU(),
                nn.Linear(t_dim, t_dim),
            )

    def forward(self, t: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        emb = self.t_mlp(timestep_embedding(t, self.t_mlp[0].in_features))
        if ctx is not None:
            c = ctx.float()
            if self.n_ctx_tokens > 0:
                if c.dim() != 3 or c.shape[1] != self.n_ctx_tokens:
                    raise ValueError(
                        f'expected full context [B, {self.n_ctx_tokens}, d_ctx], '
                        f'got {tuple(c.shape)} -- this refiner was built for all '
                        f'tokens, not the pooled vector')
                c = self.ctx_norm(c).flatten(1)
            elif c.dim() == 3:                      # tolerate tokens: pool them
                c = c.mean(dim=1)
            emb = emb + self.ctx_proj(c)
        return emb                                             # [B, t_dim]


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class FiLMResBlock(nn.Module):
    """GN → SiLU → conv → FiLM(emb) → GN → SiLU → conv, with residual."""

    def __init__(self, c_in: int, c_out: int, t_dim: int,
                 groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.film = nn.Linear(t_dim, 2 * c_out)
        nn.init.zeros_(self.film.weight)                       # conditioning starts neutral
        nn.init.zeros_(self.film.bias)
        self.norm2 = nn.GroupNorm(min(groups, c_out), c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class RefinerUNet(nn.Module):
    """
    Small 3-level UNet velocity field v_theta(x_t, t | pred, x_in, mask, ctx).

    Deliberately boring: GroupNorm+SiLU ResBlocks with FiLM time/context
    conditioning, strided-conv downsampling, nearest-neighbour upsampling,
    zero-initialised output head.
    """

    def __init__(self, rcfg: RefinerConfig):
        super().__init__()
        self.rcfg = rcfg
        w = [rcfg.base_width * m for m in rcfg.channel_mults]  # e.g. [64, 128, 192]
        g, td, dr, nb = rcfg.groups, rcfg.t_dim, rcfg.dropout, rcfg.blocks_per_level

        self.embed = ConditionEmbed(td, rcfg.d_ctx,
                                    vars(rcfg).get('n_ctx_tokens', 0))
        self.in_conv = nn.Conv2d(rcfg.cond_channels, w[0], 3, padding=1)

        def blocks(c_in, c_out):
            return nn.ModuleList(
                [FiLMResBlock(c_in if i == 0 else c_out, c_out, td, g, dr)
                 for i in range(nb)]
            )

        self.enc1 = blocks(w[0], w[0])
        self.down1 = nn.Conv2d(w[0], w[1], 3, stride=2, padding=1)
        self.enc2 = blocks(w[1], w[1])
        self.down2 = nn.Conv2d(w[1], w[2], 3, stride=2, padding=1)
        self.enc3 = blocks(w[2], w[2])
        self.mid = blocks(w[2], w[2])
        # Decoder blocks each consume one encoder skip via channel-concat.
        self.dec3 = nn.ModuleList([FiLMResBlock(2 * w[2], w[2], td, g, dr)
                                   for _ in range(nb)])
        self.up1 = nn.Conv2d(w[2], w[1], 3, padding=1)
        self.dec2 = nn.ModuleList([FiLMResBlock(2 * w[1], w[1], td, g, dr)
                                   for _ in range(nb)])
        self.up2 = nn.Conv2d(w[1], w[0], 3, padding=1)
        self.dec1 = nn.ModuleList([FiLMResBlock(2 * w[0], w[0], td, g, dr)
                                   for _ in range(nb)])

        self.out_norm = nn.GroupNorm(min(g, w[0]), w[0])
        self.out_conv = nn.Conv2d(w[0], rcfg.in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)                   # v == 0 at init
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x_t: torch.Tensor,          # [B, C, H, W]  noisy detail
        pred_hfm: torch.Tensor,     # [B, C, H, W]  frozen base prediction
        x_in: torch.Tensor,         # [B, C, H, W]  frame the base stepped from
        pixel_mask: torch.Tensor,   # [B or 1, 1, H, W]
        t: torch.Tensor,            # [B] in [0, 1]
        ctx_vec: Optional[torch.Tensor],  # [B, d_ctx] pooled or [B, K, d_ctx] full
    ) -> torch.Tensor:
        B = x_t.shape[0]
        m = pixel_mask.to(x_t.dtype).expand(B, 1, x_t.shape[2], x_t.shape[3])
        h = self.in_conv(torch.cat([x_t, pred_hfm, x_in, m], dim=1))
        emb = self.embed(t, ctx_vec)

        skips = []
        for b in self.enc1:
            h = b(h, emb); skips.append(h)
        h = self.down1(h)
        for b in self.enc2:
            h = b(h, emb); skips.append(h)
        h = self.down2(h)
        for b in self.enc3:
            h = b(h, emb); skips.append(h)
        for b in self.mid:
            h = b(h, emb)
        for b in self.dec3:
            h = b(torch.cat([h, skips.pop()], dim=1), emb)
        h = self.up1(F.interpolate(h, scale_factor=2, mode='nearest'))
        for b in self.dec2:
            h = b(torch.cat([h, skips.pop()], dim=1), emb)
        h = self.up2(F.interpolate(h, scale_factor=2, mode='nearest'))
        for b in self.dec1:
            h = b(torch.cat([h, skips.pop()], dim=1), emb)

        return self.out_conv(F.silu(self.out_norm(h)))         # [B, C, H, W]


# ---------------------------------------------------------------------------
# Objective + sampler
# ---------------------------------------------------------------------------

def cfm_loss(
    refiner: nn.Module,
    x_target: torch.Tensor,
    pred_hfm: torch.Tensor,
    x_in: torch.Tensor,
    pixel_mask: torch.Tensor,
    ctx_vec: Optional[torch.Tensor],
    sigma_d: torch.Tensor,          # [C]
    metrics: Optional[dict] = None,
) -> torch.Tensor:
    """Rectified-flow matching loss on the detail residual.  fp32 targets/paths;
    the refiner forward may run under the caller's autocast."""
    B = x_target.shape[0]
    m = pixel_mask.to(torch.float32)
    if m.shape[0] != B:
        m = m.expand(B, -1, -1, -1)

    d_t = ((x_target - pred_hfm).float() * m) / sigma_d.view(1, -1, 1, 1)
    z = torch.randn_like(d_t) * m
    t = torch.rand(B, device=x_target.device)
    tb = t[:, None, None, None]
    x_t = (1.0 - tb) * z + tb * d_t
    u = d_t - z

    v = refiner(x_t, pred_hfm, x_in, m, t, ctx_vec).float()

    w = m.expand_as(v)
    loss = ((v - u).pow(2) * w).sum() / w.sum().clamp_min(1.0)
    if metrics is not None:
        with torch.no_grad():
            for lo, hi, key in [(0.0, 0.5, 'loss_t_lo'), (0.5, 1.0, 'loss_t_hi')]:
                sel = (t >= lo) & (t < hi)
                if bool(sel.any()):
                    ws = w[sel]
                    metrics[key] = float(
                        (((v - u).pow(2) * w)[sel].sum() / ws.sum().clamp_min(1.0)).item()
                    )
    return loss


@torch.no_grad()
def sample_detail(
    refiner: nn.Module,
    pred_hfm: torch.Tensor,
    x_in: torch.Tensor,
    pixel_mask: torch.Tensor,
    ctx_vec: Optional[torch.Tensor],
    sigma_d: torch.Tensor,          # [C]
    n_steps: int = 6,
    generator: Optional[torch.Generator] = None,
    z_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Euler-integrate the learned velocity field from noise to detail.

    Returns the detail in DATA units (sigma_d applied):
        refined = (pred_hfm + sample_detail(...)) * mask
    `z_init` overrides the initial noise (z=0 gives the deterministic path,
    used for verification).
    """
    B, C, H, W = pred_hfm.shape
    m = pixel_mask.to(torch.float32)
    if m.shape[0] != B:
        m = m.expand(B, -1, -1, -1)

    if z_init is not None:
        x = z_init.float() * m
    else:
        x = torch.randn(B, C, H, W, device=pred_hfm.device,
                        generator=generator, dtype=torch.float32) * m

    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=pred_hfm.device)
        v = refiner(x, pred_hfm, x_in, m, t, ctx_vec).float() * m
        x = x + dt * v

    return x * sigma_d.view(1, -1, 1, 1) * m


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of a module's parameters.  Few-step ODE
    sampling is visibly sensitive to weight noise; always sample from EMA."""

    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for k, v in module.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {k: v.clone().float() for k, v in sd.items()}

    def copy_to(self, module: nn.Module) -> None:
        module.load_state_dict({k: v for k, v in self.shadow.items()})


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    rcfg = RefinerConfig()
    net = RefinerUNet(rcfg)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'params: {n_params / 1e6:.2f}M')

    B, C, H, W = 2, 4, 256, 256
    x_t = torch.randn(B, C, H, W)
    pred = torch.randn(B, C, H, W)
    x_in = torch.randn(B, C, H, W)
    mask = (torch.rand(1, 1, H, W) > 0.1).float()
    t = torch.rand(B)
    ctx = torch.randn(B, rcfg.d_ctx)

    v = net(x_t, pred, x_in, mask, t, ctx)
    print('forward:', tuple(v.shape), ' zero-init max|v| =', float(v.abs().max()))

    sigma = torch.full((C,), 0.05)
    loss = cfm_loss(net, torch.randn(B, C, H, W), pred, x_in, mask, ctx, sigma)
    print(f'cfm loss at init: {loss.item():.3f}  (expect ~ E|u|^2 = 1 + E|d~|^2)')
    loss.backward()
    n_grad = sum(1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f'params with grad: {n_grad}/{sum(1 for _ in net.parameters())}')

    d_hat = sample_detail(net, pred, x_in, mask, ctx, sigma, n_steps=4)
    print('sample:', tuple(d_hat.shape),
          ' holes zero:', bool((d_hat * (1 - mask)).abs().max() == 0))
