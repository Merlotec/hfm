"""
Training loop for the Hierarchical Fluid Model.

Each step:
  1. ContextEncoder encodes n_context_frames ground-truth frames → context [B, K, d_ctx]
  2. HFM predicts the next frame from frames[n_context_frames] conditioned on context
  3. Reconstruction loss (+ adversarial loss once GAN is active)

The context encoder and main model are optimised jointly — gradients from the
prediction loss flow freely through both.  No warmup BPTT or sys/resid logic.
"""

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Optional, Tuple

from .model import HFM
from .quadtree import QuadtreeHFM
from .context_encoder import ContextEncoder
from .discriminator import HFMDiscriminator
from .config import HFMConfig
from .distributed import allreduce_grads, allreduce_stats, host_grad_sync_enabled


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FluidLoss(nn.Module):
    """MSE + L1 reconstruction loss over valid (non-hole) pixels only, with
    optional gradient-difference and tile-mean terms.

    The tile-mean term penalises the PER-TILE average error: the error field is
    mean-pooled over the decoder's 16 px token tiles (mask-weighted, so partially
    masked tiles average over their fluid pixels only) and charged with L2 + L1.
    Motivation, measured on rollouts: 80-90% of the rho error variance is a
    per-tile DC offset -- each decoded tile sits at a slightly wrong level.  Per
    step these offsets are spatially smooth and pixelwise tiny, so MSE barely
    notices them, but through autoregressive feedback adjacent tiles decorrelate
    and surface as a 16 px blob grid from ~10 steps out (beyond the BPTT
    horizon).  Pooling first makes the offset THE quantity being penalised
    instead of a rounding error inside the pixel loss, attacking the artifact at
    its per-step seed.

    The gradient term penalises WRONG spatial gradients, |grad(pred) - grad(gt)|,
    not gradient magnitude.  That distinction is what lets it suppress grain
    without blurring sharp features: speckle manufactures thousands of small
    gradients absent from the target and pays for every one, while a sharp shear
    layer is REWARDED for being sharp, since matching the target's steep gradient
    reduces the term.  L1 on the differences (not L2) so the few large errors at
    true edges do not drown the many small ones grain produces.  Differences are
    counted only where BOTH pixels of the pair are fluid, so mask boundaries and
    collider rims contribute nothing.
    """

    def __init__(
        self,
        l1_weight: float = 0.1,
        pixel_mask: Optional[torch.Tensor] = None,
        grad_weight: float = 0.0,
        tile_weight: float = 0.0,
        tile_px: int = 16,
        **kwargs,  # absorb removed hole_weight / hole_fill_sigma args
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.tile_weight = tile_weight
        self.tile_px = tile_px
        if pixel_mask is not None:
            self.register_buffer('pixel_mask', pixel_mask)
        else:
            self.pixel_mask: Optional[torch.Tensor] = None

    def _tile_term(self, pred: torch.Tensor, target: torch.Tensor,
                   w: Optional[torch.Tensor]) -> torch.Tensor:
        d = pred - target
        k = self.tile_px
        if w is None:
            tm = F.avg_pool2d(d, k)                        # per-tile mean error
            return tm.pow(2).mean() + self.l1_weight * tm.abs().mean()
        # Mask-weighted tile mean: average the error over each tile's FLUID
        # pixels, and drop tiles that are entirely hole.  A plain avg_pool would
        # dilute rim tiles toward zero and reward nothing.
        ws = F.avg_pool2d(w, k)                            # fluid fraction per tile
        tm = F.avg_pool2d(d * w, k) / ws.clamp_min(1e-6)
        tw = (ws > 0).to(d.dtype)                          # tile has any fluid
        n = tw.expand_as(tm).sum().clamp_min(1.0)
        return ((tm.pow(2) * tw).sum() / n
                + self.l1_weight * (tm.abs() * tw).sum() / n)

    def _grad_term(self, pred: torch.Tensor, target: torch.Tensor,
                   w: Optional[torch.Tensor]) -> torch.Tensor:
        d = pred - target
        gx = d[:, :, :, 1:] - d[:, :, :, :-1]      # d/dx of the error field
        gy = d[:, :, 1:, :] - d[:, :, :-1, :]      # d/dy
        if w is None:
            return gx.abs().mean() + gy.abs().mean()
        wx = w[:, :, :, 1:] * w[:, :, :, :-1]      # pair-valid: both pixels fluid
        wy = w[:, :, 1:, :] * w[:, :, :-1, :]
        nx = wx.expand_as(gx).sum().clamp_min(1.0)
        ny = wy.expand_as(gy).sum().clamp_min(1.0)
        return (gx.abs() * wx).sum() / nx + (gy.abs() * wy).sum() / ny

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`pixel_mask` overrides the constructor buffer -- pass the PER-SAMPLE mask
        [B, 1, H, W] when a batch mixes geometries, otherwise every sample is scored
        against the wrong collider."""
        m = pixel_mask if pixel_mask is not None else self.pixel_mask
        if m is None:
            d = pred - target
            loss = d.pow(2).mean() + self.l1_weight * d.abs().mean()
            if self.grad_weight > 0.0:
                loss = loss + self.grad_weight * self._grad_term(pred, target, None)
            if self.tile_weight > 0.0:
                loss = loss + self.tile_weight * self._tile_term(pred, target, None)
            return loss
        # Boolean indexing flattens across samples; weight instead so a per-sample
        # mask broadcasts and the normaliser counts only that sample's fluid pixels.
        w = m[:, :1].to(pred.dtype)
        if w.shape[0] != pred.shape[0]:
            w = w.expand(pred.shape[0], -1, -1, -1)
        d = pred - target
        n = w.expand_as(d).sum().clamp_min(1.0)
        loss = ((d.pow(2) * w).sum() / n) + self.l1_weight * ((d.abs() * w).sum() / n)
        if self.grad_weight > 0.0:
            loss = loss + self.grad_weight * self._grad_term(pred, target, w)
        if self.tile_weight > 0.0:
            loss = loss + self.tile_weight * self._tile_term(pred, target, w)
        return loss


# ---------------------------------------------------------------------------
# Stride probe: does the context actually encode the timestep?
# ---------------------------------------------------------------------------

# Fixed affine standardisation of the log-timestep regression target.  dt = s *
# save_t with s in {1,2,4} and save_t ~ U(0.01, 0.2), so log(dt) runs about
# -4.6 .. -0.2 with mean ~ -2.5 and spread ~ 1.  Without this the raw target
# sits near -4.6 while a fresh zero-init head predicts ~0, so the probe term
# starts at ~20 and swamps the reconstruction loss (~0.01) for the first
# thousand steps.  CONSTANTS, not batch statistics: a running normaliser would
# make the loss scale drift and the metric incomparable across runs.
LOG_DT_MEAN = -2.5
LOG_DT_STD  = 1.0


class ContextProbe(nn.Module):
    """
    Recover from the context tokens alone the quantities the context must carry:

      dt : the physical timestep the transition spans, as log(dt) (REGRESSION).
           gen_alternating draws save_t per segment from a continuous range and
           the trainer samples a stride s on top, so the timestep is a continuous
           quantity dt = s * save_t -- not a member of a small fixed set.  A
           classifier over discrete strides cannot represent that, and would also
           be degenerate here: the same stride means different physical dt in
           different runs, so the label would not identify the prediction task.
           Regressing log(dt) makes the target scale-free (dt spans ~1.5 decades),
           and lets the model interpolate and extrapolate along the timestep axis
           rather than memorising a handful of classes.

      ctx: the hidden per-segment PHYSICS (regression on 8 z-scored parameters,
           plus a 4-way classification of the viscosity model).  This is the label
           gen_alternating exists to teach: viscosity, its shear-thinning
           exponent, bulk viscosity, thermal conductivity, C_v and gamma are all
           resampled per segment and are invisible in any single frame -- they can
           only be inferred from how the field MOVES.  Without this head the probe
           supervised only dt and the freestream.

      bc : per-run boundary-condition summary (regression, z-scored).  Note this
           is fixed per TRAJECTORY, not per segment (gen_alternating draws the
           freestream once and holds it across the chain), so it is genuine
           physics but it also identifies the trajectory -- read it alongside the
           ctx head rather than on its own.

    Trained jointly with a small weight, each head both MEASURES whether the
    information is present in the context and applies gradient pressure on the
    encoder to put it there: a collapsed/constant context satisfies neither.
    """

    def __init__(self, d_ctx: int, bc_dim: int = 0, ctx_dim: int = 0,
                 n_visc_models: int = 0, **_legacy):
        super().__init__()
        h = d_ctx // 2
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_ctx), nn.Linear(d_ctx, h), nn.GELU(),
        )
        self.dt_head   = nn.Linear(h, 1)
        self.bc_head   = nn.Linear(h, bc_dim) if bc_dim > 0 else None
        self.ctx_head  = nn.Linear(h, ctx_dim) if ctx_dim > 0 else None
        self.visc_head = nn.Linear(h, n_visc_models) if n_visc_models > 0 else None

    def forward(self, context: torch.Tensor) -> dict:
        """context: [B, K, d_ctx] -> dict of head outputs."""
        z = self.trunk(context.mean(dim=1))
        out = {'log_dt': self.dt_head(z).squeeze(-1)}
        if self.bc_head is not None:
            out['bc'] = self.bc_head(z)
        if self.ctx_head is not None:
            out['ctx'] = self.ctx_head(z)
        if self.visc_head is not None:
            out['visc'] = self.visc_head(z)
        return out


def probe_losses(
    probe: nn.Module,
    context: torch.Tensor,
    dt: Optional[torch.Tensor] = None,
    bc: Optional[torch.Tensor] = None,
    ctx: Optional[torch.Tensor] = None,
    ctx_cls: Optional[torch.Tensor] = None,
    metrics: Optional[dict] = None,
) -> torch.Tensor:
    """
    Combined probe loss.  Both terms are z-scale-ish regressions, so they are
    summed unweighted and the caller applies one overall weight.  Non-finite
    rows (runs with unreadable BCs) are masked out per sample.

    dt: [B] physical timestep of the transition being predicted (seconds of
        simulation time).  Regressed in log space.
    """
    out = probe(context)
    dev = context.device
    total = context.sum() * 0.0            # typed zero on the right device/graph

    if dt is not None:
        log_dt = torch.log(dt.to(dev).float().clamp_min(1e-8))
        tgt  = (log_dt - LOG_DT_MEAN) / LOG_DT_STD
        pred = out['log_dt'].float()
        err  = F.mse_loss(pred, tgt)
        total = total + err
        if metrics is not None:
            with torch.no_grad():
                # median |dt_pred/dt - 1|: the interpretable number, and median
                # rather than mean so one badly-predicted sample cannot dominate
                # while the head is still warming up.
                rel = (((pred - tgt) * LOG_DT_STD).exp() - 1.0).abs().median()
                metrics['dt_mse'] = float(err.item())
                metrics['dt_rel_err'] = float(rel.item())

    if 'bc' in out and bc is not None:
        tgt = bc.to(dev).float()
        valid = torch.isfinite(tgt).all(dim=-1)                     # [B]
        if bool(valid.any()):
            err = (out['bc'].float()[valid] - tgt[valid]).pow(2).mean()
            total = total + err
            if metrics is not None:
                metrics['bc_mse'] = float(err.item())

    if 'ctx' in out and ctx is not None:
        tgt = ctx.to(dev).float()
        valid = torch.isfinite(tgt).all(dim=-1)                     # [B]
        if bool(valid.any()):
            err = (out['ctx'].float()[valid] - tgt[valid]).pow(2).mean()
            total = total + err
            if metrics is not None:
                metrics['ctx_mse'] = float(err.item())

    if 'visc' in out and ctx_cls is not None:
        lbl = ctx_cls.to(dev).long()
        valid = lbl >= 0                     # -1 marks "not recorded for this run"
        if bool(valid.any()):
            logits = out['visc'].float()[valid]
            err = F.cross_entropy(logits, lbl[valid])
            total = total + err
            if metrics is not None:
                metrics['visc_ce'] = float(err.item())
                metrics['visc_acc'] = float(
                    (logits.argmax(-1) == lbl[valid]).float().mean().item())

    return total


# ---------------------------------------------------------------------------
# GAN training step
# ---------------------------------------------------------------------------

def train_step_gan(
    model: HFM,
    context_encoder: ContextEncoder,
    discriminator: Optional[HFMDiscriminator],
    frames: List[torch.Tensor],
    gen_optimizer: optim.Optimizer,
    disc_optimizer: Optional[optim.Optimizer],
    criterion: nn.Module,
    n_context: int,
    adv_weight: float = 0.0,
    clip_grad: float = 1.0,
    pixel_mask: Optional[torch.Tensor] = None,
    disc_update_threshold: float = 0.5,
    self_input_prob: float = 0.0,
    rollout_horizon: int = 1,
    persist_norm: bool = False,
    accum_steps: int = 1,
    micro_first: bool = True,
    micro_last: bool = True,
    probe: Optional[nn.Module] = None,
    probe_dt: Optional[torch.Tensor] = None,
    probe_bc: Optional[torch.Tensor] = None,
    probe_ctx: Optional[torch.Tensor] = None,
    probe_ctx_cls: Optional[torch.Tensor] = None,
    stride_cls_weight: float = 0.0,
    metrics: Optional[dict] = None,
) -> Tuple[float, float]:
    """
    One training step.

    frames[0 .. n_context-1]      → ContextEncoder → context (encoded ONCE)
    frames[n_context]             → HFM(context)   → pred_0
    frames[n_context + 1 + k]     → target for step k

    rollout_horizon: number of autoregressive steps to unroll with gradient.
    With horizon H the model predicts H frames, feeding each prediction back as
    the next input, and the reconstruction loss is averaged over all H targets.
    Gradient flows through the whole chain (backprop-through-time), so the model
    is optimised to propagate the dynamics rather than to reproduce a single
    near-identity step — and because the one context is reused across H evolving
    states, it is forced to encode the invariant dynamics (BCs/forcing) instead
    of collapsing to a constant.  horizon=1 recovers the original single-step
    behaviour exactly.

    self_input_prob: scheduled sampling (exposure-bias fix).  With this
    probability the FIRST input frame is replaced by the model's own no-grad
    prediction of it (from frames[n_context-1]).  Largely subsumed by horizon>1,
    but kept for horizon=1.

    accum_steps / micro_first / micro_last: gradient accumulation.  Over a window
    of `accum_steps` micro-batches the grads are accumulated (losses scaled by
    1/accum_steps) and the optimizers step ONLY on the last micro-batch — so the
    expensive host-staged allreduce_grads runs once per window instead of once per
    micro-batch, cutting the sync count `accum_steps`×.  Defaults (1/True/True)
    reproduce the original single-step behaviour exactly.

    NOTE: this saves comms only on the host-grad-sync path (no DDP wrapper).  Under
    a real DDP wrapper the reducer still fires every backward; accumulation stays
    numerically correct there but does not reduce traffic without model.no_sync().

    Returns (recon_loss, disc_loss).  disc_loss is 0.0 when adv_weight == 0.
    """
    scale = 1.0 / max(1, accum_steps)
    if micro_first:
        gen_optimizer.zero_grad()
        if disc_optimizer is not None:          # None when the GAN is disabled
            disc_optimizer.zero_grad()

    device_type = frames[0].device.type
    # bf16 autocast on CUDA *and* XPU — PVC's matrix engines run bf16; leaving this
    # cuda-only silently trains fp32 on Dawn at a large throughput cost.
    amp = device_type in ('cuda', 'xpu')

    x_in     = frames[n_context]
    x_target = frames[n_context + 1]

    # ---- encode context (gradients flow through context_encoder) ----
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        context = context_encoder(frames[:n_context], pixel_mask=pixel_mask)

    # ---- context probe: recover stride / mesh / BCs from the context alone ----
    # frames[] is already stride-subsampled by the caller; the stride reaches
    # the encoder only through the inter-frame motion.  The probe both measures
    # and enforces that the context carries each labelled quantity.
    probe_term = None
    if probe is not None and probe_dt is not None and stride_cls_weight > 0.0:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            probe_term = probe_losses(probe, context, dt=probe_dt, bc=probe_bc,
                                      ctx=probe_ctx, ctx_cls=probe_ctx_cls,
                                      metrics=metrics)

    # ---- scheduled sampling: sometimes feed the model its own prediction ----
    # frames[n_context-1] always exists (n_context >= 1); the supervision target
    # stays the TRUE frame n_context+1, exactly the pushforward setup.
    if self_input_prob > 0.0 and float(torch.rand(())) < self_input_prob:
        with torch.no_grad(), torch.autocast(device_type=device_type,
                                             dtype=torch.bfloat16, enabled=amp):
            x_in = model(frames[n_context - 1], context, pixel_mask=pixel_mask).float()
        if pixel_mask is not None:
            x_in = x_in * pixel_mask

    # ---- unrolled prediction (backprop-through-time over the horizon) ----
    # The reconstruction loss is the mean over all horizon targets.  pred_0 (the
    # first step) is also what the discriminator / adv term operate on below, so
    # the GAN behaviour is unchanged relative to single-step training.
    horizon = max(1, min(rollout_horizon, len(frames) - n_context - 1))
    recon_terms = []                      # raw loss, logged
    norm_terms = []                       # persistence-normalised, trained
    pred_masked = None
    x_cur = x_in
    for k in range(horizon):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            pred_k = model(x_cur, context, pixel_mask=pixel_mask)
        pred_k = pred_k.float() * pixel_mask if pixel_mask is not None else pred_k.float()
        target = frames[n_context + 1 + k]
        term = criterion(pred_k, target, pixel_mask=pixel_mask)
        recon_terms.append(term)
        if persist_norm:
            # Per-term persistence baseline from the CLEAN input frame (not the
            # scheduled-sampling replacement), so the normaliser is a stable
            # scale reference.  Floor guards near-static windows from blowing
            # the division up; typical baselines run 0.01 (s=1) to 0.1+ (s=4).
            with torch.no_grad():
                base = criterion(frames[n_context], target, pixel_mask=pixel_mask)
            norm_terms.append(term / base.clamp_min(5e-3))
        if pred_masked is None:
            pred_masked = pred_k          # first step feeds the GAN, as before
        x_cur = pred_k                    # feed prediction forward (keeps grad)
    assert pred_masked is not None
    recon_rollout = torch.stack(recon_terms).mean()
    # What the generator actually optimises: relative-to-persistence error when
    # normalisation is on (== the logged `ratio`), else the raw loss.
    gen_recon = torch.stack(norm_terms).mean() if persist_norm else recon_rollout

    def _zero_and_restore() -> None:
        gen_optimizer.zero_grad()
        if disc_optimizer is not None:
            disc_optimizer.zero_grad()
        if discriminator is not None:
            for p in discriminator.parameters():
                p.requires_grad_(True)

    # Modules the generator backward writes grads into (probe included so its
    # grads are averaged and clipped alongside the rest on every path).
    gen_modules = [model, context_encoder] + ([probe] if probe is not None else [])
    gen_params  = [p for mod in gen_modules for p in mod.parameters()]

    # ---- reconstruction only ----
    if adv_weight == 0.0:
        recon_loss = recon_rollout        # raw value, for the log
        gen_loss = gen_recon if probe_term is None \
            else gen_recon + stride_cls_weight * probe_term
        # NaN bail must be GLOBAL: if one rank bailed while others proceeded to a
        # collective (DDP reducer or host allreduce), the job would hang/diverge.
        (bad,) = allreduce_stats(0.0 if torch.isfinite(gen_loss) else 1.0)
        if bad > 0.0:
            _zero_and_restore()
            return float('nan'), 0.0
        (gen_loss * scale).backward()
        if micro_last:
            if host_grad_sync_enabled():
                allreduce_grads(gen_modules)   # average, THEN clip (DDP semantics)
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(gen_params, clip_grad)
            gen_optimizer.step()
        return recon_loss.item(), 0.0

    # ---- discriminator update ----
    ctx_detach = context.detach()

    # The discriminator must see real and fake with IDENTICAL hole treatment.
    # Raw target frames carry the renderer's constant fill at hole pixels (raw 0 =>
    # up to -4.4 sigma after normalisation) while pred_masked has exact 0 there —
    # an unmasked x_target lets the disc win by reading a single hole pixel.  The
    # generator can't change its holes (the mask zeroes their gradient), so the adv
    # term instead distorts the fluid ring around the collider (the visible hue) and
    # the health gate keeps collapsing.  Mask both sides so holes are non-informative.
    x_target_d = x_target * pixel_mask if pixel_mask is not None else x_target
    x_in_d     = x_in     * pixel_mask if pixel_mask is not None else x_in

    for p in discriminator.parameters():
        p.requires_grad_(True)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        real_logit = discriminator(x_target_d,            x_in_d, ctx_detach)
        fake_logit = discriminator(pred_masked.detach(),  x_in_d, ctx_detach)

    real_labels = torch.full_like(real_logit, 0.9)
    d_loss = (
        F.binary_cross_entropy_with_logits(real_logit, real_labels) +
        F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
    )
    d_loss_val = d_loss.item()

    # The health gate and NaN bail must be decided on the GLOBAL disc loss: gating
    # on the local value lets ranks take different branches around a backward/step,
    # which desyncs weights (and hangs any collective).  Mean over ranks.
    finite = math.isfinite(d_loss_val)
    n, d_sum, bad_sum = allreduce_stats(1.0, d_loss_val if finite else 0.0,
                                        0.0 if finite else 1.0)
    if bad_sum > 0.0:
        _zero_and_restore()
        return float('nan'), float('nan')
    d_gate = d_sum / n                       # global mean disc loss

    # Accumulate the disc grad on healthy micro-batches; step once per window.
    if disc_update_threshold < d_gate < 2.0:
        (d_loss * scale).backward()
    if micro_last:
        # allreduce_grads / clip / step are no-ops when no micro in this window was
        # healthy (grads absent) — and the health gate is GLOBAL, so grad presence is
        # identical across ranks, keeping the collective consistent.
        if host_grad_sync_enabled():
            allreduce_grads([discriminator])
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
        disc_optimizer.step()

    # ---- generator update ----
    for p in discriminator.parameters():
        p.requires_grad_(False)

    recon_loss = recon_rollout            # raw value, for the log

    disc_healthy = disc_update_threshold < d_gate < 2.0   # same GLOBAL gate as above
    if disc_healthy:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            adv_logit = discriminator(pred_masked, x_in_d, context)
        adv_loss = F.binary_cross_entropy_with_logits(
            adv_logit, torch.ones_like(adv_logit)
        )
        total_loss = gen_recon + adv_weight * adv_loss
    else:
        total_loss = gen_recon
    if probe_term is not None:
        total_loss = total_loss + stride_cls_weight * probe_term
    (bad,) = allreduce_stats(0.0 if torch.isfinite(total_loss) else 1.0)
    if bad > 0.0:
        _zero_and_restore()
        return float('nan'), d_loss_val

    (total_loss * scale).backward()
    if micro_last:
        if host_grad_sync_enabled():
            allreduce_grads(gen_modules)
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(gen_params, clip_grad)
        gen_optimizer.step()

    for p in discriminator.parameters():
        p.requires_grad_(True)

    return recon_loss.item(), d_loss_val


# ---------------------------------------------------------------------------
# GAN Trainer class
# ---------------------------------------------------------------------------

class GANTrainer:
    """
    Training wrapper pairing HFM + ContextEncoder with HFMDiscriminator.

    GAN curriculum
    --------------
    Stage 1 (step < gan_start_step): reconstruction only
    Stage 2 (step >= gan_start_step): adv_weight ramps 0 → cfg.disc_adv_weight
        over gan_ramp_steps steps.

    Usage
    -----
    trainer = GANTrainer(cfg, ...)
    trainer.to(device)
    for batch in dataloader:          # batch: [B, T, C, H, W]
        frames = [batch[:, t] for t in range(T)]
        recon, disc = trainer.step(frames)
    """

    def __init__(
        self,
        cfg: HFMConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        l1_weight: float = 0.1,
        grad_weight: float = 0.0,
        tile_weight: float = 0.0,
        hole_weight: float = 0.1,
        hole_fill_sigma: float = 15.0,
        gan_start_step: int = 10_000,
        gan_ramp_steps: int = 2_000,
        disc_update_threshold: float = 0.5,
        pixel_mask: Optional[torch.Tensor] = None,
        self_input_prob: float = 0.5,
        cosine_t_max: int = 10_000,
        accum_steps: int = 1,
        use_gan: bool = True,
        probe_bc_dim: int = 0,
        probe_ctx_dim: int = 0,
        probe_n_visc_models: int = 0,
        **_legacy_probe_args,
    ):
        self.self_input_prob  = self_input_prob
        self.cosine_t_max     = cosine_t_max
        self.accum_steps      = max(1, int(accum_steps))
        self._micro           = 0     # position within the current accumulation window
        self.use_gan          = use_gan
        self.cfg              = cfg
        self.model            = QuadtreeHFM(cfg) if cfg.use_quadtree else HFM(cfg)
        # Loud, once, at construction: the flat-HFM run that ignored
        # residual_prediction silently cost a full training run.  This prints the
        # flag the MODEL actually holds, not what the config claims.
        print(f'[trainer] model={type(self.model).__name__}  '
              f'residual_prediction={getattr(self.model, "residual_prediction", "MISSING")}  '
              f'time_strides={getattr(cfg, "time_strides", (1,))}')
        self.context_encoder  = ContextEncoder(cfg)
        # Complete GAN toggle: with use_gan=False the discriminator is never built,
        # trained, saved, or applied — training is pure reconstruction.
        self.discriminator    = HFMDiscriminator(cfg) if use_gan else None
        self.criterion        = FluidLoss(
            l1_weight, pixel_mask=pixel_mask, grad_weight=grad_weight,
            tile_weight=tile_weight,
            hole_weight=hole_weight, hole_fill_sigma=hole_fill_sigma,
        )

        # Context probe.  Always built now: the timestep is continuous
        # (dt = stride * run save_t), so it varies sample to sample even with a
        # single stride, and the BC head is useful regardless.
        strides = tuple(getattr(cfg, 'time_strides', (1,)) or (1,))
        self.time_strides = strides
        self.stride_probe = ContextProbe(
            cfg.d_ctx, bc_dim=probe_bc_dim, ctx_dim=probe_ctx_dim,
            n_visc_models=probe_n_visc_models)
        self._stride_metrics: dict = {}
        self._last_stride: int = strides[0]

        gen_params = list(self.model.parameters()) + list(self.context_encoder.parameters())
        if self.stride_probe is not None:
            gen_params += list(self.stride_probe.parameters())
        self.gen_optimizer = optim.AdamW(gen_params, lr=lr, weight_decay=weight_decay)
        self.disc_optimizer = (
            optim.Adam(self.discriminator.parameters(), lr=cfg.disc_lr, betas=(0.5, 0.999))
            if self.discriminator is not None else None
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.gen_optimizer, T_max=cosine_t_max
        )

        self.gan_start_step        = gan_start_step
        self.gan_ramp_steps        = gan_ramp_steps
        self.disc_update_threshold = disc_update_threshold
        self.global_step           = 0
        # Normalisation stats this run trains with — saved into the checkpoint so
        # inference cannot silently use a different normalisation (see infer.load_stats).
        self.norm_mean: Optional[torch.Tensor] = None
        self.norm_std:  Optional[torch.Tensor] = None

    def _no_sync_if(self, active: bool):
        """Context manager that enters ``.no_sync()`` on any DDP-wrapped module when
        ``active`` — used to skip the reducer's allreduce on non-final accumulation
        micro-batches.  A no-op when modules are not DDP-wrapped (host-grad-sync or
        single-process), so it is safe on every path."""
        from torch.nn.parallel import DistributedDataParallel as _DDP
        stack = contextlib.ExitStack()
        if active:
            for m in (self.model, self.context_encoder, self.discriminator,
                      self.stride_probe):
                if isinstance(m, _DDP):
                    stack.enter_context(m.no_sync())
        return stack

    def _current_adv_weight(self) -> float:
        if not self.use_gan or self.global_step < self.gan_start_step:
            return 0.0
        steps_in = self.global_step - self.gan_start_step
        ramp = min(1.0, steps_in / max(1, self.gan_ramp_steps))
        return self.cfg.disc_adv_weight * ramp

    @torch.no_grad()
    def validate(
        self,
        dataloader,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Held-out metrics: {'recon', 'persist', 'ratio'}.

        `ratio` is the number to read.  The raw loss is not comparable across
        anything: the persistence baseline spans ~7500x within a single run
        (3.59 over the startup transient, 0.00048 once the flow is developed) and
        scales with dt, which now varies per run.  So a raw val loss mostly
        reports which windows landed in the set.  ratio = recon / persistence
        divides that scale out, is the SAME quantity the training objective
        minimises when persist_norm_loss is on, and reads absolutely: 1.0 means
        "no better than copying the input", < 1 means it genuinely beats it.

        COLLECTIVE under DDP: every rank must call this, with a dataloader
        holding its own DISJOINT slice (dm.val_dataloader(shard=True)); the
        three sums are allreduced at the end and every rank returns the same
        reduced numbers.  The sharding lives in the DATALOADER, not here: the
        old batch-skip scheme (score batch i on rank i % world) still fetched
        and rendered every batch on every rank, so it saved almost nothing.
        """
        self.model.eval()
        self.context_encoder.eval()
        device      = next(self.model.parameters()).device
        device_type = device.type
        amp         = device_type in ('cuda', 'xpu')
        nc      = self.cfg.n_context_frames
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank  = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        tot_recon, tot_persist, count = 0.0, 0.0, 0
        for batch in dataloader:
            # (frames, mesh_id, ...) when the datamodule tags labels, else frames
            if isinstance(batch, (tuple, list)):
                batch, mesh_b = batch[0], batch[1]
            else:
                mesh_b = None
            mask     = self._mask_ctx(pixel_mask, mesh_b, val=True)
            frames   = [batch[:, t].to(device) for t in range(batch.shape[1])]
            horizon  = max(1, min(getattr(self.cfg, 'rollout_horizon', 1),
                                  len(frames) - nc - 1))
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
                context = self.context_encoder(frames[:nc], pixel_mask=mask)
                # Same unrolled metric as training: predict horizon steps, feeding
                # predictions forward, average over the horizon.
                x_cur, terms, bases = frames[nc], [], []
                for k in range(horizon):
                    pred = self.model(x_cur, context, pixel_mask=mask)
                    pred_m = pred.float() * mask if mask is not None else pred.float()
                    target = frames[nc + 1 + k]
                    terms.append(self.criterion(pred_m, target, pixel_mask=mask).item())
                    # Baseline from the CLEAN input frame, matched to the SAME
                    # rollout: "predict no change" scored against every horizon
                    # target, exactly as in training.
                    bases.append(self.criterion(frames[nc], target,
                                                pixel_mask=mask).item())
                    x_cur = pred_m
            loss = sum(terms) / len(terms)
            base = sum(bases) / len(bases)
            if loss == loss:   # finite check (NaN != NaN)
                tot_recon   += loss
                tot_persist += base
                count += 1
        if world > 1:
            # long=True: ranks arrive here minutes apart (each rendered its own
            # shard, at Lustre speed, possibly through corrupt-run retries); the
            # default 300 s group turned that spread into a timeout that killed
            # the whole job at an epoch boundary.
            tot_recon, tot_persist, count = allreduce_stats(
                tot_recon, tot_persist, float(count), long=True)
            count = int(count)
        if count == 0:
            return {'recon': float('nan'), 'persist': float('nan'),
                    'ratio': float('nan')}
        recon, persist = tot_recon / count, tot_persist / count
        # Ratio of the MEANS, not the mean of per-batch ratios: a batch that is
        # nearly static has a near-zero baseline, and averaging its ratio would
        # let that one batch dominate the number.
        return {'recon': recon, 'persist': persist,
                'ratio': recon / persist if persist > 0 else float('inf')}

    def to(self, device: torch.device) -> "GANTrainer":
        self.model           = self.model.to(device)
        self.context_encoder = self.context_encoder.to(device)
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(device)
        if self.stride_probe is not None:
            self.stride_probe = self.stride_probe.to(device)
        self.criterion       = self.criterion.to(device)
        return self

    def wrap_ddp(self, device: torch.device) -> "GANTrainer":
        """Wrap the three modules for DistributedDataParallel.  No-op single-process.

        Call AFTER .to(device) and AFTER any .load().  DDP reuses the same Parameter
        objects, so gen_optimizer/disc_optimizer (built in __init__ over the raw
        modules) stay valid.

        find_unused_parameters=True is required here: the generator backward touches
        no discriminator params and the discriminator backward touches no generator
        params, so each reducer sees params that never receive a grad.  This is the
        same reason the Lightning path used 'ddp_find_unused_parameters_true'.
        """
        from .distributed import wrap_ddp as _wrap
        if host_grad_sync_enabled():
            # No DDP wrapper at all: gradients are averaged manually on the host in
            # train_step_gan (gloo group).  Used where no device collective backend
            # works — DDP's own reducer/verify collectives are exactly what crashes.
            return self
        self.model           = _wrap(self.model, device, find_unused_parameters=True)
        self.context_encoder = _wrap(self.context_encoder, device, find_unused_parameters=True)
        if self.discriminator is not None:
            self.discriminator = _wrap(self.discriminator, device, find_unused_parameters=True)
        if self.stride_probe is not None:
            self.stride_probe = _wrap(self.stride_probe, device, find_unused_parameters=True)
        return self

    def set_mesh_tables(self, masks) -> None:
        """Install FVMDataModule.mesh_masks so the mask can be gathered per sample."""
        self.mesh_masks = masks

    def set_val_mesh_tables(self, masks) -> None:
        """Validation geometries are a separate 0-based set -- indexing val ids into
        the train table would silently pick unrelated geometries."""
        self.val_mesh_masks = masks

    def _mask_ctx(self, pixel_mask, mesh_ids, val: bool = False):
        """Per-sample mask for this batch, or the shared one when there are no ids."""
        masks = getattr(self, 'val_mesh_masks' if val else 'mesh_masks', None)
        if mesh_ids is None or masks is None:
            return pixel_mask
        return masks[mesh_ids.to(masks.device).long()]      # [B, 1, H, W]

    def step(
        self,
        frames: List[torch.Tensor],
        pixel_mask: Optional[torch.Tensor] = None,
        mesh_ids: Optional[torch.Tensor] = None,
        bc: Optional[torch.Tensor] = None,
        save_t: Optional[torch.Tensor] = None,
        ctx: Optional[torch.Tensor] = None,
        ctx_cls: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """frames: list of (n_context_frames + rollout_horizon) * max(time_strides) + 1
        tensors [B, C, H, W].  Each step samples one temporal stride s and trains on
        every s-th frame — the timestep reaches the model only through the context."""
        pixel_mask = self._mask_ctx(pixel_mask, mesh_ids)
        self.model.train()
        self.context_encoder.train()
        if self.discriminator is not None:
            self.discriminator.train()
        if self.stride_probe is not None:
            self.stride_probe.train()

        # ---- sample a temporal stride and subsample the frame sequence ----
        # One stride per step (whole batch shares it — the subsample is on the
        # time axis).  A random start offset uses the slack left by smaller
        # strides as temporal augmentation.
        nc, H = self.cfg.n_context_frames, getattr(self.cfg, 'rollout_horizon', 1)
        s_idx = int(torch.randint(len(self.time_strides), (1,)).item())
        s = self.time_strides[s_idx]
        need = (nc + H) * s + 1
        if need > len(frames):        # sequence too short for this stride
            s_idx, s = 0, self.time_strides[0]
            need = (nc + H) * s + 1
        off = int(torch.randint(len(frames) - need + 1, (1,)).item())
        frames = frames[off : off + need : s]
        self._last_stride = s
        self._stride_metrics = {}
        # Physical timestep of each transition in this batch.  save_t is per RUN
        # (gen_alternating samples it per segment), so dt varies across the batch
        # even though the stride does not.
        B = frames[0].shape[0]
        # .float() BEFORE .to(): collate gives float64 (Python floats), and MPS
        # rejects a float64 tensor outright rather than converting on arrival.
        dt = (save_t.float().to(frames[0].device) * s if save_t is not None
              else torch.full((B,), float(s) * 0.01, device=frames[0].device))
        self._stride_metrics['dt'] = float(dt.mean().item())
        # Persistence baseline AT THIS STRIDE — the honest "model vs copy-input"
        # comparison for the log (stride-1 persistence would flatter larger strides).
        with torch.no_grad():
            _p = [self.criterion(frames[nc], frames[nc + 1 + k],
                                 pixel_mask=pixel_mask).item()
                  for k in range(min(H, len(frames) - nc - 1))]
        self._stride_metrics['persist'] = sum(_p) / max(1, len(_p))

        micro_first = (self._micro == 0)
        micro_last  = (self._micro == self.accum_steps - 1)

        # On the DDP path, suppress the per-backward allreduce on non-final micro-batches
        # so accumulation actually saves comms (and doesn't double-mark the reducer).
        # No-op on the host-grad-sync path — those modules aren't DDP-wrapped.
        with self._no_sync_if(not micro_last):
            recon_loss, disc_loss = train_step_gan(
                self.model,
                self.context_encoder,
                self.discriminator,
                frames,
                self.gen_optimizer,
                self.disc_optimizer,
                self.criterion,
                n_context=self.cfg.n_context_frames,
                adv_weight=self._current_adv_weight(),
                pixel_mask=pixel_mask,
                disc_update_threshold=self.disc_update_threshold,
                self_input_prob=self.self_input_prob,
                rollout_horizon=getattr(self.cfg, 'rollout_horizon', 1),
                persist_norm=getattr(self.cfg, 'persist_norm_loss', False),
                accum_steps=self.accum_steps,
                micro_first=micro_first,
                micro_last=micro_last,
                probe=self.stride_probe,
                probe_dt=dt,
                probe_bc=bc,
                probe_ctx=ctx,
                probe_ctx_cls=ctx_cls,
                stride_cls_weight=getattr(self.cfg, 'stride_cls_weight', 0.0),
                metrics=self._stride_metrics,
            )

        # A NaN bail zeroed the grads inside train_step_gan — abort the window and do
        # not advance the optimizer step / LR schedule on a failed micro-batch.
        if not (recon_loss == recon_loss):   # NaN check without importing math here
            self._micro = 0
            return recon_loss, disc_loss

        # `global_step` counts OPTIMIZER steps (one per accumulation window), so the LR
        # schedule, adv-weight ramp and checkpoint cadence are all in optimizer-step
        # units regardless of accum_steps.
        self._micro += 1
        if self._micro >= self.accum_steps:
            self._micro = 0
            self.scheduler.step()
            self.global_step += 1
        return recon_loss, disc_loss

    def training_info(self) -> dict:
        info = {
            'global_step': self.global_step,
            'adv_weight':  self._current_adv_weight(),
            'gan_active':  self._current_adv_weight() > 0.0,
            'stride':      self._last_stride,
        }
        info.update(self._stride_metrics)      # dt / persist / dt_rel_err / bc_mse
        return info

    @torch.no_grad()
    def predict(
        self,
        context_frames: List[torch.Tensor],
        x: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.model.eval()
        self.context_encoder.eval()
        if pixel_mask is not None:
            context_frames = [f * pixel_mask for f in context_frames]
            x = x * pixel_mask
        context = self.context_encoder(context_frames, pixel_mask=pixel_mask)
        pred = self.model(x, context, pixel_mask=pixel_mask)
        if pixel_mask is not None:
            pred = pred * pixel_mask
        return pred

    def save(self, path: str):
        # Unwrap DDP first: a wrapped module's state_dict keys carry a 'module.'
        # prefix, which would make the checkpoint unloadable by single-process
        # code (infer.py) and by a resumed non-DDP run.
        _m = getattr(self.model, 'module', self.model)
        _c = getattr(self.context_encoder, 'module', self.context_encoder)
        ckpt = {
            'model':           _m.state_dict(),
            'context_encoder': _c.state_dict(),
            'gen_optimizer':   self.gen_optimizer.state_dict(),
            'scheduler':       self.scheduler.state_dict(),
            'cfg':             self.cfg,
            'global_step':     self.global_step,
            # Pin the normalisation so inference reproduces training exactly.
            'norm_mean':       None if self.norm_mean is None else self.norm_mean.tolist(),
            'norm_std':        None if self.norm_std  is None else self.norm_std.tolist(),
        }
        # Only persist discriminator state when the GAN is enabled.
        if self.discriminator is not None and self.disc_optimizer is not None:
            _d = getattr(self.discriminator, 'module', self.discriminator)
            ckpt['discriminator']  = _d.state_dict()
            ckpt['disc_optimizer'] = self.disc_optimizer.state_dict()
        if self.stride_probe is not None:
            _s = getattr(self.stride_probe, 'module', self.stride_probe)
            ckpt['stride_probe'] = _s.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        # Load into the raw modules — checkpoints are always saved unwrapped (see
        # save()), so loading through a DDP wrapper would look for 'module.*' keys.
        # Normally load() runs before wrap_ddp(), but stay correct either way.
        self.model           = getattr(self.model, 'module', self.model)
        self.context_encoder = getattr(self.context_encoder, 'module', self.context_encoder)
        if self.discriminator is not None:
            self.discriminator = getattr(self.discriminator, 'module', self.discriminator)
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        if unexpected:
            print(f'  [warn] model unexpected keys: {unexpected}')
        if missing:
            print(f'  New/missing model keys (random init): {missing}')

        if 'context_encoder' in ckpt:
            try:
                self.context_encoder.load_state_dict(ckpt['context_encoder'])
            except RuntimeError as e:
                print(f'  [ERROR] context encoder does not match this config: {e}')
                raise SystemExit(
                    'Cannot resume: the checkpoint was trained with a different context '
                    'encoder shape.  n_ctx_tokens changed 64 -> 16 and the probe head '
                    'changed from a discrete stride classifier to a continuous log(dt) '
                    'regressor, so both the summary table and the optimizer state are '
                    'incompatible.  Train from scratch, or set n_ctx_tokens back to the '
                    "checkpoint's value in hyperparams.json.")
        else:
            print('  [warn] no context_encoder in checkpoint; starting fresh')

        if self.stride_probe is not None and 'stride_probe' in ckpt:
            self.stride_probe = getattr(self.stride_probe, 'module', self.stride_probe)
            try:
                self.stride_probe.load_state_dict(ckpt['stride_probe'])
            except RuntimeError as e:            # e.g. n_classes changed
                print(f'  [warn] stride_probe not restored ({e}); starting fresh')

        # Discriminator load is skipped entirely when the GAN is disabled (or when the
        # checkpoint carries no discriminator, e.g. saved with use_gan=False).
        disc_ok = False
        if self.discriminator is not None and 'discriminator' in ckpt:
            disc_state = ckpt['discriminator']
            new_keys = set(self.discriminator.state_dict().keys())
            remapped = {}
            for k, v in disc_state.items():
                if k.endswith('.weight'):
                    new_key = k[:-7] + '.weight_orig'
                    remapped[new_key if new_key in new_keys else k] = v
                else:
                    remapped[k] = v
            disc_ok = True
            try:
                missing, unexpected = self.discriminator.load_state_dict(remapped, strict=False)
                non_uv = [k for k in missing if not k.endswith(('.weight_u', '.weight_v'))]
                if non_uv:
                    print(f'  [warn] discriminator missing keys: {non_uv}')
                if unexpected:
                    print(f'  [warn] discriminator unexpected keys: {unexpected}')
            except RuntimeError as e:
                print(f'  [warn] discriminator incompatible ({e}); reinitialising')
                disc_ok = False
        elif self.discriminator is None:
            print('  [info] GAN disabled (use_gan=False); discriminator not loaded')

        self.gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
        if disc_ok and self.disc_optimizer is not None and 'disc_optimizer' in ckpt:
            try:
                self.disc_optimizer.load_state_dict(ckpt['disc_optimizer'])
            except Exception:
                print('  [warn] disc_optimizer reinitialised')
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt.get('global_step', 0)
