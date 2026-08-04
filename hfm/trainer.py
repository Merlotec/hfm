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
    """MSE + L1 reconstruction loss over valid (non-hole) pixels only."""

    def __init__(
        self,
        l1_weight: float = 0.1,
        pixel_mask: Optional[torch.Tensor] = None,
        **kwargs,  # absorb removed hole_weight / hole_fill_sigma args
    ):
        super().__init__()
        self.l1_weight = l1_weight
        if pixel_mask is not None:
            self.register_buffer('pixel_mask', pixel_mask)
        else:
            self.pixel_mask: Optional[torch.Tensor] = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`pixel_mask` overrides the constructor buffer -- pass the PER-SAMPLE mask
        [B, 1, H, W] when a batch mixes geometries, otherwise every sample is scored
        against the wrong collider."""
        m = pixel_mask if pixel_mask is not None else self.pixel_mask
        if m is None:
            d = pred - target
            return d.pow(2).mean() + self.l1_weight * d.abs().mean()
        # Boolean indexing flattens across samples; weight instead so a per-sample
        # mask broadcasts and the normaliser counts only that sample's fluid pixels.
        w = m[:, :1].to(pred.dtype)
        if w.shape[0] != pred.shape[0]:
            w = w.expand(pred.shape[0], -1, -1, -1)
        d = pred - target
        n = w.expand_as(d).sum().clamp_min(1.0)
        return ((d.pow(2) * w).sum() / n) + self.l1_weight * ((d.abs() * w).sum() / n)


# ---------------------------------------------------------------------------
# Stride probe: does the context actually encode the timestep?
# ---------------------------------------------------------------------------

class StrideProbe(nn.Module):
    """
    Classify the sampled temporal stride from the context tokens alone.

    The stride is never given to the model explicitly — it exists only in how
    far the flow moves between context frames — so this probe's accuracy is a
    direct measurement that the context encodes the timestep.  Trained jointly
    (small weight), it also applies gradient pressure against the context
    collapsing to a constant: a constant context cannot be classified.
    """

    def __init__(self, d_ctx: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_ctx),
            nn.Linear(d_ctx, d_ctx // 2),
            nn.GELU(),
            nn.Linear(d_ctx // 2, n_classes),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, K, d_ctx] → logits [B, n_classes]."""
        return self.net(context.mean(dim=1))


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
    disc_update_threshold: float = 0.3,
    self_input_prob: float = 0.0,
    rollout_horizon: int = 1,
    accum_steps: int = 1,
    micro_first: bool = True,
    micro_last: bool = True,
    stride_probe: Optional[nn.Module] = None,
    stride_label: Optional[int] = None,
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

    # ---- stride probe: recover the timestep from the context alone ----
    # frames[] is already stride-subsampled by the caller; the stride reaches
    # the encoder only through the inter-frame motion, so this CE term both
    # measures and enforces that the context encodes the timestep.
    probe_term = None
    if stride_probe is not None and stride_label is not None and stride_cls_weight > 0.0:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            logits = stride_probe(context)
        labels = torch.full((logits.shape[0],), stride_label,
                            dtype=torch.long, device=logits.device)
        probe_term = F.cross_entropy(logits.float(), labels)
        if metrics is not None:
            metrics['stride_loss'] = float(probe_term.item())
            metrics['stride_acc']  = float((logits.argmax(-1) == labels)
                                           .float().mean().item())

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
    recon_terms = []
    pred_masked = None
    x_cur = x_in
    for k in range(horizon):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            pred_k = model(x_cur, context, pixel_mask=pixel_mask)
        pred_k = pred_k.float() * pixel_mask if pixel_mask is not None else pred_k.float()
        recon_terms.append(criterion(pred_k, frames[n_context + 1 + k],
                                     pixel_mask=pixel_mask))
        if pred_masked is None:
            pred_masked = pred_k          # first step feeds the GAN, as before
        x_cur = pred_k                    # feed prediction forward (keeps grad)
    assert pred_masked is not None
    recon_rollout = torch.stack(recon_terms).mean()

    def _zero_and_restore() -> None:
        gen_optimizer.zero_grad()
        if disc_optimizer is not None:
            disc_optimizer.zero_grad()
        if discriminator is not None:
            for p in discriminator.parameters():
                p.requires_grad_(True)

    # Modules the generator backward writes grads into (probe included so its
    # grads are averaged and clipped alongside the rest on every path).
    gen_modules = [model, context_encoder] + ([stride_probe] if stride_probe is not None else [])
    gen_params  = [p for mod in gen_modules for p in mod.parameters()]

    # ---- reconstruction only ----
    if adv_weight == 0.0:
        recon_loss = recon_rollout
        gen_loss = recon_loss if probe_term is None \
            else recon_loss + stride_cls_weight * probe_term
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

    recon_loss = recon_rollout

    disc_healthy = disc_update_threshold < d_gate < 2.0   # same GLOBAL gate as above
    if disc_healthy:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            adv_logit = discriminator(pred_masked, x_in_d, context)
        adv_loss = F.binary_cross_entropy_with_logits(
            adv_logit, torch.ones_like(adv_logit)
        )
        total_loss = recon_loss + adv_weight * adv_loss
    else:
        total_loss = recon_loss
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
        hole_weight: float = 0.1,
        hole_fill_sigma: float = 15.0,
        gan_start_step: int = 10_000,
        gan_ramp_steps: int = 2_000,
        disc_update_threshold: float = 0.3,
        pixel_mask: Optional[torch.Tensor] = None,
        self_input_prob: float = 0.5,
        cosine_t_max: int = 10_000,
        accum_steps: int = 1,
        use_gan: bool = True,
    ):
        self.self_input_prob  = self_input_prob
        self.cosine_t_max     = cosine_t_max
        self.accum_steps      = max(1, int(accum_steps))
        self._micro           = 0     # position within the current accumulation window
        self.use_gan          = use_gan
        self.cfg              = cfg
        self.model            = QuadtreeHFM(cfg) if cfg.use_quadtree else HFM(cfg)
        self.context_encoder  = ContextEncoder(cfg)
        # Complete GAN toggle: with use_gan=False the discriminator is never built,
        # trained, saved, or applied — training is pure reconstruction.
        self.discriminator    = HFMDiscriminator(cfg) if use_gan else None
        self.criterion        = FluidLoss(
            l1_weight, pixel_mask=pixel_mask,
            hole_weight=hole_weight, hole_fill_sigma=hole_fill_sigma,
        )

        # Stride probe: only meaningful when there is more than one stride to
        # classify.  Joint-trained with a small weight (cfg.stride_cls_weight).
        strides = tuple(getattr(cfg, 'time_strides', (1,)) or (1,))
        self.time_strides = strides
        self.stride_probe = (
            StrideProbe(cfg.d_ctx, len(strides)) if len(strides) > 1 else None
        )
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
    ) -> float:
        """Mean reconstruction loss over a validation dataloader (no gradient update)."""
        self.model.eval()
        self.context_encoder.eval()
        device      = next(self.model.parameters()).device
        device_type = device.type
        amp         = device_type in ('cuda', 'xpu')
        nc      = self.cfg.n_context_frames
        total, count = 0.0, 0
        for batch in dataloader:
            # (frames, mesh_id) when the datamodule tags geometries, else just frames
            if isinstance(batch, (tuple, list)):
                batch, mesh_b = batch[0], batch[1]
            else:
                mesh_b = None
            pixel_mask = self._mask_ctx(pixel_mask, mesh_b, val=True)
            frames   = [batch[:, t].to(device) for t in range(batch.shape[1])]
            horizon  = max(1, min(getattr(self.cfg, 'rollout_horizon', 1),
                                  len(frames) - nc - 1))
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
                context = self.context_encoder(frames[:nc], pixel_mask=pixel_mask)
                # Same unrolled metric as training: predict horizon steps,
                # feeding predictions forward, average loss over the horizon.
                x_cur, terms = frames[nc], []
                for k in range(horizon):
                    pred = self.model(x_cur, context, pixel_mask=pixel_mask)
                    pred_m = pred.float() * pixel_mask if pixel_mask is not None else pred.float()
                    terms.append(self.criterion(pred_m, frames[nc + 1 + k],
                                                pixel_mask=pixel_mask).item())
                    x_cur = pred_m
            loss = sum(terms) / len(terms)
            if loss == loss:   # finite check (NaN != NaN)
                total += loss
                count += 1
        return total / count if count > 0 else float('nan')

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
                accum_steps=self.accum_steps,
                micro_first=micro_first,
                micro_last=micro_last,
                stride_probe=self.stride_probe,
                stride_label=s_idx,
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
        info.update(self._stride_metrics)      # stride_loss / stride_acc when present
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
            self.context_encoder.load_state_dict(ckpt['context_encoder'])
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
