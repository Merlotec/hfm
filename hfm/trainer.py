"""
Training loop for the Hierarchical Fluid Model.

Each step:
  1. ContextEncoder encodes n_context_frames ground-truth frames → context [B, K, d_ctx]
  2. HFM predicts the next frame from frames[n_context_frames] conditioned on context
  3. Reconstruction loss (+ adversarial loss once GAN is active)

The context encoder and main model are optimised jointly — gradients from the
prediction loss flow freely through both.  No warmup BPTT or sys/resid logic.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Optional, Tuple

from .model import HFM
from .context_encoder import ContextEncoder
from .discriminator import HFMDiscriminator
from .config import HFMConfig


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FluidLoss(nn.Module):
    """
    MSE + L1 reconstruction loss with optional hole-filling.

    Valid pixels: loss against true target (full weight).
    Hole pixels:  loss against Gaussian-weighted average of nearby valid pixels
                  (weight = hole_weight).  pred must be unmasked so gradient
                  flows back through hole regions.
    """

    def __init__(
        self,
        l1_weight: float = 0.1,
        pixel_mask: Optional[torch.Tensor] = None,
        hole_weight: float = 0.1,
        hole_fill_sigma: float = 15.0,
    ):
        super().__init__()
        self.l1_weight  = l1_weight
        self.hole_weight = hole_weight
        if pixel_mask is not None:
            self.register_buffer('pixel_mask', pixel_mask)
            self.register_buffer('fill_kernel', self._gauss1d(hole_fill_sigma))
        else:
            self.pixel_mask:  Optional[torch.Tensor] = None
            self.fill_kernel: Optional[torch.Tensor] = None

    @staticmethod
    def _gauss1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
        r = int(sigma * truncate + 0.5)
        x = torch.arange(-r, r + 1, dtype=torch.float)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        return k / k.sum()   # [K]

    def _fill_holes(self, target: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian fill: hole pixels ← weighted avg of nearby valid pixels."""
        pm = self.pixel_mask
        fk = self.fill_kernel
        assert pm is not None and fk is not None
        B, C, H, W = target.shape
        mask  = pm.float()                                  # [1, 1, H, W]
        valid = target * mask

        pad = fk.shape[0] // 2
        kh  = fk.reshape(1, 1, -1, 1)                      # vertical kernel
        kw  = fk.reshape(1, 1, 1, -1)                      # horizontal kernel

        bc = B * C
        v  = valid.reshape(bc, 1, H, W)
        m  = mask.expand(B, C, H, W).reshape(bc, 1, H, W)

        # Separable blur of values and mask weights
        vb = F.conv2d(F.conv2d(v, kh, padding=(pad, 0)), kw, padding=(0, pad))
        mb = F.conv2d(F.conv2d(m, kh, padding=(pad, 0)), kw, padding=(0, pad))

        filled = (vb / mb.clamp(min=1e-6)).reshape(B, C, H, W)
        return target * mask + filled * (1.0 - mask)       # valid pixels unchanged

    def _pixel_loss(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        d = p - t
        return d.pow(2).mean() + self.l1_weight * d.abs().mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pixel_mask is None:
            return self._pixel_loss(pred.reshape(-1), target.reshape(-1))

        mask      = self.pixel_mask.expand_as(pred).bool()
        hole_mask = ~mask

        valid_loss = self._pixel_loss(pred[mask], target[mask])

        if hole_mask.any() and self.hole_weight > 0.0:
            filled = self._fill_holes(target.float())
            hole_loss = self._pixel_loss(pred[hole_mask], filled.expand_as(pred)[hole_mask])
            return valid_loss + self.hole_weight * hole_loss

        return valid_loss


# ---------------------------------------------------------------------------
# GAN training step
# ---------------------------------------------------------------------------

def train_step_gan(
    model: HFM,
    context_encoder: ContextEncoder,
    discriminator: HFMDiscriminator,
    frames: List[torch.Tensor],
    gen_optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer,
    criterion: nn.Module,
    n_context: int,
    adv_weight: float = 0.0,
    clip_grad: float = 1.0,
    pixel_mask: Optional[torch.Tensor] = None,
    disc_update_threshold: float = 0.3,
) -> Tuple[float, float]:
    """
    One training step.

    frames[0 .. n_context-1]  → ContextEncoder → context
    frames[n_context]         → HFM(context)   → pred
    frames[n_context + 1]     → target

    Returns (recon_loss, disc_loss).  disc_loss is 0.0 when adv_weight == 0.
    """
    gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()

    device_type = frames[0].device.type
    amp = device_type == 'cuda'

    x_in     = frames[n_context]
    x_target = frames[n_context + 1]

    # ---- encode context (gradients flow through context_encoder) ----
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        context = context_encoder(frames[:n_context], pixel_mask=pixel_mask)

    # ---- predict ----
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        pred = model(x_in, context, pixel_mask=pixel_mask)
    # pred_disc: holes zeroed for discriminator consistency with real frames.
    # pred (unmasked) goes to criterion so gradient flows through hole regions.
    pred_disc = pred.float() * pixel_mask if pixel_mask is not None else pred.float()

    def _zero_and_restore() -> None:
        gen_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        for p in discriminator.parameters():
            p.requires_grad_(True)

    # ---- reconstruction only ----
    if adv_weight == 0.0:
        recon_loss = criterion(pred, x_target)
        if not torch.isfinite(recon_loss):
            _zero_and_restore()
            return float('nan'), 0.0
        recon_loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(context_encoder.parameters()), clip_grad
            )
        gen_optimizer.step()
        return recon_loss.item(), 0.0

    # ---- discriminator update ----
    ctx_detach = context.detach()

    for p in discriminator.parameters():
        p.requires_grad_(True)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        real_logit = discriminator(x_target,            x_in, ctx_detach)
        fake_logit = discriminator(pred_disc.detach(),  x_in, ctx_detach)

    real_labels = torch.full_like(real_logit, 0.9)
    d_loss = (
        F.binary_cross_entropy_with_logits(real_logit, real_labels) +
        F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
    )
    d_loss_val = d_loss.item()

    if not math.isfinite(d_loss_val):
        _zero_and_restore()
        return float('nan'), float('nan')

    if disc_update_threshold < d_loss_val < 2.0:
        d_loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
        disc_optimizer.step()
    disc_optimizer.zero_grad()

    # ---- generator update ----
    for p in discriminator.parameters():
        p.requires_grad_(False)

    recon_loss = criterion(pred, x_target)

    disc_healthy = disc_update_threshold < d_loss_val < 2.0
    if disc_healthy:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            adv_logit = discriminator(pred_disc, x_in, context)
        adv_loss = F.binary_cross_entropy_with_logits(
            adv_logit, torch.ones_like(adv_logit)
        )
        total_loss = recon_loss + adv_weight * adv_loss
    else:
        total_loss = recon_loss
    if not torch.isfinite(total_loss):
        _zero_and_restore()
        return float('nan'), d_loss_val

    total_loss.backward()
    if clip_grad > 0:
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(context_encoder.parameters()), clip_grad
        )
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
    ):
        self.cfg              = cfg
        self.model            = HFM(cfg)
        self.context_encoder  = ContextEncoder(cfg)
        self.discriminator    = HFMDiscriminator(cfg)
        self.criterion        = FluidLoss(
            l1_weight, pixel_mask=pixel_mask,
            hole_weight=hole_weight, hole_fill_sigma=hole_fill_sigma,
        )

        gen_params = list(self.model.parameters()) + list(self.context_encoder.parameters())
        self.gen_optimizer = optim.AdamW(gen_params, lr=lr, weight_decay=weight_decay)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=cfg.disc_lr, betas=(0.5, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.gen_optimizer, T_max=10_000
        )

        self.gan_start_step        = gan_start_step
        self.gan_ramp_steps        = gan_ramp_steps
        self.disc_update_threshold = disc_update_threshold
        self.global_step           = 0

    def _current_adv_weight(self) -> float:
        if self.global_step < self.gan_start_step:
            return 0.0
        steps_in = self.global_step - self.gan_start_step
        ramp = min(1.0, steps_in / max(1, self.gan_ramp_steps))
        return self.cfg.disc_adv_weight * ramp

    def to(self, device: torch.device) -> "GANTrainer":
        self.model           = self.model.to(device)
        self.context_encoder = self.context_encoder.to(device)
        self.discriminator   = self.discriminator.to(device)
        self.criterion       = self.criterion.to(device)
        return self

    def step(
        self,
        frames: List[torch.Tensor],
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """frames: list of (n_context_frames + 2) tensors [B, C, H, W]."""
        self.model.train()
        self.context_encoder.train()
        self.discriminator.train()

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
        )

        self.scheduler.step()
        self.global_step += 1
        return recon_loss, disc_loss

    def training_info(self) -> dict:
        return {
            'global_step': self.global_step,
            'adv_weight':  self._current_adv_weight(),
            'gan_active':  self._current_adv_weight() > 0.0,
        }

    @torch.no_grad()
    def predict(
        self,
        context_frames: List[torch.Tensor],
        x: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.model.eval()
        self.context_encoder.eval()
        context = self.context_encoder(context_frames, pixel_mask=pixel_mask)
        pred = self.model(x, context, pixel_mask=pixel_mask)
        if pixel_mask is not None:
            pred = pred * pixel_mask
        return pred

    def save(self, path: str):
        torch.save({
            'model':           self.model.state_dict(),
            'context_encoder': self.context_encoder.state_dict(),
            'discriminator':   self.discriminator.state_dict(),
            'gen_optimizer':   self.gen_optimizer.state_dict(),
            'disc_optimizer':  self.disc_optimizer.state_dict(),
            'scheduler':       self.scheduler.state_dict(),
            'cfg':             self.cfg,
            'global_step':     self.global_step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        if unexpected:
            print(f'  [warn] model unexpected keys: {unexpected}')
        if missing:
            print(f'  New/missing model keys (random init): {missing}')

        if 'context_encoder' in ckpt:
            self.context_encoder.load_state_dict(ckpt['context_encoder'])
        else:
            print('  [warn] no context_encoder in checkpoint; starting fresh')

        disc_state = ckpt.get('discriminator', {})
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

        self.gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
        if disc_ok and 'disc_optimizer' in ckpt:
            try:
                self.disc_optimizer.load_state_dict(ckpt['disc_optimizer'])
            except Exception:
                print('  [warn] disc_optimizer reinitialised')
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt.get('global_step', 0)
