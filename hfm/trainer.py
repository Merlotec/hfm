"""
Training loop for the Hierarchical Fluid Model.

Warmup phase (runs before each training step)
----------------------------------------------
Given a window of ground-truth frames [f_0, f_1, ..., f_{T-1}]:

  sys ← zeros
  for t in 0..T-2:
      pred_t, sys ← model(f_t, sys, resid=None)          # first pass: no residual
      err_t       = f_{t+1} - pred_t                      # prediction error
      pred_t2, sys ← model(f_t, sys, resid=err_t)         # refine sys with residual
  # sys now encodes the dynamics of this sequence

Training phase
--------------
  sys_frozen = sys.detach_copy()    # gradients can still flow back through warmup!
  pred_T, _ ← model(f_{T-1}, sys_frozen, resid=None, freeze_sys=True)
  loss = criterion(pred_T, f_T)
  loss.backward()   # backprops through training step AND warmup accumulation

freeze_sys=True means the system tokens are detached WITHIN the training
forward pass (they don't change further), but the tensors themselves were
produced by earlier graph nodes so gradients do flow back into the warmup.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Optional, Tuple

from .model import HFM
from .discriminator import HFMDiscriminator
from .config import HFMConfig


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FluidLoss(nn.Module):
    def __init__(self, l1_weight: float = 0.1,
                 pixel_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.l1_weight = l1_weight
        if pixel_mask is not None:
            self.register_buffer('pixel_mask', pixel_mask)
        else:
            self.pixel_mask: Optional[torch.Tensor] = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pixel_mask is not None:
            valid = self.pixel_mask.expand_as(target)
            p = pred[valid]
            t = target[valid]
        else:
            p = pred.reshape(-1)
            t = target.reshape(-1)
        mse = (p - t).pow(2).mean()
        l1  = (p - t).abs().mean()
        return mse + self.l1_weight * l1


# ---------------------------------------------------------------------------
# Warmup: build system embeddings from a sequence of frames
# ---------------------------------------------------------------------------

def warmup_system(
    model: HFM,
    frames: List[torch.Tensor],
    n_warmup: int,
    pixel_mask: Optional[torch.Tensor] = None,
    criterion: Optional[nn.Module] = None,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Run n_warmup steps of (forward → compute residual → forward-with-residual)
    to build up meaningful system embeddings.

    When criterion is provided, accumulates a prediction loss at each warmup
    step (averaged over steps).  This loss, backpropped before sys is detached,
    is the only gradient signal that reaches SystemLevelBlock and sys_pos —
    it trains sys to encode information that actually improves predictions.

    Returns (sys_emb, warmup_loss).  warmup_loss is None when criterion is None.
    """
    B = frames[0].shape[0]
    device = frames[0].device
    sys = model.init_sys_emb(B, device)

    T = min(n_warmup, len(frames) - 1)
    warmup_loss: Optional[torch.Tensor] = None

    for t in range(T):
        x_t    = frames[t]
        x_next = frames[t + 1]

        # First pass: predict next frame, update sys (no residual yet)
        pred, sys, _ = model(x_t, sys_emb=sys, resid=None, freeze_sys=False,
                             pixel_mask=pixel_mask)
        if pixel_mask is not None:
            pred = pred * pixel_mask

        if criterion is not None:
            step_loss = criterion(pred, x_next)
            warmup_loss = step_loss if warmup_loss is None else warmup_loss + step_loss

        err = (x_next - pred).detach()

        # Second pass: refine sys using the residual signal
        _, sys, _ = model(x_t, sys_emb=sys, resid=err, freeze_sys=False,
                          pixel_mask=pixel_mask)

    if warmup_loss is not None and T > 0:
        warmup_loss = warmup_loss / T   # average over warmup steps

    return sys, warmup_loss


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    model: HFM,
    frames: List[torch.Tensor],          # length >= n_warmup + 2
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    n_warmup: Optional[int] = None,
    clip_grad: float = 1.0,
    pixel_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Run one full training step:
      1. warmup to build sys
      2. freeze sys and predict the next frame
      3. compute loss and backprop
    Returns the scalar loss value.
    """
    nw: int = n_warmup if n_warmup is not None else model.cfg.n_warmup_frames

    optimizer.zero_grad()

    # Warmup uses frames[0 .. nw-1] to build sys
    warmup_frames = frames[:nw + 1]
    sys, _ = warmup_system(model, warmup_frames, nw)

    # Training pass: freeze sys within this forward call, predict frame nw+1
    x_in = frames[nw]
    x_target = frames[nw + 1]
    pred, _, _ = model(x_in, sys_emb=sys, resid=None, freeze_sys=True)
    if pixel_mask is not None:
        pred = pred * pixel_mask

    loss = criterion(pred, x_target)
    loss.backward()

    if clip_grad > 0:
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class HFMTrainer:
    """
    High-level training wrapper.

    Usage
    -----
    trainer = HFMTrainer(cfg)
    for epoch in range(n_epochs):
        for frame_batch in dataloader:   # frame_batch: [T, B, C, H, W]
            frames = [frame_batch[t] for t in range(T)]
            loss = trainer.step(frames)
    """

    def __init__(self, cfg: HFMConfig, lr: float = 1e-4,
                 weight_decay: float = 1e-5, l1_weight: float = 0.1):
        self.cfg = cfg
        self.model = HFM(cfg)
        self.criterion = FluidLoss(l1_weight)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10_000
        )

    def to(self, device: torch.device) -> "HFMTrainer":
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        return self

    def step(self, frames: List[torch.Tensor]) -> float:
        """frames: list of T tensors each [B, C, H, W]."""
        self.model.train()
        loss = train_step(
            self.model, frames, self.optimizer, self.criterion,
            self.cfg.n_warmup_frames,
        )
        self.scheduler.step()
        return loss

    @torch.no_grad()
    def predict(
        self,
        warmup_frames: List[torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warm up on warmup_frames, then predict the next frame from x.
        Returns predicted [B, C, H, W].
        """
        self.model.eval()
        sys, _ = warmup_system(self.model, warmup_frames, len(warmup_frames) - 1)
        pred, _, _ = self.model(x, sys_emb=sys, freeze_sys=True)
        return pred

    def save(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'cfg': self.cfg,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])


# ---------------------------------------------------------------------------
# GAN training step
# ---------------------------------------------------------------------------

def train_step_gan(
    model: HFM,
    discriminator: HFMDiscriminator,
    frames: List[torch.Tensor],
    gen_optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer,
    criterion: nn.Module,
    n_warmup: Optional[int] = None,
    adv_weight: float = 0.1,
    warmup_loss_weight: float = 0.5,
    clip_grad: float = 1.0,
    pixel_mask: Optional[torch.Tensor] = None,
    gan_active: bool = True,
    disc_update_threshold: float = 0.3,
) -> Tuple[float, float]:
    """
    One training step, with optional GAN.

    Returns (recon_loss, disc_loss).  disc_loss is 0.0 when gan_active=False.

    When gan_active=True:
      - Discriminator step: real vs fake with feat/sys as conditioning context;
        update is skipped when d_loss <= disc_update_threshold (prevents the
        discriminator from over-specialising once it already separates well).
      - Generator step: non-saturating adversarial loss + reconstruction.
        Feature and system tokens flow gradients freely — the discriminator
        imposes no direct value targets, only real/fake classification signal.

    When gan_active=False:
      - Pure reconstruction step; discriminator is not called.

    warmup_loss_weight: weight applied to the warmup prediction loss before
      backpropping.  This is the only gradient path into SystemLevelBlock and
      sys_pos, which are unreachable from the training loss (freeze_sys=True
      detaches sys inside model.forward).  Set to 0.0 to disable.
    """
    nw: int = n_warmup if n_warmup is not None else model.cfg.n_warmup_frames

    gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()

    # Warmup with truncated BPTT depth=1: backward each step's prediction loss
    # immediately and detach sys before the next step.  This bounds peak graph
    # memory to ONE forward pass (vs 2*nw with a single accumulated backward)
    # while still routing gradients into SystemLevelBlock and sys_pos — the only
    # path that can train them (training forward uses freeze_sys=True).
    # Each step_loss.backward() accumulates into the gradient buffer; the single
    # gen_optimizer.step() at the end applies warmup + training gradients together.
    B      = frames[0].shape[0]
    device = frames[0].device
    sys    = model.init_sys_emb(B, device)

    for t in range(nw):
        x_t    = frames[t]
        x_next = frames[t + 1]

        pred_t, sys, _ = model(x_t, sys_emb=sys, resid=None, freeze_sys=False,
                               pixel_mask=pixel_mask)
        if pixel_mask is not None:
            pred_t = pred_t * pixel_mask

        if warmup_loss_weight > 0.0:
            step_loss = criterion(pred_t, x_next)
            if torch.isfinite(step_loss):
                (warmup_loss_weight * step_loss / nw).backward()

        sys = [s.detach() for s in sys]
        err = (x_next - pred_t).detach()

        _, sys, _ = model(x_t, sys_emb=sys, resid=err, freeze_sys=False,
                          pixel_mask=pixel_mask)
        sys = [s.detach() for s in sys]

    x_in     = frames[nw]
    x_target = frames[nw + 1]

    pred, _, feat = model(x_in, sys_emb=sys, resid=None, freeze_sys=True,
                          pixel_mask=pixel_mask)
    if pixel_mask is not None:
        pred = pred * pixel_mask

    def _zero_and_restore() -> None:
        gen_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        for p in discriminator.parameters():
            p.requires_grad_(True)

    if not gan_active:
        recon_loss = criterion(pred, x_target)
        if not torch.isfinite(recon_loss):
            _zero_and_restore()
            return float('nan'), 0.0
        recon_loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        gen_optimizer.step()
        return recon_loss.item(), 0.0

    # Detached context: same conditioning for real and fake discriminator calls.
    # Detaching here keeps the D-step backward isolated from the generator graph.
    feat_ctx = [f.detach() for f in feat]
    sys_ctx  = [s.detach() for s in sys]

    # === Discriminator update ===
    for p in discriminator.parameters():
        p.requires_grad_(True)

    real_logit = discriminator(x_target,      feat_ctx, sys_ctx)
    fake_logit = discriminator(pred.detach(), feat_ctx, sys_ctx)

    real_labels = torch.full_like(real_logit, 0.9)
    fake_labels = torch.zeros_like(fake_logit)

    d_loss = (F.binary_cross_entropy_with_logits(real_logit, real_labels) +
              F.binary_cross_entropy_with_logits(fake_logit, fake_labels))
    d_loss_val = d_loss.item()

    if not math.isfinite(d_loss_val):
        _zero_and_restore()
        return float('nan'), float('nan')

    # Gate: skip the discriminator update when it is already separating well,
    # to prevent it from memorising training samples and stalling the generator.
    if d_loss_val > disc_update_threshold:
        d_loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
        disc_optimizer.step()
    disc_optimizer.zero_grad()

    # === Generator update ===
    for p in discriminator.parameters():
        p.requires_grad_(False)

    recon_loss = criterion(pred, x_target)

    adv_logit = discriminator(pred, feat, sys)
    adv_loss  = F.binary_cross_entropy_with_logits(
        adv_logit, torch.ones_like(adv_logit)
    )

    total_loss = recon_loss + adv_weight * adv_loss
    if not torch.isfinite(total_loss):
        _zero_and_restore()
        return float('nan'), d_loss_val

    total_loss.backward()

    if clip_grad > 0:
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    gen_optimizer.step()

    for p in discriminator.parameters():
        p.requires_grad_(True)

    return recon_loss.item(), d_loss_val


# ---------------------------------------------------------------------------
# GAN Trainer class
# ---------------------------------------------------------------------------

class GANTrainer:
    """
    Two-stage training wrapper that pairs HFM with an HFMDiscriminator.

    Stage 1 — reconstruction + warmup curriculum (no GAN)
    ------------------------------------------------------
    n_warmup ramps linearly from 1 → cfg.n_warmup_frames over
    warmup_ramp_steps steps.  Starting at n_warmup=1 means early training
    resembles pure supervised prediction (sys ≈ zeros), and the warmup
    mechanism is introduced gradually as the model stabilises.

    Stage 2 — GAN active  (begins at gan_start_step)
    -------------------------------------------------
    adv_weight ramps from 0 → cfg.disc_adv_weight over gan_ramp_steps steps.
    The discriminator update is skipped whenever d_loss <= disc_update_threshold
    to prevent it from over-specialising and stalling the generator gradient.

    Usage
    -----
    trainer = GANTrainer(cfg)
    trainer.to(device)
    for frame_batch in dataloader:          # [T, B, C, H, W]
        frames = [frame_batch[t] for t in range(T)]
        recon_loss, disc_loss = trainer.step(frames)
        print(trainer.training_info())
    """

    def __init__(
        self,
        cfg: HFMConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        l1_weight: float = 0.1,
        warmup_ramp_steps: int = 5_000,
        gan_start_step: int = 10_000,
        gan_ramp_steps: int = 1_000,
        disc_update_threshold: float = 0.3,
        pixel_mask: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.model = HFM(cfg)
        self.discriminator = HFMDiscriminator(cfg)
        self.criterion = FluidLoss(l1_weight, pixel_mask=pixel_mask)

        self.gen_optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=cfg.disc_lr, betas=(0.5, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.gen_optimizer, T_max=10_000
        )

        self.warmup_ramp_steps    = warmup_ramp_steps
        self.gan_start_step       = gan_start_step
        self.gan_ramp_steps       = gan_ramp_steps
        self.disc_update_threshold = disc_update_threshold
        self.global_step          = 0

    # ---- curriculum helpers ----

    def _current_n_warmup(self) -> int:
        max_nw = self.cfg.n_warmup_frames
        if self.warmup_ramp_steps <= 0 or self.global_step >= self.warmup_ramp_steps:
            return max_nw
        frac = self.global_step / self.warmup_ramp_steps
        return max(1, round(1 + (max_nw - 1) * frac))

    def _current_adv_weight(self) -> float:
        if self.global_step < self.gan_start_step:
            return 0.0
        steps_in = self.global_step - self.gan_start_step
        ramp = min(1.0, steps_in / max(1, self.gan_ramp_steps))
        return self.cfg.disc_adv_weight * ramp

    # ---- public API ----

    def to(self, device: torch.device) -> "GANTrainer":
        self.model         = self.model.to(device)
        self.discriminator = self.discriminator.to(device)
        self.criterion     = self.criterion.to(device)
        return self

    def step(
        self,
        frames: List[torch.Tensor],
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """frames: list of T tensors each [B, C, H, W].  Returns (recon_loss, disc_loss)."""
        self.model.train()
        self.discriminator.train()

        adv_weight = self._current_adv_weight()

        recon_loss, disc_loss = train_step_gan(
            self.model, self.discriminator, frames,
            self.gen_optimizer, self.disc_optimizer,
            self.criterion,
            n_warmup=self._current_n_warmup(),
            adv_weight=adv_weight,
            pixel_mask=pixel_mask,
            gan_active=(adv_weight > 0.0),
            disc_update_threshold=self.disc_update_threshold,
        )

        self.scheduler.step()
        self.global_step += 1
        return recon_loss, disc_loss

    def training_info(self) -> dict:
        """Current curriculum state — log this alongside losses."""
        adv_weight = self._current_adv_weight()
        return {
            'global_step': self.global_step,
            'n_warmup':    self._current_n_warmup(),
            'adv_weight':  adv_weight,
            'gan_active':  adv_weight > 0.0,
        }

    @torch.no_grad()
    def predict(
        self,
        warmup_frames: List[torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        sys, _ = warmup_system(self.model, warmup_frames, len(warmup_frames) - 1)
        pred, _, _ = self.model(x, sys_emb=sys, freeze_sys=True)
        return pred

    def save(self, path: str):
        torch.save({
            'model':         self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'disc_optimizer':self.disc_optimizer.state_dict(),
            'scheduler':     self.scheduler.state_dict(),
            'cfg':           self.cfg,
            'global_step':   self.global_step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.discriminator.load_state_dict(ckpt['discriminator'])
        self.gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
        self.disc_optimizer.load_state_dict(ckpt['disc_optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt.get('global_step', 0)
