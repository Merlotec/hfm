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

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional

from .model import HFM
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
    frames: List[torch.Tensor],      # list of [B, C, H, W], already masked
    n_warmup: int,
    pixel_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Run n_warmup steps of (forward → compute residual → forward-with-residual)
    to build up meaningful system embeddings.

    Returns the updated sys_emb list.  The computation graph is kept alive
    so that loss.backward() later can propagate through these steps.
    """
    B = frames[0].shape[0]
    device = frames[0].device
    sys = model.init_sys_emb(B, device)

    T = min(n_warmup, len(frames) - 1)
    for t in range(T):
        x_t = frames[t]
        x_next = frames[t + 1]

        # First pass: encode current frame, update sys (no residual yet)
        pred, sys = model(x_t, sys_emb=sys, resid=None, freeze_sys=False,
                          pixel_mask=pixel_mask)
        if pixel_mask is not None:
            pred = pred * pixel_mask

        # Compute prediction error
        err = (x_next - pred).detach()  # detach error from pred graph — we only
                                         # want the residual to inform sys, not
                                         # backprop through the error itself here

        # Second pass: refine sys using the residual signal
        _, sys = model(x_t, sys_emb=sys, resid=err, freeze_sys=False,
                       pixel_mask=pixel_mask)

    return sys


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
    sys = warmup_system(model, warmup_frames, nw)

    # Training pass: freeze sys within this forward call, predict frame nw+1
    x_in = frames[nw]
    x_target = frames[nw + 1]
    pred, _ = model(x_in, sys_emb=sys, resid=None, freeze_sys=True)
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
        sys = warmup_system(self.model, warmup_frames, len(warmup_frames) - 1)
        pred, _ = self.model(x, sys_emb=sys, freeze_sys=True)
        return pred

    def save(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'cfg': self.cfg,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
