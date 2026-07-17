"""
PyTorch Lightning module and data module for HFM GAN training.

Supports single-GPU and multi-GPU DDP without code changes — just pass
--devices N to the training script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader

from .config import HFMConfig

# HFMConfig is stored as a Python object in checkpoints; allowlist it so
# Lightning's torch.load (weights_only=True default in PyTorch 2.6) doesn't fail.
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([HFMConfig])
from .context_encoder import ContextEncoder
from .data import FVMDataModule
from .discriminator import HFMDiscriminator
from .model import HFM
from .trainer import FluidLoss


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class HFMLightningModule(L.LightningModule):
    """
    GAN training wrapper: HFM (generator) + ContextEncoder + HFMDiscriminator.

    automatic_optimization=False — manual GAN update ordering.

    GAN curriculum
    --------------
    Steps 0 → gan_start_step:  reconstruction only
    Steps ≥ gan_start_step:    adv_weight ramps 0 → cfg.disc_adv_weight
    """

    def __init__(
        self,
        cfg: HFMConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        l1_weight: float = 0.1,
        gan_start_step: int = 10_000,
        gan_ramp_steps: int = 2_000,
        disc_update_threshold: float = 0.3,
        cosine_t_max: int = 10_000,
        pixel_mask: Optional[torch.Tensor] = None,
        self_input_prob: float = 0.5,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['cfg', 'pixel_mask'])

        self.cfg                   = cfg
        self.gan_start_step        = gan_start_step
        self.gan_ramp_steps        = gan_ramp_steps
        self.disc_update_threshold = disc_update_threshold

        self.model           = HFM(cfg)
        self.context_encoder = ContextEncoder(cfg)
        self.discriminator   = HFMDiscriminator(cfg)
        self.criterion       = FluidLoss(l1_weight, pixel_mask=pixel_mask)

        if pixel_mask is not None:
            self.register_buffer('pixel_mask', pixel_mask)
        else:
            self.pixel_mask: Optional[torch.Tensor] = None

        self._step_offset = 0   # set by load_from_pt to preserve GAN curriculum

    # ------------------------------------------------------------------

    def load_from_pt(self, path: str) -> int:
        """
        Load weights from a GANTrainer .pt checkpoint.

        Restores model, context_encoder, and discriminator weights.
        Returns the saved global_step so the caller can set _step_offset.
        Optimizer/scheduler state is NOT restored (fresh start).
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        if unexpected:
            print(f'  [warn] model unexpected keys: {unexpected}')
        if missing:
            print(f'  New/missing model keys (random init): {missing}')
        if 'context_encoder' in ckpt:
            self.context_encoder.load_state_dict(ckpt['context_encoder'])
        else:
            print('  [warn] no context_encoder in checkpoint; keeping random weights')
        if 'discriminator' in ckpt:
            try:
                self.discriminator.load_state_dict(ckpt['discriminator'], strict=False)
            except RuntimeError as e:
                print(f'  [warn] discriminator not restored: {e}')
        saved_step = ckpt.get('global_step', 0)
        print(f'  Loaded .pt checkpoint (global_step={saved_step})')
        return saved_step

    def _adv_weight(self) -> float:
        s = self.global_step + self._step_offset
        if s < self.gan_start_step:
            return 0.0
        ramp = min(1.0, (s - self.gan_start_step) / max(1, self.gan_ramp_steps))
        return self.cfg.disc_adv_weight * ramp

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        gen_opt, disc_opt = self.optimizers()  # type: ignore[misc]
        scheduler = self.lr_schedulers()

        frames = [batch[:, t] for t in range(batch.shape[1])]
        n        = self.cfg.n_context_frames
        x_in     = frames[n]
        x_target = frames[n + 1]
        mask     = self.pixel_mask
        adv_w    = self._adv_weight()

        context = self.context_encoder(frames[:n], pixel_mask=mask)

        # scheduled sampling (exposure-bias fix): with probability self_input_prob,
        # feed the model its own no-grad prediction of frame n instead of the GT —
        # the target stays GT frame n+1 (see trainer.train_step_gan for rationale).
        if float(torch.rand(())) < self.hparams['self_input_prob']:
            with torch.no_grad():
                x_in = self.model(frames[n - 1], context, pixel_mask=mask).float()
            if mask is not None:
                x_in = x_in * mask

        pred    = self.model(x_in, context, pixel_mask=mask)

        # ---- reconstruction only (pre-GAN) ----
        if adv_w == 0.0:
            recon = self.criterion(pred, x_target)
            gen_opt.zero_grad()
            self.manual_backward(recon)
            self.clip_gradients(gen_opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')  # type: ignore[arg-type]
            gen_opt.step()
            scheduler.step()  # type: ignore[union-attr]
            self.log('recon', recon, prog_bar=True, sync_dist=True)
            return

        # pred_disc: holes zeroed for discriminator; pred stays unmasked for criterion
        # so gradient flows back through hole regions via the hole-filling loss.
        pred_disc = pred.float() * mask if mask is not None else pred.float()

        # ---- discriminator update ----
        # Mask BOTH disc inputs: raw target frames carry the renderer's hole fill
        # (up to -4.4 sigma after normalisation) while pred_disc has holes at 0, so
        # an unmasked x_target lets the disc win from a single hole pixel (see
        # trainer.train_step_gan for the full explanation).
        ctx_d      = context.detach()
        x_target_d = x_target * mask if mask is not None else x_target
        x_in_d     = x_in     * mask if mask is not None else x_in
        real_logit = self.discriminator(x_target_d,         x_in_d, ctx_d)
        fake_logit = self.discriminator(pred_disc.detach(), x_in_d, ctx_d)
        d_loss = (
            F.binary_cross_entropy_with_logits(real_logit, torch.full_like(real_logit, 0.9)) +
            F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
        )
        
        # In DDP, sync the loss scalar across ranks so all GPUs make the exact same decision 
        # on whether to run manual_backward. Differing decisions lead to NCCL timeouts.
        d_loss_val = d_loss.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(d_loss_val, op=torch.distributed.ReduceOp.AVG)
            
        disc_healthy = self.disc_update_threshold < d_loss_val.item() < 2.0

        disc_opt.zero_grad()
        if disc_healthy:
            self.manual_backward(d_loss)
            self.clip_gradients(disc_opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')  # type: ignore[arg-type]
            disc_opt.step()
        disc_opt.zero_grad()

        # ---- generator update ----
        for p in self.discriminator.parameters():
            p.requires_grad_(False)

        recon = self.criterion(pred, x_target)
        # Only apply adversarial loss when the discriminator is in the healthy range
        if disc_healthy:
            adv_logit = self.discriminator(pred_disc, x_in_d, context)
            adv_loss  = F.binary_cross_entropy_with_logits(adv_logit, torch.ones_like(adv_logit))
            g_loss    = recon + adv_w * adv_loss
        else:
            g_loss = recon

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        self.clip_gradients(gen_opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
        gen_opt.step()
        scheduler.step()  # type: ignore[union-attr]

        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        self.log_dict(
            {'recon': recon, 'disc': d_loss, 'adv_w': adv_w},
            prog_bar=True, sync_dist=True,
        )

    def configure_optimizers(self):  # type: ignore[override]
        gen_params = list(self.model.parameters()) + list(self.context_encoder.parameters())
        gen_opt  = optim.AdamW(gen_params,
                               lr=self.hparams['lr'],
                               weight_decay=self.hparams['weight_decay'])
        disc_opt = optim.Adam(self.discriminator.parameters(),
                              lr=self.cfg.disc_lr, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            gen_opt, T_max=self.hparams['cosine_t_max']
        )
        return (
            [gen_opt, disc_opt],
            [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}],
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Inject HFM-native keys so infer.py can read this .ckpt file directly
        checkpoint['model']           = self.model.state_dict()
        checkpoint['context_encoder'] = self.context_encoder.state_dict()
        checkpoint['discriminator']   = self.discriminator.state_dict()
        checkpoint['cfg']             = self.cfg
        checkpoint['global_step']     = self.global_step

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Fill keys present in the current model but absent from the checkpoint
        # (e.g. new layers added after checkpoint was saved) with random-init values.
        # Drop keys that no longer exist (e.g. removed buffers like criterion.fill_kernel).
        current = self.state_dict()
        ckpt_sd = checkpoint['state_dict']
        for k, v in current.items():
            if k not in ckpt_sd:
                ckpt_sd[k] = v
        for k in list(ckpt_sd.keys()):
            if k not in current:
                del ckpt_sd[k]


# ---------------------------------------------------------------------------
# Lightning data module
# ---------------------------------------------------------------------------

class FVMLightningDataModule(L.LightningDataModule):
    """
    Wraps FVMDataModule for use with Lightning's multi-GPU training.

    prepare_data(): runs on rank 0 only — ensures renderer + stats caches exist.
    setup():        runs on every rank — builds per-process dataset.
    """

    def __init__(
        self,
        data_dir:    str,
        seq_len:     int,
        resolution:  tuple[int, int] = (256, 256),
        batch_size:  int = 4,
        num_workers: int = 4,
        first_frame: int = 20,
    ):
        super().__init__()
        self._data_dir    = data_dir
        self._seq_len     = seq_len
        self._resolution  = resolution
        self._batch_size  = batch_size
        self._num_workers = num_workers
        self._first_frame = first_frame
        self._inner: Optional[FVMDataModule] = None

    def _make_dm(self) -> FVMDataModule:
        return FVMDataModule(
            data_dir    = Path(self._data_dir),
            seq_len     = self._seq_len,
            resolution  = self._resolution,
            batch_size  = self._batch_size,
            num_workers = self._num_workers,
            first_frame = self._first_frame,
        )

    def prepare_data(self) -> None:
        # Rank-0 only: build renderer cache and normalisation stats on disk
        self._make_dm().setup()

    def setup(self, stage: Optional[str] = None) -> None:
        self._inner = self._make_dm()
        self._inner.setup()

    def train_dataloader(self) -> DataLoader:
        assert self._inner is not None, 'setup() not called'
        return self._inner.train_dataloader()
