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
from .quadtree import QuadtreeHFM
from .trainer import FluidLoss, ContextProbe, probe_losses


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
        mesh_masks: Optional[torch.Tensor] = None,
        self_input_prob: float = 0.5,
        use_gan: bool = True,
        probe_bc_dim: int = 8,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['cfg', 'pixel_mask', 'mesh_masks'])

        self.cfg                   = cfg
        self.gan_start_step        = gan_start_step
        self.gan_ramp_steps        = gan_ramp_steps
        self.disc_update_threshold = disc_update_threshold
        self.use_gan               = use_gan

        self.model           = QuadtreeHFM(cfg) if cfg.use_quadtree else HFM(cfg)
        self.context_encoder = ContextEncoder(cfg)
        # Complete GAN toggle: no discriminator is built or applied when use_gan=False.
        self.discriminator   = HFMDiscriminator(cfg) if use_gan else None
        self.criterion       = FluidLoss(l1_weight, pixel_mask=pixel_mask)

        # Multi-timestep training: each step samples a stride s from time_strides and
        # trains on every s-th frame, so the timestep reaches the model only through
        # the context (spacing of its input frames).  The probe recovers stride,
        # geometry, and BC summary from the context alone — evidence + gradient
        # pressure that the context carries them (see trainer.ContextProbe).
        self.time_strides = tuple(getattr(cfg, 'time_strides', (1,)) or (1,))
        n_meshes = 0 if mesh_masks is None else int(mesh_masks.shape[0])
        self.stride_probe = (
            ContextProbe(cfg.d_ctx, len(self.time_strides),
                         n_meshes=n_meshes, bc_dim=probe_bc_dim)
            if len(self.time_strides) > 1 else None
        )

        if pixel_mask is not None:
            self.register_buffer('pixel_mask', pixel_mask)
        else:
            self.pixel_mask: Optional[torch.Tensor] = None
        # Per-geometry masks, gathered per sample in training_step.  A batch mixes
        # geometries (ConcatDataset + shuffle), so the single shared pixel_mask above
        # is wrong for every mesh but the first -- masks differ by ~13% of the frame.
        # persistent=False: moves with .to(device) but stays out of the checkpoint.
        if mesh_masks is not None:
            self.register_buffer('mesh_masks', mesh_masks, persistent=False)
        else:
            self.mesh_masks: Optional[torch.Tensor] = None

        self._probe_metrics: dict = {}
        self._step_offset = 0   # set by load_from_pt to preserve GAN curriculum
        # Normalisation stats this run trains with — written into the checkpoint so
        # inference cannot silently use different stats (see infer.load_stats).
        self.norm_mean: Optional[list] = None
        self.norm_std:  Optional[list] = None

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
        if self.discriminator is not None and 'discriminator' in ckpt:
            try:
                self.discriminator.load_state_dict(ckpt['discriminator'], strict=False)
            except RuntimeError as e:
                print(f'  [warn] discriminator not restored: {e}')
        if self.stride_probe is not None and 'stride_probe' in ckpt:
            try:
                self.stride_probe.load_state_dict(ckpt['stride_probe'])
            except RuntimeError as e:          # e.g. number of strides changed
                print(f'  [warn] stride_probe not restored ({e}); starting fresh')
        saved_step = ckpt.get('global_step', 0)
        print(f'  Loaded .pt checkpoint (global_step={saved_step})')
        return saved_step

    def _adv_weight(self) -> float:
        s = self.global_step + self._step_offset
        if not self.use_gan or s < self.gan_start_step:
            return 0.0
        ramp = min(1.0, (s - self.gan_start_step) / max(1, self.gan_ramp_steps))
        return self.cfg.disc_adv_weight * ramp

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # With the GAN disabled, configure_optimizers returns a single optimizer, so
        # self.optimizers() is that optimizer rather than a [gen, disc] list.  adv_w is
        # then always 0, so the reconstruction-only branch runs and disc_opt is unused.
        if self.use_gan:
            gen_opt, disc_opt = self.optimizers()  # type: ignore[misc]
        else:
            gen_opt = self.optimizers()            # type: ignore[assignment]
        scheduler = self.lr_schedulers()

        # (frames, mesh_id, bc) when the datamodule tags labels, else bare frames
        mesh_ids = bc = None
        if isinstance(batch, (tuple, list)):
            mesh_ids = batch[1] if len(batch) > 1 else None
            bc       = batch[2] if len(batch) > 2 else None
            batch    = batch[0]
        frames = [batch[:, t] for t in range(batch.shape[1])]
        n        = self.cfg.n_context_frames

        # ---- sample a temporal stride and subsample the frame sequence ----
        # One stride per step (batch shares it — the subsample is on the time axis);
        # the random start offset uses the slack left by smaller strides as temporal
        # augmentation.  Mirrors GANTrainer.step exactly.
        H = max(1, getattr(self.cfg, 'rollout_horizon', 1))
        s_idx = int(torch.randint(len(self.time_strides), (1,)).item())
        s = self.time_strides[s_idx]
        need = (n + H) * s + 1
        if need > len(frames):            # sequence too short for this stride
            s_idx, s = 0, self.time_strides[0]
            need = (n + H) * s + 1
        off = int(torch.randint(len(frames) - need + 1, (1,)).item())
        frames = frames[off : off + need : s]

        x_in     = frames[n]
        x_target = frames[n + 1]
        # per-sample geometry: index the mesh table, else fall back to the shared mask
        mask     = (self.mesh_masks[mesh_ids.to(self.mesh_masks.device).long()]
                    if (mesh_ids is not None and self.mesh_masks is not None)
                    else self.pixel_mask)
        adv_w    = self._adv_weight()

        context = self.context_encoder(frames[:n], pixel_mask=mask)

        # ---- stride probe: recover the timestep from the context alone ----
        probe_term = None
        sacc = None
        w_probe = getattr(self.cfg, 'stride_cls_weight', 0.0)
        if self.stride_probe is not None and w_probe > 0.0:
            pm = {}
            probe_term = probe_losses(self.stride_probe, context, s_idx,
                                      mesh_ids=mesh_ids, bc=bc, metrics=pm)
            sacc = torch.tensor(pm.get('stride_acc', 0.0))
            self._probe_metrics = pm

        # scheduled sampling (exposure-bias fix): with probability self_input_prob,
        # feed the model its own no-grad prediction of frame n instead of the GT —
        # the target stays GT frame n+1 (see trainer.train_step_gan for rationale).
        # Largely subsumed by the horizon>1 rollout below, but kept for horizon=1.
        if float(torch.rand(())) < self.hparams['self_input_prob']:
            with torch.no_grad():
                x_in = self.model(frames[n - 1], context, pixel_mask=mask).float()
            if mask is not None:
                x_in = x_in * mask

        # ---- unrolled prediction (backprop-through-time over the horizon) ----
        # Predict `horizon` frames, feeding each prediction back as the next input with
        # gradient kept, so the model is trained to survive its OWN outputs — the fix
        # for autoregressive rollout collapse.  The reconstruction loss averages all
        # horizon targets; pred_0 (first step) is what the GAN/adv term operates on, so
        # GAN behaviour matches single-step training.
        horizon = max(1, min(getattr(self.cfg, 'rollout_horizon', 1), len(frames) - n - 1))
        recon_terms = []
        x_cur = x_in
        pred_disc = None           # first-step prediction, holes zeroed, for the GAN
        for k in range(horizon):
            pred_k = self.model(x_cur, context, pixel_mask=mask)
            # Unmasked for the criterion (gradient flows through holes via the hole
            # loss); masked copy is fed forward and used by the discriminator.
            recon_terms.append(self.criterion(pred_k, frames[n + 1 + k],
                                              pixel_mask=mask))
            pred_k_m = pred_k.float() * mask if mask is not None else pred_k.float()
            if pred_disc is None:
                pred_disc = pred_k_m
            x_cur = pred_k_m       # feed prediction forward, keeps grad
        recon = torch.stack(recon_terms).mean()

        # Persistence baseline, matched to the SAME rollout: "predict no change" scored
        # against every horizon target, exactly as recon is.  A single-step baseline is
        # wrong here — the field drifts further from x_in at each step, so an EXACT
        # persistence model (which is what a zero-init residual model is) scores ~4.9x
        # the 1-step loss at horizon 4, and `ratio` would start ~4.9 rather than 1.
        # Matched this way, ratio == 1 means "as good as persistence" and < 1 means the
        # model genuinely beats it — at any horizon.
        with torch.no_grad():
            persist = torch.stack(
                [self.criterion(x_in, frames[n + 1 + k], pixel_mask=mask)
                 for k in range(horizon)]).mean()
            ratio   = recon.detach() / persist.clamp(min=1e-8)

        # ---- reconstruction only (GAN off / pre-GAN) ----
        if adv_w == 0.0:
            gen_loss = recon if probe_term is None else recon + w_probe * probe_term
            gen_opt.zero_grad()
            self.manual_backward(gen_loss)
            self.clip_gradients(gen_opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')  # type: ignore[arg-type]
            gen_opt.step()
            scheduler.step()  # type: ignore[union-attr]
            logs = {'recon': recon, 'persist': persist, 'ratio': ratio,
                    's': float(s)}
            if sacc is not None:
                logs['sacc'] = sacc
                for k in ('mesh_acc', 'bc_mse'):
                    if k in self._probe_metrics:
                        logs[k] = self._probe_metrics[k]
            self.log_dict(logs, prog_bar=True, sync_dist=True)
            return

        assert pred_disc is not None

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

        # recon is the rollout mean computed above; the adv term acts on pred_0.
        # Only apply adversarial loss when the discriminator is in the healthy range
        if disc_healthy:
            adv_logit = self.discriminator(pred_disc, x_in_d, context)
            adv_loss  = F.binary_cross_entropy_with_logits(adv_logit, torch.ones_like(adv_logit))
            g_loss    = recon + adv_w * adv_loss
        else:
            g_loss = recon
        if probe_term is not None:
            g_loss = g_loss + w_probe * probe_term

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        self.clip_gradients(gen_opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
        gen_opt.step()
        scheduler.step()  # type: ignore[union-attr]

        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        logs = {'recon': recon, 'persist': persist, 'ratio': ratio,
                'disc': d_loss, 'adv_w': adv_w, 's': float(s)}
        if sacc is not None:
            logs['sacc'] = sacc
            for k in ('mesh_acc', 'bc_mse'):
                if k in self._probe_metrics:
                    logs[k] = self._probe_metrics[k]
        self.log_dict(logs, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):  # type: ignore[override]
        gen_params = list(self.model.parameters()) + list(self.context_encoder.parameters())
        if self.stride_probe is not None:
            gen_params += list(self.stride_probe.parameters())
        gen_opt  = optim.AdamW(gen_params,
                               lr=self.hparams['lr'],
                               weight_decay=self.hparams['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            gen_opt, T_max=self.hparams['cosine_t_max']
        )
        sched_cfg = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        if self.discriminator is None:        # GAN disabled → generator optimizer only
            return {'optimizer': gen_opt, 'lr_scheduler': sched_cfg}
        disc_opt = optim.Adam(self.discriminator.parameters(),
                              lr=self.cfg.disc_lr, betas=(0.5, 0.999))
        return ([gen_opt, disc_opt], [sched_cfg])

    def on_fit_start(self) -> None:
        """Capture the datamodule's normalisation stats once its setup() has run, so
        on_save_checkpoint can pin them into every checkpoint."""
        dm = getattr(self.trainer, 'datamodule', None)
        inner = getattr(dm, '_inner', None) if dm is not None else None
        if inner is not None and getattr(inner, 'mean', None) is not None \
                             and getattr(inner, 'std', None) is not None:
            self.norm_mean = [float(v) for v in inner.mean]
            self.norm_std  = [float(v) for v in inner.std]
            print(f'  Normalisation pinned into checkpoints: mean={self.norm_mean}')

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Inject HFM-native keys so infer.py can read this .ckpt file directly
        checkpoint['model']           = self.model.state_dict()
        checkpoint['context_encoder'] = self.context_encoder.state_dict()
        if self.discriminator is not None:
            checkpoint['discriminator'] = self.discriminator.state_dict()
        if self.stride_probe is not None:
            checkpoint['stride_probe'] = self.stride_probe.state_dict()
        checkpoint['cfg']             = self.cfg
        checkpoint['global_step']     = self.global_step
        # Pin the normalisation so inference reproduces training exactly.
        _lst = lambda v: None if v is None else [float(x) for x in v]
        checkpoint['norm_mean'] = _lst(self.norm_mean)
        checkpoint['norm_std']  = _lst(self.norm_std)

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
        return_mesh_id: bool = True,
    ):
        super().__init__()
        self._return_mesh_id = return_mesh_id
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
            return_mesh_id = self._return_mesh_id,
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
