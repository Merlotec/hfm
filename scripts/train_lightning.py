"""
Lightning training script for HFM — single GPU or multi-GPU DDP.

Single GPU:
    python scripts/train_lightning.py

Dual 5090:
    python scripts/train_lightning.py --devices 2

Resume from latest checkpoint:
    python scripts/train_lightning.py --devices 2 --resume latest

Resume from specific checkpoint:
    python scripts/train_lightning.py --devices 2 --resume checkpoints/ckpt-step010000.ckpt
"""

from __future__ import annotations
from typing import Optional, Union

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfm.config import HFMConfig
from hfm.data import build_renderer, load_pixel_mask
from hfm.lightning_module import HFMLightningModule, FVMLightningDataModule

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT            = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = _ROOT.parent / 'data' / 'fvm_gen_v2'
CKPT_DIR         = _ROOT / 'checkpoints'
HYPERPARAMS      = _ROOT / 'hyperparams.json'

# ---------------------------------------------------------------------------
# GAN curriculum constants
# ---------------------------------------------------------------------------

# Keep in sync with scripts/train.py — a checkpoint must see the same GAN
# curriculum regardless of which script trains it.
GAN_START_STEP        = 2_500   # step at which adversarial loss switches on
GAN_RAMP_STEPS        = 2_000   # adv_weight ramps 0 → disc_adv_weight over this
DISC_UPDATE_THRESHOLD = 0.5
SELF_INPUT_PROB       = 0.5     # scheduled sampling: P(input = own prediction);
                                # forced to 0 whenever random context is active

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> tuple[HFMConfig, dict]:
    with open(HYPERPARAMS) as f:
        hp = json.load(f)
    m = hp['model']
    t = hp['training']
    # Pass through the fixed-quadtree knobs that are present (same as scripts/train.py),
    # so the Lightning path honours ml_dims/ml_passes/ml_untie_passes/etc. instead of
    # silently using the HFMConfig defaults.
    ml_keys = ('use_quadtree', 'ml_levels', 'ml_finest_px', 'ml_dims', 'ml_passes',
               'ml_blocks_per_level', 'ml_untie_passes',
               'residual_prediction', 'mask_aware_decoder', 'persist_norm_loss',
               'ctx_temporal_diffs')
    ml_kwargs = {k: m[k] for k in ml_keys if k in m}
    cfg = HFMConfig(
        img_size               = m['img_size'],
        in_channels            = m['in_channels'],
        patch_px               = m['patch_px'],
        d_patch                = m['d_patch'],
        skip_ch                = m.get('skip_ch', 32),
        n_global_tokens        = m.get('n_global_tokens', 16),
        n_heads                = m.get('n_heads', 8),
        noise_std              = m.get('noise_std', 0.05),
        n_layers               = m['n_layers'],
        mlp_ratio              = m['mlp_ratio'],
        dropout                = m['dropout'],
        rollout_horizon        = m.get('rollout_horizon', 4),
        time_strides           = tuple(m.get('time_strides',
                                          HFMConfig.time_strides)),
        stride_cls_weight      = m.get('stride_cls_weight', 0.5),
        n_context_frames       = m['n_context_frames'],
        ctx_patch_px           = m['ctx_patch_px'],
        d_ctx                  = m['d_ctx'],
        n_ctx_tokens           = m['n_ctx_tokens'],
        n_ctx_layers           = m['n_ctx_layers'],
        n_ctx_heads            = m['n_ctx_heads'],
        disc_dim               = m['disc_dim'],
        disc_adv_weight        = m['disc_adv_weight'],
        disc_lr                = m['disc_lr'],
        gradient_checkpointing = True,
        **ml_kwargs,
    )
    return cfg, t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Train HFM with Lightning')
    parser.add_argument('--data',        type=Path, nargs='+',
                        default=[DEFAULT_DATA_DIR],
                        help='Dataset root(s); several may be given, and the '
                             'corpora are balanced by the datamodule')
    parser.add_argument('--val-fraction', type=float, default=0.05,
                        help='Fraction of runs held out for validation '
                             '(deterministic run-name split; 0 disables)')
    parser.add_argument('--val-budget',  type=int, default=512,
                        help='Max validation windows scored per epoch')
    parser.add_argument('--settle-time', type=float, default=None,
                        help='Override settle_time from hyperparams.json: '
                             'sim-time (s) excluded from the start of cold-start runs')
    parser.add_argument('--resume',      type=str,  default=None, nargs='?', const='latest',
                        help='Lightning .ckpt path, or "latest" to auto-select')
    parser.add_argument('--resume-pt',   type=str,  default=None,
                        help='Load weights from a GANTrainer .pt checkpoint (fresh optimizer)')
    parser.add_argument('--epochs',      type=int,  default=None)
    parser.add_argument('--devices',     type=int,  default=1,
                        help='GPUs per node')
    parser.add_argument('--nodes',       type=int,  default=1,
                        help='Number of nodes for multi-node training')
    parser.add_argument('--batch-size',  type=int,  default=None,
                        help='Override batch_size from hyperparams.json')
    parser.add_argument('--log-every',   type=int,  default=50)
    parser.add_argument('--ckpt-every-steps', type=int, default=500,
                        help='Write a step checkpoint (ckpt-step*.ckpt) every N steps (0 disables)')
    parser.add_argument('--workers',     type=int,  default=4)
    args = parser.parse_args()

    # Let fp32 matmuls use TF32 tensor cores (Ampere+/CUDA, and the XPU equivalent).
    # bf16 autocast already covers most matmuls; this picks up the ones that stay fp32
    # and silences Lightning's "you have Tensor Cores" warning.  Precision loss is
    # confined to matmul accumulation and is irrelevant next to bf16 activations.
    torch.set_float32_matmul_precision('high')

    cfg, train_hp = load_config()
    n_epochs   = args.epochs    or train_hp['n_epochs']
    batch_size = args.batch_size or train_hp['batch_size']
    # context frames + input frame + one target per rollout step (matches scripts/train.py)
    # (context + rollout targets) at the LARGEST stride, + 1 — each training step
    # subsamples every s-th frame from this window (see HFMLightningModule).
    max_stride = max(getattr(cfg, 'time_strides', (1,)) or (1,))
    seq_len    = (cfg.n_context_frames + getattr(cfg, 'rollout_horizon', 1)) * max_stride + 1

    # Random causal context block, unless temporal striding is on — the two are
    # mutually exclusive (the stride subsample would cut across the two-block
    # splice; see scripts/train.py for the full rationale).
    ctx_n = cfg.n_context_frames
    if max_stride > 1:
        print(f'  [WARN] time_strides={cfg.time_strides}: random context frames '
              f'disabled (incompatible with temporal striding)')
        ctx_n = None
    settle_time = (args.settle_time if args.settle_time is not None
                   else train_hp.get('settle_time', 0.0))

    # ---- data module (built and set up FIRST: the probe label dimensions, the
    # per-sample mesh masks, and whether a val split exists all come from it,
    # exactly as in scripts/train.py) ----
    dm = FVMLightningDataModule(
        data_dir     = args.data,
        seq_len      = seq_len,
        resolution   = (cfg.img_size, cfg.img_size),
        batch_size   = batch_size,
        num_workers  = args.workers,
        val_fraction = args.val_fraction,
        val_budget   = args.val_budget,
        n_context    = ctx_n,
        settle_time  = settle_time,
    )
    dm.setup()
    inner = dm._inner
    assert inner is not None

    # ---- pixel mask (loaded once in main process; moved to each GPU as a buffer) ----
    primary = args.data[0]
    mesh_dirs = []
    if (primary / 'shared_mesh.pkl').exists():
        mesh_dirs.append(primary)
    else:
        for p in primary.iterdir():
            if p.is_dir() and (p / 'shared_mesh.pkl').exists():
                mesh_dirs.append(p)

    if not mesh_dirs:
        raise RuntimeError(f'No shared_mesh.pkl found in {primary} or its subdirectories')

    first_mdir = mesh_dirs[0]
    renderer   = build_renderer(first_mdir, (cfg.img_size, cfg.img_size))
    pixel_mask = load_pixel_mask(first_mdir, renderer, (cfg.img_size, cfg.img_size))
    # Per-sample masks from the datamodule: canonical (sorted) ordering across
    # EVERY root passed to --data, so row i really is geometry i.
    mesh_masks = inner.mesh_masks
    if mesh_masks is not None:
        print(f'Per-sample masks: {mesh_masks.shape[0]} geometries')
    has_val = inner._val_dataset is not None
    if has_val:
        print(f'Validation:      {len(inner._val_dataset)} sequences from held-out '
              f'runs ({inner.val_fraction:.0%} of the corpus)')
    else:
        print('Validation:      disabled (val_fraction=0)')

    # ---- resolve checkpoint ----
    resume: Optional[str] = args.resume
    if resume == 'latest':
        candidates = sorted(CKPT_DIR.glob('ckpt-step*.ckpt'))
        if not candidates:
            print(f'No ckpt-step*.ckpt found in {CKPT_DIR}')
            sys.exit(1)
        resume = str(candidates[-1])
        print(f'Auto-selected: {candidates[-1].name}')

    # ---- module ----
    module = HFMLightningModule(
        cfg                   = cfg,
        lr                    = train_hp['lr'],
        weight_decay          = train_hp['weight_decay'],
        l1_weight             = train_hp['l1_weight'],
        grad_weight           = train_hp.get('grad_weight', 0.0),
        tile_weight           = train_hp.get('tile_weight', 0.0),
        gan_start_step        = GAN_START_STEP,
        gan_ramp_steps        = GAN_RAMP_STEPS,
        disc_update_threshold = train_hp.get('disc_update_threshold', DISC_UPDATE_THRESHOLD),
        cosine_t_max          = train_hp.get('cosine_t_max', 10_000),
        pixel_mask            = pixel_mask,
        mesh_masks            = mesh_masks,
        # Scheduled sampling is invalid with the random context block (the
        # module hard-errors on the combination; scripts/train.py has the full
        # story).  The horizon>1 BPTT rollout is the real exposure-bias fix.
        self_input_prob       = (0.0 if ctx_n is not None else SELF_INPUT_PROB),
        ctx_random            = ctx_n is not None,
        # Context-probe label spaces, read off the corpus like scripts/train.py:
        # without these the theta-regression and viscosity-class heads are never
        # built and the physics context gets no gradient pressure.
        probe_bc_dim          = getattr(inner, 'bc_dim', 0),
        probe_ctx_dim         = getattr(inner, 'ctx_dim', 0),
        probe_n_visc_models   = getattr(inner, 'n_visc_models', 0),
        use_gan               = (os.environ.get('HFM_USE_GAN') not in ('0', 'false', 'False'))
                                 if os.environ.get('HFM_USE_GAN') is not None
                                 else bool(train_hp.get('use_gan', True)),
    )

    if args.resume_pt:
        print(f'Loading weights from .pt: {args.resume_pt}')
        saved_step = module.load_from_pt(args.resume_pt)
        module._step_offset = saved_step   # keeps GAN curriculum position correct

    n_gen  = sum(p.numel() for p in module.model.parameters())           / 1e6
    n_ctx  = sum(p.numel() for p in module.context_encoder.parameters()) / 1e6
    n_disc = (sum(p.numel() for p in module.discriminator.parameters()) / 1e6
              if module.discriminator is not None else 0.0)
    print(f'Generator:      {n_gen:.1f}M params')
    print(f'ContextEncoder: {n_ctx:.1f}M params')
    print(f'Discriminator:  {n_disc:.1f}M params'
          + ('' if module.discriminator is not None else '  [DISABLED: use_gan=false]'))
    print(f'Devices:        {args.devices}  |  seq_len: {seq_len}  |  batch: {batch_size}\n')

    # ---- callbacks ----
    CKPT_DIR.mkdir(exist_ok=True)
    callbacks = [
        # Epoch checkpoints: coarse safety net.
        ModelCheckpoint(
            dirpath          = str(CKPT_DIR),
            filename         = 'ckpt-epoch{epoch:03d}',
            every_n_epochs   = 2,
            save_top_k       = -1,
            save_last        = False,
            auto_insert_metric_name = False,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    # Step checkpoints: ckpt-step*.ckpt is also what `--resume latest` globs, so this
    # is what makes mid-epoch resume work.  save_last writes last.ckpt on this cadence.
    if args.ckpt_every_steps > 0:
        callbacks.insert(1, ModelCheckpoint(
            dirpath             = str(CKPT_DIR),
            filename            = 'ckpt-step{step:06d}',
            every_n_train_steps = args.ckpt_every_steps,
            save_top_k          = -1,
            save_last           = True,
            auto_insert_metric_name = False,
        ))

    # ---- trainer ----
    multi_gpu = args.devices > 1 or args.nodes > 1
    trainer = L.Trainer(
        accelerator          = 'auto',
        devices              = args.devices,
        num_nodes            = args.nodes,
        strategy             = 'ddp_find_unused_parameters_true' if multi_gpu else 'auto',
        max_epochs           = n_epochs,
        callbacks            = callbacks,
        log_every_n_steps    = args.log_every,
        num_sanity_val_steps = 0,
        # 0 disables the val loop entirely (val_dataloader is never even
        # requested) when there is no held-out split to score.
        limit_val_batches    = 1.0 if has_val else 0,
        enable_progress_bar  = True,
        precision            = train_hp.get('precision', 'bf16-mixed'),
    )

    trainer.fit(module, dm, ckpt_path=resume)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
