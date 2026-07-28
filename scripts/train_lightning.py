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

GAN_START_STEP        = 10_000
GAN_RAMP_STEPS        = 2_000
DISC_UPDATE_THRESHOLD = 0.5

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
               'ml_blocks_per_level', 'ml_untie_passes')
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
    parser.add_argument('--data',        type=Path, default=DEFAULT_DATA_DIR,
                        help='Dataset directory')
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

    cfg, train_hp = load_config()
    n_epochs   = args.epochs    or train_hp['n_epochs']
    batch_size = args.batch_size or train_hp['batch_size']
    # context frames + input frame + one target per rollout step (matches scripts/train.py)
    seq_len    = cfg.n_context_frames + 1 + getattr(cfg, 'rollout_horizon', 1)

    # ---- pixel mask (loaded once in main process; moved to each GPU as a buffer) ----
    mesh_dirs = []
    if (args.data / 'shared_mesh.pkl').exists():
        mesh_dirs.append(args.data)
    else:
        for p in args.data.iterdir():
            if p.is_dir() and (p / 'shared_mesh.pkl').exists():
                mesh_dirs.append(p)
                
    if not mesh_dirs:
        raise RuntimeError(f'No shared_mesh.pkl found in {args.data} or its subdirectories')
        
    first_mdir = mesh_dirs[0]
    renderer   = build_renderer(first_mdir, (cfg.img_size, cfg.img_size))
    pixel_mask = load_pixel_mask(first_mdir, renderer, (cfg.img_size, cfg.img_size))

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
        gan_start_step        = GAN_START_STEP,
        gan_ramp_steps        = GAN_RAMP_STEPS,
        disc_update_threshold = DISC_UPDATE_THRESHOLD,
        cosine_t_max          = train_hp.get('cosine_t_max', 10_000),
        pixel_mask            = pixel_mask,
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

    # ---- data module ----
    dm = FVMLightningDataModule(
        data_dir    = str(args.data),
        seq_len     = seq_len,
        resolution  = (cfg.img_size, cfg.img_size),
        batch_size  = batch_size,
        num_workers = args.workers,
    )

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
        enable_progress_bar  = True,
        precision            = train_hp.get('precision', 'bf16-true'),
    )

    trainer.fit(module, dm, ckpt_path=resume)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
