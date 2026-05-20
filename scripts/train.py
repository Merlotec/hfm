"""
Full-dataset training for HFM (GANTrainer).

Two-stage curriculum, managed automatically:
  Stage 1 (steps 0 → gan_start_step):   reconstruction + warmup ramp
    n_warmup ramps 1 → cfg.n_warmup_frames over warmup_ramp_steps steps.
  Stage 2 (steps gan_start_step → end): GAN active
    adv_weight ramps 0 → cfg.disc_adv_weight over gan_ramp_steps steps.

Run from the hfm/ root:
    python scripts/train.py
    python scripts/train.py --data /path/to/dataset
    python scripts/train.py --resume checkpoints/train_step010000.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfm import HFMConfig, GANTrainer
from hfm.data import FVMDataModule, build_renderer, load_pixel_mask

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT     = Path(__file__).resolve().parents[1]
_DATA_ROOT = _ROOT.parent / 'fvm_model' / 'data'

DEFAULT_DATA_DIR = _DATA_ROOT / 'fvm_gen_datasets'
CKPT_DIR         = _ROOT / 'checkpoints'
HYPERPARAMS      = _ROOT / 'hyperparams.json'

# ---------------------------------------------------------------------------
# GAN curriculum constants (not in hyperparams.json)
# ---------------------------------------------------------------------------

WARMUP_RAMP_STEPS     = 5_000   # n_warmup ramps 1 → max over this many steps
GAN_START_STEP        = 10_000  # step at which adversarial loss switches on
GAN_RAMP_STEPS        = 1_000   # adv_weight ramps 0 → disc_adv_weight over this
DISC_UPDATE_THRESHOLD = 0.3     # skip disc update when d_loss <= this

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> tuple[HFMConfig, dict]:
    with open(HYPERPARAMS) as f:
        hp = json.load(f)
    m = hp['model']
    t = hp['training']
    cfg = HFMConfig(
        img_size             = m['img_size'],
        in_channels          = m['in_channels'],
        patch_px             = m['patch_px'],
        d_patch              = m['d_patch'],
        d_resid              = m['d_resid'],
        feat_sizes           = tuple(m['feat_sizes']),
        d_feat               = tuple(m['d_feat']),
        d_sys                = tuple(m['d_sys']),
        patch_local_radius   = m['patch_local_radius'],
        feat_window_size     = m['feat_window_size'],
        n_heads_patch        = m['n_heads_patch'],
        n_heads_feat         = tuple(m['n_heads_feat']),
        n_heads_sys          = tuple(m['n_heads_sys']),
        n_heads_resid        = m['n_heads_resid'],
        n_layers             = m['n_layers'],
        mlp_ratio            = m['mlp_ratio'],
        dropout              = m['dropout'],
        n_warmup_frames      = t['n_warmup_frames'],
        gradient_checkpointing = True,
    )
    return cfg, t


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train HFM on fluid simulation data')
    parser.add_argument('--data',       type=Path, default=DEFAULT_DATA_DIR,
                        help='Dataset directory containing simulation subdirs')
    parser.add_argument('--resume',     type=Path, default=None,
                        help='GANTrainer checkpoint to resume from')
    parser.add_argument('--epochs',     type=int,  default=None,
                        help='Override n_epochs from hyperparams.json')
    parser.add_argument('--log-every',  type=int,  default=50,
                        help='Print a log line every N steps')
    parser.add_argument('--ckpt-every', type=int,  default=200,
                        help='Save a mid-epoch checkpoint every N steps')
    args = parser.parse_args()

    device = get_device()
    print(f'Device: {device}')

    cfg, train_hp = load_config()
    n_epochs = args.epochs or train_hp['n_epochs']
    seq_len  = cfg.n_warmup_frames + 2   # warmup frames + input frame + target frame

    # ---- data ----
    dm = FVMDataModule(
        data_dir    = args.data,
        seq_len     = seq_len,
        batch_size  = train_hp['batch_size'],
        num_workers = 4,
    )
    dm.setup()

    renderer   = build_renderer(args.data, (cfg.img_size, cfg.img_size))
    pixel_mask = load_pixel_mask(
        args.data, renderer, (cfg.img_size, cfg.img_size)
    ).to(device)

    # ---- trainer ----
    trainer = GANTrainer(
        cfg,
        lr                    = train_hp['lr'],
        weight_decay          = train_hp['weight_decay'],
        l1_weight             = train_hp['l1_weight'],
        warmup_ramp_steps     = WARMUP_RAMP_STEPS,
        gan_start_step        = GAN_START_STEP,
        gan_ramp_steps        = GAN_RAMP_STEPS,
        disc_update_threshold = DISC_UPDATE_THRESHOLD,
    )
    trainer.to(device)

    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer.load(str(args.resume))

    n_gen  = sum(p.numel() for p in trainer.model.parameters())         / 1e6
    n_disc = sum(p.numel() for p in trainer.discriminator.parameters()) / 1e6
    print(f'Generator:     {n_gen:.1f}M params')
    print(f'Discriminator: {n_disc:.1f}M params')
    print(f'Dataset:       {len(dm._dataset)} sequences  (seq_len={seq_len})')
    print(f'Curriculum:    warmup ramps over {WARMUP_RAMP_STEPS} steps, '
          f'GAN activates at step {GAN_START_STEP}\n')

    CKPT_DIR.mkdir(exist_ok=True)
    nan_streak = 0

    for epoch in range(n_epochs):
        for batch in dm.train_dataloader():          # [B, T, C, H, W]
            frames = [batch[:, t].to(device) for t in range(batch.shape[1])]

            recon, disc = trainer.step(frames, pixel_mask=pixel_mask)
            info = trainer.training_info()
            step = info['global_step']

            if not math.isfinite(recon):
                nan_streak += 1
                print(f'  [WARN] step {step}: NaN/inf loss (streak={nan_streak})')
                if nan_streak >= 10:
                    path = CKPT_DIR / f'train_EMERGENCY_step{step:06d}.pt'
                    trainer.save(str(path))
                    print(f'10 consecutive NaN steps — saved emergency checkpoint and stopping.')
                    sys.exit(1)
                continue
            nan_streak = 0

            if step % args.log_every == 0:
                print(
                    f'epoch {epoch:3d}  step {step:6d} | '
                    f'nw={info["n_warmup"]}  '
                    f'recon={recon:.4f}  '
                    f'disc={disc:.4f}  '
                    f'adv_w={info["adv_weight"]:.3f}'
                )

            if step % args.ckpt_every == 0:
                path = CKPT_DIR / f'train_step{step:06d}.pt'
                trainer.save(str(path))
                print(f'  [ckpt] {path.name}')

        path = CKPT_DIR / f'train_epoch{epoch:03d}.pt'
        trainer.save(str(path))
        print(f'  [ckpt] {path.name}')


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
