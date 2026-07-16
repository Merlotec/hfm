"""
Full-dataset training for HFM (GANTrainer).

Two-stage curriculum, managed automatically:
  Stage 1 (steps 0 → gan_start_step):   reconstruction only
  Stage 2 (steps gan_start_step → end): GAN active
    adv_weight ramps 0 → cfg.disc_adv_weight over gan_ramp_steps steps.

Run from the hfm/ root:
    python scripts/train.py
    python scripts/train.py --data /path/to/dataset
    python scripts/train.py --resume checkpoints/train_step010000.pt
"""

import argparse
import csv
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

_ROOT      = Path(__file__).resolve().parents[1]
_DATA_ROOT = _ROOT.parent / 'data'

DEFAULT_DATA_DIR = _DATA_ROOT / 'fvm_gen_datasets'
DEFAULT_TEST_DIR = _DATA_ROOT / 'test'
CKPT_DIR         = _ROOT / 'checkpoints'
HYPERPARAMS      = _ROOT / 'hyperparams.json'

# ---------------------------------------------------------------------------
# GAN curriculum constants (not in hyperparams.json)
# ---------------------------------------------------------------------------

GAN_START_STEP        = 10_000  # step at which adversarial loss switches on
GAN_RAMP_STEPS        = 2_000   # adv_weight ramps 0 → disc_adv_weight over this
DISC_UPDATE_THRESHOLD = 0.5     # skip disc update when d_loss <= this

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> tuple[HFMConfig, dict]:
    with open(HYPERPARAMS) as f:
        hp = json.load(f)
    m = hp['model']
    t = hp['training']
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
    )
    return cfg, t


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _save_loss_plot(log_path: Path, plot_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            v = row['val_loss']
            val_losses.append(float(v) if v and v != 'nan' else float('nan'))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label='train', linewidth=1.5)
    if any(math.isfinite(v) for v in val_losses):
        ax.plot(epochs, val_losses, label='val (test set)', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction loss')
    ax.set_title('HFM training loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train HFM on fluid simulation data')
    parser.add_argument('--data',       type=Path, default=DEFAULT_DATA_DIR,
                        help='Dataset directory containing simulation subdirs')
    parser.add_argument('--test-data',  type=Path, default=DEFAULT_TEST_DIR,
                        help='Test/validation directory for out-of-sample loss (default: ../data/test)')
    parser.add_argument('--resume',     type=str, default=None, nargs='?', const='latest',
                        help='Checkpoint to resume from. Omit value to pick the latest train_step checkpoint.')
    parser.add_argument('--epochs',     type=int,  default=None,
                        help='Override n_epochs from hyperparams.json')
    parser.add_argument('--log-every',  type=int,  default=50,
                        help='Print a log line every N steps')
    args = parser.parse_args()

    device = get_device()
    print(f'Device: {device}')

    cfg, train_hp = load_config()
    n_epochs = args.epochs or train_hp['n_epochs']
    seq_len  = cfg.n_context_frames + 2   # context frames + input frame + target frame

    # ---- training data ----
    dm = FVMDataModule(
        data_dir    = args.data,
        seq_len     = seq_len,
        batch_size  = train_hp['batch_size'],
        num_workers = 4,
    )
    dm.setup()

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
    pixel_mask = load_pixel_mask(
        first_mdir, renderer, (cfg.img_size, cfg.img_size)
    ).to(device)

    # ---- test/validation data (optional) ----
    val_dl         = None
    val_pixel_mask = None
    if args.test_data.exists():
        print(f'Test data: {args.test_data}')
        val_dm = FVMDataModule(
            data_dir    = args.test_data,
            seq_len     = seq_len,
            batch_size  = train_hp['batch_size'],
            num_workers = 4,
            mean        = dm.mean,
            std         = dm.std,
        )
        val_dm.setup()
        val_mesh_dirs = []
        if (args.test_data / 'shared_mesh.pkl').exists():
            val_mesh_dirs.append(args.test_data)
        else:
            for p in args.test_data.iterdir():
                if p.is_dir() and (p / 'shared_mesh.pkl').exists():
                    val_mesh_dirs.append(p)
                    
        if not val_mesh_dirs:
            raise RuntimeError(f'No shared_mesh.pkl found in {args.test_data} or its subdirectories')
            
        first_val_mdir = val_mesh_dirs[0]
        val_renderer   = build_renderer(first_val_mdir, (cfg.img_size, cfg.img_size))
        val_pixel_mask = load_pixel_mask(
            first_val_mdir, val_renderer, (cfg.img_size, cfg.img_size)
        ).to(device)
        val_dl = val_dm.val_dataloader()
        assert val_dm._dataset is not None
        print(f'Test dataset:    {len(val_dm._dataset)} sequences\n')
    else:
        print(f'Test data dir not found ({args.test_data}) — val loss will not be computed\n')

    # ---- trainer ----
    trainer = GANTrainer(
        cfg,
        lr                    = train_hp['lr'],
        weight_decay          = train_hp['weight_decay'],
        l1_weight             = train_hp['l1_weight'],
        gan_start_step        = GAN_START_STEP,
        gan_ramp_steps        = GAN_RAMP_STEPS,
        disc_update_threshold = DISC_UPDATE_THRESHOLD,
        pixel_mask            = pixel_mask,
    )
    trainer.to(device)

    if args.resume == 'latest':
        candidates = sorted(CKPT_DIR.glob('train_step*.pt'))
        if not candidates:
            print(f'No train_step checkpoints found in {CKPT_DIR}')
            sys.exit(1)
        args.resume = str(candidates[-1])
        print(f'Auto-selected checkpoint: {candidates[-1].name}')
    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer.load(str(args.resume))

    n_gen  = sum(p.numel() for p in trainer.model.parameters())           / 1e6
    n_ctx  = sum(p.numel() for p in trainer.context_encoder.parameters()) / 1e6
    n_disc = sum(p.numel() for p in trainer.discriminator.parameters())   / 1e6
    print(f'Generator:       {n_gen:.1f}M params')
    print(f'ContextEncoder:  {n_ctx:.1f}M params')
    print(f'Discriminator:   {n_disc:.1f}M params')
    assert dm._dataset is not None
    steps_per_epoch = math.ceil(len(dm._dataset) / train_hp['batch_size'])
    start_epoch     = trainer.global_step // steps_per_epoch
    print(f'Dataset:         {len(dm._dataset)} sequences  (seq_len={seq_len})')
    print(f'Curriculum:      GAN activates at step {GAN_START_STEP}')
    if start_epoch > 0:
        print(f'Resuming:        epoch {start_epoch} (step {trainer.global_step})')
    print()

    CKPT_DIR.mkdir(exist_ok=True)
    loss_log_path  = CKPT_DIR / 'loss_log.csv'
    loss_plot_path = CKPT_DIR / 'loss_plot.png'

    # Append mode — safe for resume; write header only when starting fresh
    write_header = not loss_log_path.exists()
    loss_csv    = open(loss_log_path, 'a', newline='')
    loss_writer = csv.writer(loss_csv)
    if write_header:
        loss_writer.writerow(['epoch', 'train_loss', 'val_loss'])
        loss_csv.flush()

    nan_streak = 0

    for epoch in range(start_epoch, n_epochs):
        epoch_recon_sum = 0.0
        epoch_recon_cnt = 0

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
                    loss_csv.close()
                    sys.exit(1)
                continue
            nan_streak = 0

            epoch_recon_sum += recon
            epoch_recon_cnt += 1

            if step % args.log_every == 0:
                print(
                    f'epoch {epoch:3d}  step {step:6d} | '
                    f'recon={recon:.4f}  '
                    f'disc={disc:.4f}  '
                    f'adv_w={info["adv_weight"]:.3f}'
                )

        train_loss = epoch_recon_sum / epoch_recon_cnt if epoch_recon_cnt > 0 else float('nan')

        val_loss = float('nan')
        if val_dl is not None:
            val_loss = trainer.validate(val_dl, pixel_mask=val_pixel_mask)

        print(
            f'  [epoch {epoch:3d}] train_loss={train_loss:.4f}  '
            f'val_loss={val_loss:.4f}'
        )
        loss_writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}'])
        loss_csv.flush()
        _save_loss_plot(loss_log_path, loss_plot_path)

        if (epoch + 1) % 2 == 0:
            path = CKPT_DIR / f'train_epoch{epoch:03d}.pt'
            trainer.save(str(path))
            print(f'  [ckpt] {path.name}')

    loss_csv.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
