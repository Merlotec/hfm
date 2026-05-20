"""
HFM inference script.

Loads a checkpoint, runs warmup on an initial window of frames from a simulation
run, then autoregressively predicts subsequent frames.  Outputs are saved to
--out-dir as:
  - frames_gt.npy   / frames_pred.npy   — raw normalised arrays [T, C, H, W]
  - frames_gt_phys.npy / frames_pred_phys.npy — denormalised physical values
  - images/t{NNN}_ch{C}.png             — ground-truth vs prediction per channel

Usage:
    python infer.py --checkpoint checkpoints/overfit_stage2_step02000.pt \\
                    --data-dir   /path/to/sim_dataset \\
                    --out-dir    out/overfit
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hfm import HFM
from hfm.data import build_renderer, FVMSequenceDataset, load_pixel_mask

DATA_ROOT = Path(__file__).resolve().parents[1] / 'fvm_model' / 'data'
FOUNDATION_STATS = Path(__file__).resolve().parents[1] / \
                   'fvm_model' / 'fvm_foundation' / 'input_stats.json'

CHANNEL_NAMES = ['rho', 'u', 'v', 'p']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_stats(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    for p in [data_dir / 'hfm_input_stats.json', FOUNDATION_STATS]:
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            return torch.tensor(s['mean']), torch.tensor(s['std'])
    raise FileNotFoundError('No normalisation stats found. Run overfit.py first.')


def denorm(frames: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """[T, C, H, W] normalised → physical units."""
    m = mean.numpy()[None, :, None, None]
    s = std.numpy()[None, :, None, None]
    return frames * s + m


def save_viewer_frames(
    gt_phys: np.ndarray,
    pred_phys: np.ndarray,
    timestamps: list[float],
    n_warmup: int,
    run_name: str,
    viewer_dir: Path,
):
    """
    Write t_*.npz files in the format expected by fvm_viewer/viewer.py -c.

    Layout:  viewer_dir / run_name / t_{timestamp}.npz
    Each file contains: grid (4, H, W), t (scalar float32), is_seed (bool).

    Warmup frames are marked is_seed=True; predicted frames is_seed=False.
    """
    run_dir = viewer_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    T = len(timestamps)
    for i, ts in enumerate(timestamps):
        if i < n_warmup:
            grid     = gt_phys[i].astype(np.float32)
            is_seed  = True
        else:
            grid     = pred_phys[i - n_warmup].astype(np.float32)
            is_seed  = False
        np.savez(run_dir / f't_{ts:.4g}.npz',
                 grid=grid, t=np.float32(ts), is_seed=np.bool_(is_seed))

    print(f'  Saved {T} viewer frames → {run_dir}')


def save_images(gt: np.ndarray, pred: np.ndarray, out_dir: Path,
                mean: torch.Tensor, std: torch.Tensor):
    """Save side-by-side GT vs prediction PNGs, one per timestep per channel."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available — skipping PNG output')
        return

    img_dir = out_dir / 'images'
    img_dir.mkdir(exist_ok=True)

    gt_phys   = denorm(gt,   mean, std)
    pred_phys = denorm(pred, mean, std)

    T, C = gt.shape[:2]
    for t in range(T):
        for c in range(C):
            _, axes = plt.subplots(1, 3, figsize=(14, 4))
            vmin = gt_phys[t, c].min()
            vmax = gt_phys[t, c].max()

            im0 = axes[0].imshow(gt_phys[t, c],   vmin=vmin, vmax=vmax, cmap='RdBu_r')
            im1 = axes[1].imshow(pred_phys[t, c], vmin=vmin, vmax=vmax, cmap='RdBu_r')
            err = np.abs(gt_phys[t, c] - pred_phys[t, c])
            im2 = axes[2].imshow(err, cmap='hot')

            axes[0].set_title(f'Ground truth — {CHANNEL_NAMES[c]}')
            axes[1].set_title(f'Prediction   — {CHANNEL_NAMES[c]}')
            axes[2].set_title(f'|Error|  max={err.max():.3f}')
            for ax in axes:
                ax.axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046)
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            plt.suptitle(f't={t}  channel={CHANNEL_NAMES[c]}', fontsize=12)
            plt.tight_layout()
            plt.savefig(img_dir / f't{t:03d}_ch{c}_{CHANNEL_NAMES[c]}.png',
                        dpi=100, bbox_inches='tight')
            plt.close()

    print(f'  Saved {T * C} images → {img_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='HFM inference')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--data-dir',   required=True,
                        help='Dataset directory (contains subdirs with t_*.npz files)')
    parser.add_argument('--out-dir',    required=True,
                        help='Output directory')
    parser.add_argument('--n-warmup',   type=int, default=None,
                        help='Warmup frames (defaults to cfg.n_warmup_frames)')
    parser.add_argument('--n-predict',  type=int, default=10,
                        help='Number of frames to predict autoregressively')
    parser.add_argument('--seq-start',  type=int, default=None,
                        help='Frame index to start from (defaults to middle of run)')
    parser.add_argument('--first-frame', type=int, default=20,
                        help='Skip this many initial transient frames')
    parser.add_argument('--no-images',  action='store_true',
                        help='Skip PNG generation')
    args = parser.parse_args()

    device   = get_device()
    out_dir  = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device:  {device}')
    print(f'Out dir: {out_dir}')

    # ---- load checkpoint ----
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg   = ckpt['cfg']
    model = HFM(cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'  stage={ckpt["stage"]}  step={ckpt["step"]}  loss={ckpt["loss"]:.5f}')

    n_warmup = args.n_warmup if args.n_warmup is not None else cfg.n_warmup_frames

    # ---- load data ----
    print(f'\nLoading data from {data_dir}')
    mean, std = load_stats(data_dir)
    renderer  = build_renderer(data_dir, (cfg.img_size, cfg.img_size), device='cpu')
    pixel_mask = load_pixel_mask(data_dir, renderer, (cfg.img_size, cfg.img_size)).to(device)

    sim_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not sim_dirs:
        raise RuntimeError(f'No simulation subdirectories found in {data_dir}')
    sim_dir = sim_dirs[0]
    print(f'  Using run: {sim_dir.name}')

    seq_len = n_warmup + args.n_predict + 1
    ds = FVMSequenceDataset.with_cache(
        sim_dir, renderer, seq_len, mean, std, first_frame=args.first_frame
    )
    print(f'  {len(ds)} sequences available (seq_len={seq_len})')

    start = args.seq_start if args.seq_start is not None else len(ds) // 2
    seq   = ds[start]                                   # [T, C, H, W]
    frames_gt = [seq[t:t+1].to(device) * pixel_mask for t in range(seq_len)]

    # Timestamps from source filenames (t_<value>.npz)
    timestamps = [float(ds.paths[start + t].stem[2:]) for t in range(seq_len)]

    # ---- warmup ----
    print(f'\nRunning warmup ({n_warmup} frames)...')
    with torch.no_grad():
        B   = 1
        sys = model.init_sys_emb(B, device)
        for t in range(n_warmup):
            pred_t, sys, _ = model(frames_gt[t], sys_emb=sys, resid=None,
                                   pixel_mask=pixel_mask)
            pred_t = pred_t * pixel_mask
            err_t = (frames_gt[t + 1] - pred_t)
            _, sys, _ = model(frames_gt[t], sys_emb=sys, resid=err_t,
                              pixel_mask=pixel_mask)
        sys_norm = sum(s.norm().item() for s in sys) / len(sys)
        print(f'  sys_emb norm after warmup: {sys_norm:.4f}')

    # ---- autoregressive prediction ----
    print(f'\nPredicting {args.n_predict} frames autoregressively...')
    preds = []
    gt    = []
    x = frames_gt[n_warmup]

    with torch.no_grad():
        for t in range(args.n_predict):
            pred, _, _ = model(x, sys_emb=sys, resid=None, freeze_sys=True,
                               pixel_mask=pixel_mask)
            pred = pred * pixel_mask
            preds.append(pred.cpu())
            gt.append(frames_gt[n_warmup + t].cpu())

            err = frames_gt[n_warmup + t + 1] - pred
            mae = err.abs().mean().item()
            print(f'  t={t+1:3d}  MAE={mae:.5f}')

            # Feed prediction (not ground truth) as next input
            x = pred

    # ---- save outputs ----
    gt_arr   = torch.cat(gt,    dim=0).numpy()    # [T, C, H, W]
    pred_arr = torch.cat(preds, dim=0).numpy()

    gt_phys   = denorm(gt_arr,   mean, std)
    pred_phys = denorm(pred_arr, mean, std) * pixel_mask.cpu().numpy()  # re-zero holes after denorm

    np.save(out_dir / 'frames_gt.npy',        gt_arr)
    np.save(out_dir / 'frames_pred.npy',       pred_arr)
    np.save(out_dir / 'frames_gt_phys.npy',   gt_phys)
    np.save(out_dir / 'frames_pred_phys.npy', pred_phys)
    print(f'\nSaved arrays → {out_dir}')

    # Viewer-compatible output: warmup frames (is_seed=True) + predictions
    print('Saving viewer frames...')
    warmup_phys = denorm(
        torch.cat([frames_gt[t].cpu() for t in range(n_warmup)], dim=0).numpy(),
        mean, std,
    ) * pixel_mask.cpu().numpy()
    all_ts     = timestamps[:n_warmup] + timestamps[n_warmup:n_warmup + args.n_predict]
    save_viewer_frames(
        gt_phys   = warmup_phys,
        pred_phys = pred_phys,
        timestamps = all_ts,
        n_warmup  = n_warmup,
        run_name  = sim_dir.name,
        viewer_dir = out_dir / 'viewer',
    )

    if not args.no_images:
        print('Saving images...')
        save_images(gt_arr, pred_arr, out_dir, mean, std)

    print(f'\nDone.  Viewer output → {out_dir / "viewer"}')


if __name__ == '__main__':
    main()
