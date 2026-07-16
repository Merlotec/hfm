from typing import Optional, Union
"""
HFM inference script.

Loads a GANTrainer checkpoint, feeds n_context_frames ground-truth frames into
the ContextEncoder, then autoregressively predicts subsequent frames with HFM.

Outputs are saved to --out-dir as:
  - frames_gt.npy / frames_pred.npy       — normalised arrays [T, C, H, W]
  - frames_gt_phys.npy / frames_pred_phys.npy — denormalised physical values
  - images/t{NNN}_ch{C}.png              — ground-truth vs prediction per channel

Usage:
    python infer.py --checkpoint checkpoints/train_step010000.pt \\
                    --data-dir   /path/to/sim_dataset \\
                    --out-dir    out/infer
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hfm import HFM, ContextEncoder
from hfm.config import HFMConfig
from hfm.data import build_renderer, FVMSequenceDataset, load_pixel_mask
from hfm.discriminator import HFMDiscriminator

DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'
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
    raise FileNotFoundError('No normalisation stats found.')


def denorm(frames: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """[T, C, H, W] normalised → physical units."""
    m = mean.numpy()[None, :, None, None]
    s = std.numpy()[None, :, None, None]
    return frames * s + m


def save_viewer_frames(
    gt_phys: np.ndarray,
    pred_phys: np.ndarray,
    timestamps: list[float],
    n_context: int,
    run_name: str,
    viewer_dir: Path,
):
    """Write t_*.npz files in the format expected by fvm_viewer/viewer.py -c."""
    run_dir = viewer_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    T = len(timestamps)
    for i, ts in enumerate(timestamps):
        if i < n_context:
            grid    = gt_phys[i].astype(np.float32)
            is_seed = True
        else:
            grid    = pred_phys[i - n_context].astype(np.float32)
            is_seed = False
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

            axes[0].imshow(gt_phys[t, c],   vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[1].imshow(pred_phys[t, c], vmin=vmin, vmax=vmax, cmap='RdBu_r')
            err = np.abs(gt_phys[t, c] - pred_phys[t, c])
            axes[2].imshow(err, cmap='hot')

            axes[0].set_title(f'Ground truth — {CHANNEL_NAMES[c]}')
            axes[1].set_title(f'Prediction   — {CHANNEL_NAMES[c]}')
            axes[2].set_title(f'|Error| max={err.max():.3f}')
            for ax in axes:
                ax.axis('off')

            plt.suptitle(f't={t}  channel={CHANNEL_NAMES[c]}', fontsize=12)
            plt.tight_layout()
            plt.savefig(img_dir / f't{t:03d}_ch{c}_{CHANNEL_NAMES[c]}.png',
                        dpi=100, bbox_inches='tight')
            plt.close()

    print(f'  Saved {T * C} images → {img_dir}')


# ---------------------------------------------------------------------------
# Discriminator saliency
# ---------------------------------------------------------------------------

def disc_saliency(
    discriminator: HFMDiscriminator,
    frame: torch.Tensor,
    x_in: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient of discriminator logit w.r.t. input pixels → [1, H, W] saliency map.
    High values = pixels the discriminator is most sensitive to.
    """
    inp = frame.detach().float().requires_grad_(True)
    logit = discriminator(inp, x_in.detach(), context.detach())
    logit.mean().backward()
    assert inp.grad is not None
    return inp.grad.abs().mean(dim=1, keepdim=True)  # [1, 1, H, W]


def save_disc_saliency(
    saliency_fake: list[torch.Tensor],
    saliency_real: list[torch.Tensor],
    pred_phys: np.ndarray,
    gt_phys: np.ndarray,
    out_dir: Path,
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available — skipping saliency output')
        return

    sal_dir = out_dir / 'disc_saliency'
    sal_dir.mkdir(exist_ok=True)

    for t, (sf, sr) in enumerate(zip(saliency_fake, saliency_real)):
        sf_np = sf.squeeze().cpu().numpy()
        sr_np = sr.squeeze().cpu().numpy()

        # Normalise each map to [0, 1] for display
        def _norm(x: np.ndarray) -> np.ndarray:
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        # Show channel 1 (u-velocity) as representative frame
        axes[0].imshow(pred_phys[t, 1], cmap='RdBu_r')
        axes[0].set_title(f'Prediction (u)  t={t}')

        axes[1].imshow(gt_phys[t, 1], cmap='RdBu_r')
        axes[1].set_title(f'Ground truth (u)  t={t}')

        axes[2].imshow(_norm(sf_np), cmap='hot')
        axes[2].set_title('Disc saliency — fake\n(where it sees artefacts)')

        axes[3].imshow(_norm(sr_np), cmap='hot')
        axes[3].set_title('Disc saliency — real\n(where it sees "real")')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(sal_dir / f't{t:03d}_disc_saliency.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f'  Saved {len(saliency_fake)} saliency maps → {sal_dir}')


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
    parser.add_argument('--n-context',  type=int, default=None,
                        help='Context frames fed to ContextEncoder (defaults to cfg.n_context_frames)')
    parser.add_argument('--n-predict',  type=int, default=10,
                        help='Number of frames to predict autoregressively')
    parser.add_argument('--seq-start',  type=int, default=None,
                        help='Frame index to start from (defaults to middle of run)')
    parser.add_argument('--first-frame', type=int, default=20,
                        help='Skip this many initial transient frames')
    parser.add_argument('--no-images',    action='store_true',
                        help='Skip PNG generation')
    parser.add_argument('--disc-saliency', action='store_true',
                        help='Compute and save discriminator saliency heatmaps')
    args = parser.parse_args()

    device   = get_device()
    out_dir  = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device:  {device}')
    print(f'Out dir: {out_dir}')

    # ---- load checkpoint ----
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg: HFMConfig = ckpt['cfg']
    step = ckpt.get('global_step', '?')
    print(f'  step={step}')

    model = HFM(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if unexpected:
        print(f'  [warn] model unexpected keys: {unexpected}')
    if missing:
        print(f'  New/missing model keys (random init): {missing}')
    model.eval()

    context_encoder = ContextEncoder(cfg).to(device)
    if 'context_encoder' in ckpt:
        context_encoder.load_state_dict(ckpt['context_encoder'])
    else:
        print('  [warn] no context_encoder in checkpoint; using random weights')
    context_encoder.eval()

    discriminator: Optional[HFMDiscriminator] = None
    if args.disc_saliency:
        if 'discriminator' in ckpt:
            discriminator = HFMDiscriminator(cfg).to(device)
            discriminator.load_state_dict(ckpt['discriminator'], strict=False)
            discriminator.eval()
            print('  Discriminator loaded for saliency.')
        else:
            print('  [warn] no discriminator in checkpoint — skipping saliency')

    n_context = args.n_context if args.n_context is not None else cfg.n_context_frames

    # ---- load data ----
    print(f'\nLoading data from {data_dir}')
    mean, std = load_stats(data_dir)
    renderer_cache = {}
    
    mesh_dirs = []
    if (data_dir / 'shared_mesh.pkl').exists():
        mesh_dirs.append(data_dir)
    else:
        for p in data_dir.iterdir():
            if p.is_dir() and (p / 'shared_mesh.pkl').exists():
                mesh_dirs.append(p)

    if not mesh_dirs:
        raise RuntimeError(f'No shared_mesh.pkl found in {data_dir} or its subdirectories')

    runs = []
    for mdir in mesh_dirs:
        renderer = build_renderer(mdir, (cfg.img_size, cfg.img_size), device='cpu')
        pixel_mask = load_pixel_mask(mdir, renderer, (cfg.img_size, cfg.img_size)).to(device)
        sim_dirs = sorted([p for p in mdir.iterdir() if p.is_dir() and p.name.startswith('run')])
        for sdir in sim_dirs:
            runs.append((sdir, renderer, pixel_mask))

    if not runs:
        raise RuntimeError(f'No simulation subdirectories found in {data_dir}')
    print(f'  Found {len(runs)} simulation runs across {len(mesh_dirs)} meshes\n')

    seq_len = n_context + args.n_predict + 1

    for sim_idx, (sim_dir, renderer, pixel_mask) in enumerate(runs):
        print(f'[{sim_idx+1}/{len(runs)}] {sim_dir.name}')
        run_out = out_dir / sim_dir.name
        run_out.mkdir(parents=True, exist_ok=True)

        ds = FVMSequenceDataset.with_cache(
            sim_dir, renderer, seq_len, mean, std, first_frame=args.first_frame
        )
        if len(ds) == 0:
            print(f'  [skip] no sequences available')
            continue

        start = args.seq_start if args.seq_start is not None else len(ds) // 2
        seq   = ds[start]
        frames_gt = [seq[t:t+1].to(device) * pixel_mask for t in range(seq_len)]
        timestamps = [float(ds.paths[start + t].stem[2:]) for t in range(seq_len)]

        # ---- encode context ----
        with torch.no_grad():
            context = context_encoder(frames_gt[:n_context], pixel_mask=pixel_mask)

        # ---- autoregressive prediction ----
        preds          = []
        gt             = []
        saliency_fakes = []
        saliency_reals = []
        x = frames_gt[n_context]

        with torch.no_grad():
            for t in range(args.n_predict):
                pred = model(x, context, pixel_mask=pixel_mask)
                pred = pred.float() * pixel_mask
                preds.append(pred.cpu())
                gt.append(frames_gt[n_context + t].cpu())

                if t + 1 < args.n_predict:
                    mae = (frames_gt[n_context + t + 1] - pred).abs().mean().item()
                    print(f'  t={t+1:3d}  MAE={mae:.5f}')

                if discriminator is not None:
                    x_in_t = frames_gt[n_context + t]
                    saliency_fakes.append(disc_saliency(discriminator, pred,   x_in_t, context))
                    saliency_reals.append(disc_saliency(discriminator, x_in_t, x_in_t, context))

                x = pred

        # ---- save outputs ----
        gt_arr   = torch.cat(gt,    dim=0).numpy()
        pred_arr = torch.cat(preds, dim=0).numpy()
        gt_phys   = denorm(gt_arr,   mean, std)
        pred_phys = denorm(pred_arr, mean, std) * pixel_mask.cpu().numpy()

        np.save(run_out / 'frames_gt.npy',       gt_arr)
        np.save(run_out / 'frames_pred.npy',      pred_arr)
        np.save(run_out / 'frames_gt_phys.npy',  gt_phys)
        np.save(run_out / 'frames_pred_phys.npy', pred_phys)

        context_phys = denorm(
            torch.cat([frames_gt[t].cpu() for t in range(n_context)], dim=0).numpy(),
            mean, std,
        ) * pixel_mask.cpu().numpy()
        all_ts = timestamps[:n_context] + timestamps[n_context:n_context + args.n_predict]
        save_viewer_frames(
            gt_phys    = context_phys,
            pred_phys  = pred_phys,
            timestamps = all_ts,
            n_context  = n_context,
            run_name   = sim_dir.name,
            viewer_dir = out_dir / 'viewer',
        )

        if not args.no_images:
            save_images(gt_arr, pred_arr, run_out, mean, std)

        if discriminator is not None and saliency_fakes:
            save_disc_saliency(saliency_fakes, saliency_reals, pred_phys, gt_phys, run_out)

        print(f'  Saved → {run_out}')

    print(f'\nDone.  Viewer output → {out_dir / "viewer"}')


if __name__ == '__main__':
    main()
