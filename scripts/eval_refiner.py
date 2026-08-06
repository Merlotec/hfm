"""
Did the refiner actually sharpen?  Spectral comparison of GT vs base prediction
vs refined prediction, from the arrays infer.py saves per run:

    python scripts/eval_refiner.py out/infer/run_0000_xxx [more run dirs...]

Success criteria:
  * radial power spectrum: pred falls below GT at high wavenumber (the MSE
    blur); refined closes that gap toward GT WITHOUT overshooting it
    (overshoot = hallucinated energy);
  * high-frequency energy ratio E_{k>32}(x)/E_{k>32}(gt): pred << 1 -> refined ~ 1;
  * masked MSE: refined is expected to be slightly WORSE than pred (a correct
    sampler gives up the posterior-mean advantage); a blow-up means a
    mis-scaled sigma_d or diverging sampler.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

CHANNELS = ['rho', 'u', 'v', 'p']


def radial_spectrum(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """frames [T, C, H, W] -> (k_bins, power[C, n_bins]) radially averaged."""
    T, C, H, W = frames.shape
    F = np.fft.fftshift(np.abs(np.fft.fft2(frames)) ** 2, axes=(-2, -1))
    ky = np.fft.fftshift(np.fft.fftfreq(H)) * H
    kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    K = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    k_max = int(min(H, W) // 2)
    bins = np.arange(k_max + 1)
    idx = np.clip(K.astype(int), 0, k_max)
    power = np.zeros((C, k_max + 1))
    counts = np.bincount(idx.ravel(), minlength=k_max + 1)
    for c in range(C):
        p = F[:, c].mean(axis=0)
        power[c] = np.bincount(idx.ravel(), weights=p.ravel(),
                               minlength=k_max + 1) / np.maximum(counts, 1)
    return bins, power


def hf_energy(frames: np.ndarray, k_cut: int = 32) -> np.ndarray:
    """Per-channel energy above wavenumber k_cut."""
    bins, power = radial_spectrum(frames)
    return power[:, bins > k_cut].sum(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dirs', nargs='+', type=Path)
    p.add_argument('--k-cut', type=int, default=32)
    p.add_argument('--plot', type=Path, default=None,
                   help='Save a spectra PNG here')
    args = p.parse_args()

    agg = {}
    for rd in args.run_dirs:
        gt = np.load(rd / 'frames_gt.npy')
        pred = np.load(rd / 'frames_pred.npy')
        refined_path = rd / 'frames_pred_refined.npy'
        refined = np.load(refined_path) if refined_path.exists() else None
        if refined is None:
            print(f'[skip] {rd}: no frames_pred_refined.npy (run infer.py --refine)')
            continue

        # gt array in infer.py is offset by one frame vs pred (pred[t] predicts
        # gt[t+1]); align by dropping the edges.
        gt_al, pred_al, ref_al = gt[1:], pred[:-1], refined[:-1]

        for name, arr in [('gt', gt_al), ('pred', pred_al), ('refined', ref_al)]:
            agg.setdefault(name, []).append(arr)

        mse_p = float(((pred_al - gt_al) ** 2).mean())
        mse_r = float(((ref_al - gt_al) ** 2).mean())
        hf_g, hf_p, hf_r = (hf_energy(a, args.k_cut) for a in (gt_al, pred_al, ref_al))
        print(f'{rd.name}:')
        print(f'  MSE      pred={mse_p:.5f}  refined={mse_r:.5f}  '
              f'(+{100 * (mse_r - mse_p) / max(mse_p, 1e-12):.1f}% — slight increase is healthy)')
        with np.errstate(divide="ignore", invalid="ignore"):
            print(f'  HF ratio (k>{args.k_cut}) pred/gt='
                  f'{np.array2string(hf_p / hf_g, precision=3)}  '
                  f'refined/gt={np.array2string(hf_r / hf_g, precision=3)}  (want -> 1, not > 1)')

    if not agg:
        sys.exit(1)

    gt_all = np.concatenate(agg['gt'])
    pred_all = np.concatenate(agg['pred'])
    ref_all = np.concatenate(agg['refined'])
    bins, P_gt = radial_spectrum(gt_all)
    _, P_pr = radial_spectrum(pred_all)
    _, P_rf = radial_spectrum(ref_all)

    print('\naggregate spectral gap closure (mean over channels, log-power):')
    for lo, hi in [(8, 16), (16, 32), (32, 64), (64, 128)]:
        sel = (bins >= lo) & (bins < hi)
        gap_p = float(np.log10(P_pr[:, sel].mean() / P_gt[:, sel].mean()))
        gap_r = float(np.log10(P_rf[:, sel].mean() / P_gt[:, sel].mean()))
        print(f'  k in [{lo:3d},{hi:3d}): pred {gap_p:+.2f} dex  refined {gap_r:+.2f} dex'
              f'   (0 = matches GT, negative = blurred, positive = overshoot)')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        for c, ax in enumerate(axes):
            ax.loglog(bins[1:], P_gt[c, 1:], label='GT', lw=1.6)
            ax.loglog(bins[1:], P_pr[c, 1:], label='pred', lw=1.2)
            ax.loglog(bins[1:], P_rf[c, 1:], label='refined', lw=1.2)
            ax.set_title(CHANNELS[c] if c < len(CHANNELS) else f'ch{c}')
            ax.set_xlabel('wavenumber k')
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('radial power')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f'\nspectra plot -> {args.plot}')


if __name__ == '__main__':
    main()
