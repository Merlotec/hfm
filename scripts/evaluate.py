"""
Consolidated evaluation for the report: one command, all tables and figures.

    python scripts/evaluate.py --checkpoint checkpoints/ckpt-step037500.ckpt \
        --data ../data/fvm_validation --out out/eval

Runs, in order:

  1. rollout      Relative L2 vs horizon, per stride, against the PERSISTENCE
                  baseline; plus a stability count (fraction of rollouts still
                  bounded).  This is the headline accuracy table.
  2. context      Ablations: zero / shuffled / other-run / other-stride context,
                  measured as the shift in output relative to the true-context
                  prediction.  Quantifies how much the model actually uses C.
  3. stride       Does the predicted delta SCALE with the stride encoded in the
                  context?  Reports |pred delta| ratios against |true delta|
                  ratios, and corr(pred delta, true delta).
  4. spectra      Radially-averaged power spectra (GT / prediction / refined) at
                  several rollout depths, with a high-wavenumber energy ratio.
  5. probe        Information content of the context tokens: between-run
                  deviation, effective rank, and the k-token ablation.

Everything is measured on FLUID CELLS ONLY and reported per stride, because a
stride-averaged number hides the conditioning behaviour that matters here.

Add --refine <refiner.pt> to include the flow-matching detail model in the
spectral comparison.  Results are written as JSON (machine-readable, for the
report's numbers) and PNG (figures).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfm import build_model, ContextEncoder
from hfm.data import (build_renderer, load_pixel_mask, FVMSequenceDataset,
                      mesh_dirs_for)

CHANNELS = ['rho', 'u', 'v', 'p']


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_models(ckpt_path: str, device: torch.device, force_flat: bool):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    if force_flat:
        cfg.use_quadtree = False
    model = build_model(cfg)
    model.load_state_dict(ck['model'])
    ce = ContextEncoder(cfg)
    ce.load_state_dict(ck['context_encoder'])
    model.eval().to(device)
    ce.eval().to(device)
    mean = torch.tensor(ck['norm_mean'])
    std = torch.tensor(ck['norm_std'])
    return model, ce, cfg, ck, mean, std


def load_refiner(path: str, device: torch.device):
    from hfm.refiner import RefinerUNet
    rck = torch.load(path, map_location='cpu', weights_only=False)
    net = RefinerUNet(rck['refiner_cfg'])
    net.load_state_dict(rck.get('refiner_ema') or rck['refiner'])
    net.eval().to(device)
    return net, rck['sigma_d'].to(device), rck


def gather_samples(data_dir: Path, cfg, mean, std, device,
                   runs_per_mesh: int, span: int):
    """One long window per run, plus its geometry mask."""
    out = []
    for mi, md in enumerate(mesh_dirs_for(data_dir)):
        rend = build_renderer(md, (cfg.img_size,) * 2, device='cpu')
        pm = load_pixel_mask(md, rend, (cfg.img_size,) * 2).to(device)
        got = 0
        for sim in sorted(p for p in md.iterdir()
                          if p.is_dir() and p.name.startswith('run')):
            if got >= runs_per_mesh:
                break
            try:
                ds = FVMSequenceDataset.with_cache(sim, rend, span, mean, std,
                                                   first_frame=0)
            except Exception:
                continue
            if len(ds) == 0:
                continue
            seq = ds[len(ds) // 2]
            frames = [seq[t:t + 1].to(device) * pm for t in range(seq.shape[0])]
            out.append({'run': f'{md.name}/{sim.name}', 'mesh': mi,
                        'frames': frames, 'mask': pm})
            got += 1
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rel_l2(a, b, mask):
    """Relative L2 over fluid cells: ||a-b|| / ||b||."""
    fl = mask.bool().expand_as(a)
    d = (a - b)[fl]
    r = b[fl]
    return float(d.norm() / r.norm().clamp_min(1e-12))


def mae(a, b, mask):
    fl = mask.bool().expand_as(a)
    return float((a - b)[fl].abs().mean())


def encode_ctx(ce, frames, t0, s, nc, pm):
    """Context from the nc frames ending at t0-s, spaced by stride s."""
    return ce([frames[t0 - s * (nc - i)] for i in range(nc)], pixel_mask=pm)


# ---------------------------------------------------------------------------
# 1. Rollout accuracy + stability
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_rollout(model, ce, cfg, samples, strides, horizon, device):
    nc = cfg.n_context_frames
    res = {}
    for s in strides:
        per_step = [[] for _ in range(horizon)]
        per_step_pers = [[] for _ in range(horizon)]
        diverged = 0
        n = 0
        for smp in samples:
            f, pm = smp['frames'], smp['mask']
            t0 = nc * max(strides)
            if t0 + s * horizon >= len(f):
                continue
            n += 1
            C = encode_ctx(ce, f, t0, s, nc, pm)
            x = f[t0]
            blew_up = False
            for k in range(horizon):
                x = model(x, C, pixel_mask=pm).float() * pm
                tgt = f[t0 + s * (k + 1)]
                if not torch.isfinite(x).all() or x.abs().max() > 1e3:
                    blew_up = True
                    break
                per_step[k].append(rel_l2(x, tgt, pm))
                per_step_pers[k].append(rel_l2(f[t0], tgt, pm))
            if blew_up:
                diverged += 1
        if n == 0:
            continue
        res[s] = {
            'n_runs': n,
            'stable_frac': 1.0 - diverged / n,
            'rel_l2': [float(np.mean(v)) if v else None for v in per_step],
            'rel_l2_persistence': [float(np.mean(v)) if v else None
                                   for v in per_step_pers],
        }
        res[s]['ratio'] = [
            (a / b if (a is not None and b) else None)
            for a, b in zip(res[s]['rel_l2'], res[s]['rel_l2_persistence'])
        ]
    return res


# ---------------------------------------------------------------------------
# 2. Context ablations
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_context(model, ce, cfg, samples, strides, device):
    nc = cfg.n_context_frames
    t0 = nc * max(strides)
    # pre-encode every (run, stride) context
    ctx = {}
    for i, smp in enumerate(samples):
        if t0 + max(strides) >= len(smp['frames']):
            continue
        for s in strides:
            ctx[(i, s)] = encode_ctx(ce, smp['frames'], t0, s, nc, smp['mask'])

    res = {}
    for s in strides:
        acc = {}
        for i, smp in enumerate(samples):
            if (i, s) not in ctx:
                continue
            f, pm = smp['frames'], smp['mask']
            x, tgt = f[t0], f[t0 + s]
            C = ctx[(i, s)]
            p_true = model(x, C, pixel_mask=pm).float() * pm
            other_run = next((ctx[(j, s)] for j in range(len(samples))
                              if j != i and (j, s) in ctx), None)
            other_str = next((ctx[(i, s2)] for s2 in strides
                              if s2 != s and (i, s2) in ctx), None)
            variants = {'zero': torch.zeros_like(C),
                        'shuffled': C[:, torch.randperm(C.shape[1])]}
            if other_run is not None:
                variants['other_run'] = other_run
            if other_str is not None:
                variants['other_stride'] = other_str
            delta = mae(x, tgt, pm)          # size of the true one-step change
            acc.setdefault('true', []).append((mae(p_true, tgt, pm), 0.0, delta))
            for k, Cv in variants.items():
                p = model(x, Cv, pixel_mask=pm).float() * pm
                acc.setdefault(k, []).append(
                    (mae(p, tgt, pm), mae(p, p_true, pm), delta))
        res[s] = {k: {'err': float(np.mean([a for a, _, _ in v])),
                      'shift': float(np.mean([b for _, b, _ in v])),
                      'shift_frac_of_delta': float(
                          np.mean([b for _, b, _ in v]) /
                          max(np.mean([c for _, _, c in v]), 1e-12))}
                  for k, v in acc.items()}
    return res


# ---------------------------------------------------------------------------
# 3. Stride / Delta-t conditioning
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_stride(model, ce, cfg, samples, strides, device):
    nc = cfg.n_context_frames
    t0 = nc * max(strides)
    agg = {s: {'dp': [], 'dt': [], 'corr': [], 'err': []} for s in strides}
    for smp in samples:
        f, pm = smp['frames'], smp['mask']
        if t0 + max(strides) >= len(f):
            continue
        x = f[t0]
        fl = pm.bool().expand_as(x)
        for s in strides:
            C = encode_ctx(ce, f, t0, s, nc, pm)
            pred = model(x, C, pixel_mask=pm).float() * pm
            dp, dt = (pred - x)[fl], (f[t0 + s] - x)[fl]
            a, b = dp - dp.mean(), dt - dt.mean()
            agg[s]['dp'].append(float(dp.abs().mean()))
            agg[s]['dt'].append(float(dt.abs().mean()))
            agg[s]['corr'].append(float((a * b).sum() /
                                        (a.norm() * b.norm()).clamp_min(1e-12)))
            agg[s]['err'].append(rel_l2(pred, f[t0 + s], pm))
    out = {}
    base = strides[0]
    mb_p = float(np.mean(agg[base]['dp'])) if agg[base]['dp'] else None
    mb_t = float(np.mean(agg[base]['dt'])) if agg[base]['dt'] else None
    for s in strides:
        if not agg[s]['dp']:
            continue
        mp, mt = float(np.mean(agg[s]['dp'])), float(np.mean(agg[s]['dt']))
        out[s] = {'pred_delta': mp, 'true_delta': mt,
                  'corr_delta': float(np.mean(agg[s]['corr'])),
                  'rel_l2': float(np.mean(agg[s]['err'])),
                  'pred_scaling': (mp / mb_p) if mb_p else None,
                  'true_scaling': (mt / mb_t) if mb_t else None}
        if out[s]['pred_scaling'] and out[s]['true_scaling']:
            out[s]['scaling_accuracy'] = out[s]['pred_scaling'] / out[s]['true_scaling']
    return out


# ---------------------------------------------------------------------------
# 4. Spectra
# ---------------------------------------------------------------------------

def radial_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x [C,H,W] -> (k bins, power [C, nbins]) radially averaged."""
    C, H, W = x.shape
    P = np.abs(np.fft.fftshift(np.fft.fft2(x), axes=(-2, -1))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(H)) * H
    kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    K = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    kmax = int(min(H, W) // 2)
    idx = np.clip(K.astype(int), 0, kmax)
    counts = np.bincount(idx.ravel(), minlength=kmax + 1)
    out = np.stack([np.bincount(idx.ravel(), weights=P[c].ravel(),
                                minlength=kmax + 1) / np.maximum(counts, 1)
                    for c in range(C)])
    return np.arange(kmax + 1), out


@torch.no_grad()
def eval_spectra(model, ce, cfg, samples, stride, depths, device,
                 refiner=None, sigma_d=None, n_refine=6):
    from hfm.refiner import sample_detail
    nc = cfg.n_context_frames
    t0 = nc * 4
    acc = {d: {'gt': [], 'pred': [], 'refined': []} for d in depths}
    for smp in samples:
        f, pm = smp['frames'], smp['mask']
        if t0 + stride * max(depths) >= len(f):
            continue
        C = encode_ctx(ce, f, t0, stride, nc, pm)
        ctx_vec = C.float().mean(dim=1)
        x = f[t0]
        for k in range(1, max(depths) + 1):
            x_prev = x
            x = model(x, C, pixel_mask=pm).float() * pm
            if k in depths:
                acc[k]['gt'].append(f[t0 + stride * k][0].cpu().numpy())
                acc[k]['pred'].append(x[0].cpu().numpy())
                if refiner is not None:
                    d_hat = sample_detail(refiner, x, x_prev, pm, ctx_vec,
                                          sigma_d, n_steps=n_refine)
                    acc[k]['refined'].append(((x + d_hat) * pm)[0].cpu().numpy())
    res = {}
    for d, v in acc.items():
        if not v['gt']:
            continue
        entry = {}
        spec = {}
        for key in ('gt', 'pred', 'refined'):
            if not v[key]:
                continue
            bins, P = radial_spectrum(np.mean(np.stack(v[key]), axis=0))
            spec[key] = (bins, P)
        entry['bins'] = spec['gt'][0].tolist()
        for key, (_, P) in spec.items():
            entry[f'power_{key}'] = P.mean(axis=0).tolist()
        # high-wavenumber energy ratio vs GT
        bins = spec['gt'][0]
        hi = bins > 32
        e_gt = spec['gt'][1][:, hi].sum()
        for key in ('pred', 'refined'):
            if key in spec:
                entry[f'hf_ratio_{key}'] = float(spec[key][1][:, hi].sum() / e_gt)
        res[d] = entry
    return res


# ---------------------------------------------------------------------------
# 5. Context information content
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_probe(model, ce, cfg, samples, strides, device):
    nc = cfg.n_context_frames
    t0 = nc * max(strides)
    Cs, evals = [], []
    for smp in samples:
        f, pm = smp['frames'], smp['mask']
        if t0 + strides[0] >= len(f):
            continue
        C = encode_ctx(ce, f, t0, strides[0], nc, pm)
        Cs.append(C[0])
        evals.append((C, f[t0], f[t0 + strides[0]], pm))
    if len(Cs) < 2:
        return {}
    A = torch.stack(Cs).float().cpu()
    dev_frac = float((A - A.mean(0)).abs().mean() / A.abs().mean())

    def eff_rank(M):
        sv = torch.linalg.svdvals(M)
        p = sv / sv.sum()
        return float(torch.exp(-(p * (p + 1e-12).log()).sum()))

    out = {'n_samples': len(Cs),
           'between_run_deviation': dev_frac,
           'effective_rank': float(np.mean([eff_rank(c) for c in A])),
           'n_tokens': int(A.shape[1])}
    # k-token ablation: keep only k of the K context tokens
    K = A.shape[1]
    tok = {}
    for k in [K, K // 2, K // 4, K // 8, 1]:
        errs, pers = [], []
        for C, x, tgt, pm in evals:
            idx = torch.randperm(K)[:k]
            p = model(x, C[:, idx], pixel_mask=pm).float() * pm
            errs.append(rel_l2(p, tgt, pm))
            pers.append(rel_l2(x, tgt, pm))
        tok[k] = {'rel_l2': float(np.mean(errs)),
                  'ratio_vs_persistence': float(np.mean(errs) / np.mean(pers))}
    out['token_ablation'] = tok
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(R, strides):
    def hdr(t):
        print(f'\n{"=" * 74}\n{t}\n{"=" * 74}')

    if 'rollout' in R and R['rollout']:
        hdr('1. ROLLOUT ACCURACY  (relative L2 over fluid cells)')
        for s in strides:
            r = R['rollout'].get(str(s)) or R['rollout'].get(s)
            if not r:
                continue
            print(f'\n  stride {s}   ({r["n_runs"]} runs, '
                  f'{100 * r["stable_frac"]:.0f}% stable)')
            print(f'    {"step":>6}{"model":>12}{"persistence":>14}{"ratio":>9}')
            for k, (a, b, c) in enumerate(zip(r['rel_l2'], r['rel_l2_persistence'],
                                              r['ratio']), 1):
                if a is None:
                    continue
                if k <= 5 or k % 5 == 0:
                    print(f'    {k:>6}{a:>12.4f}{b:>14.4f}{c:>9.3f}')

    if 'context' in R and R['context']:
        hdr('2. CONTEXT ABLATION  (shift from true-context prediction)')
        print(f'    {"stride":>7}{"variant":>15}{"err":>10}{"shift":>10}'
              f'{"% of delta":>12}')
        for s in strides:
            c = R['context'].get(str(s)) or R['context'].get(s)
            if not c:
                continue
            for k in ('true', 'zero', 'shuffled', 'other_run', 'other_stride'):
                if k in c:
                    print(f'    {s:>7}{k:>15}{c[k]["err"]:>10.5f}'
                          f'{c[k]["shift"]:>10.5f}'
                          f'{100 * c[k]["shift_frac_of_delta"]:>11.1f}%')

    if 'stride' in R and R['stride']:
        hdr('3. TIMESTEP CONDITIONING  (does the delta scale with the context?)')
        print(f'    {"s":>4}{"|pred d|":>11}{"|true d|":>11}{"corr":>9}'
              f'{"pred scal":>11}{"true scal":>11}{"accuracy":>10}')
        for s in strides:
            v = R['stride'].get(str(s)) or R['stride'].get(s)
            if not v:
                continue
            sa = v.get('scaling_accuracy')
            ps, ts = v.get('pred_scaling'), v.get('true_scaling')
            print(f'    {s:>4}{v["pred_delta"]:>11.5f}{v["true_delta"]:>11.5f}'
                  f'{v["corr_delta"]:>+9.3f}'
                  f'{(f"{ps:.2f}x" if ps else "-"):>11}'
                  f'{(f"{ts:.2f}x" if ts else "-"):>11}'
                  f'{(f"{sa:.3f}" if sa else "-"):>10}')

    if 'spectra' in R and R['spectra']:
        hdr('4. SPECTRAL CONTENT  (high-k energy relative to ground truth)')
        print(f'    {"depth":>7}{"pred/gt":>12}{"refined/gt":>14}')
        for d in sorted(R['spectra'], key=lambda x: int(x)):
            e = R['spectra'][d]
            pr = e.get('hf_ratio_pred')
            rf = e.get('hf_ratio_refined')
            print(f'    {d:>7}{pr:>12.3f}'
                  f'{(f"{rf:.3f}" if rf is not None else "-"):>14}')
        print('    (1.0 = matches GT;  < 1 = blurred;  > 1 = hallucinated energy)')

    if 'probe' in R and R['probe']:
        hdr('5. CONTEXT INFORMATION CONTENT')
        p = R['probe']
        print(f'    samples                    {p["n_samples"]}')
        print(f'    between-run deviation      '
              f'{100 * p["between_run_deviation"]:.1f}% of magnitude'
              f'   (~0% = collapsed)')
        print(f'    effective rank             {p["effective_rank"]:.1f}'
              f' / {p["n_tokens"]} tokens')
        print(f'\n    k-token ablation:  {"k":>6}{"rel L2":>10}'
              f'{"vs persistence":>17}')
        for k in sorted(p['token_ablation'], key=lambda x: -int(x)):
            t = p['token_ablation'][k]
            print(f'                       {k:>6}{t["rel_l2"]:>10.4f}'
                  f'{t["ratio_vs_persistence"]:>17.3f}')


def make_figures(R, out_dir: Path, strides):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n[warn] matplotlib unavailable — skipping figures')
        return

    if R.get('rollout'):
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in strides:
            r = R['rollout'].get(str(s)) or R['rollout'].get(s)
            if not r:
                continue
            y = [v for v in r['rel_l2'] if v is not None]
            p = [v for v in r['rel_l2_persistence'] if v is not None]
            ax.plot(range(1, len(y) + 1), y, label=f'model  s={s}')
            ax.plot(range(1, len(p) + 1), p, ls='--', alpha=0.5,
                    label=f'persistence  s={s}')
        ax.set_xlabel('rollout step'); ax.set_ylabel('relative $L_2$')
        ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / 'rollout_error.png', dpi=140)
        plt.close(fig)

    if R.get('spectra'):
        depths = sorted(R['spectra'], key=lambda x: int(x))
        fig, axes = plt.subplots(1, len(depths), figsize=(4.5 * len(depths), 3.8),
                                 squeeze=False)
        for ax, d in zip(axes[0], depths):
            e = R['spectra'][d]
            b = np.array(e['bins'][1:])
            for key, lab in (('gt', 'ground truth'), ('pred', 'prediction'),
                             ('refined', 'refined')):
                k = f'power_{key}'
                if k in e:
                    ax.loglog(b, np.array(e[k][1:]), label=lab, lw=1.4)
            ax.set_title(f'rollout step {d}', fontsize=10)
            ax.set_xlabel('wavenumber $k$'); ax.grid(alpha=0.3)
        axes[0][0].set_ylabel('radial power'); axes[0][0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / 'spectra.png', dpi=140)
        plt.close(fig)
    print(f'\nFigures → {out_dir}')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('out/eval'))
    ap.add_argument('--refine', type=str, default=None,
                    help='refiner checkpoint, for the spectral comparison')
    ap.add_argument('--refine-steps', type=int, default=6)
    ap.add_argument('--runs-per-mesh', type=int, default=3)
    ap.add_argument('--horizon', type=int, default=30)
    ap.add_argument('--spectra-depths', type=int, nargs='+', default=[1, 10, 30])
    ap.add_argument('--no-residual', action='store_true')
    ap.add_argument('--flat', action='store_true',
                    help='force use_quadtree=False')
    ap.add_argument('--skip', nargs='*', default=[],
                    choices=['rollout', 'context', 'stride', 'spectra', 'probe'])
    args = ap.parse_args()

    from hfm.distributed import pick_device
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=True)

    model, ce, cfg, ck, mean, std = load_models(args.checkpoint, device, args.flat)
    if args.no_residual:
        cfg.residual_prediction = False
        model, ce, cfg, ck, mean, std = load_models(args.checkpoint, device, args.flat)
    strides = list(getattr(cfg, 'time_strides', (1,)) or (1,))
    nc = cfg.n_context_frames
    print(f'checkpoint  {args.checkpoint}  (step {ck.get("global_step")})')
    print(f'model       {type(model).__name__}  residual='
          f'{getattr(model, "residual_prediction", "?")}  strides={strides}')
    print(f'device      {device}')

    refiner = sigma_d = None
    if args.refine:
        refiner, sigma_d, rck = load_refiner(args.refine, device)
        base_step = rck.get('base_global_step')
        if base_step != ck.get('global_step'):
            print(f'  [WARN] refiner was trained against base step {base_step}, '
                  f'but this checkpoint is step {ck.get("global_step")}')
        print(f'refiner     {args.refine}')

    span = nc * max(strides) + max(strides) * max(args.horizon,
                                                  max(args.spectra_depths)) + 2
    print(f'\nloading data from {args.data} (window {span} frames)...')
    samples = gather_samples(args.data, cfg, mean, std, device,
                             args.runs_per_mesh, span)
    print(f'{len(samples)} runs across '
          f'{len(set(s["mesh"] for s in samples))} geometries')
    if not samples:
        sys.exit('no usable runs found')

    R = {'checkpoint': str(args.checkpoint), 'data': str(args.data),
         'base_step': ck.get('global_step'), 'strides': strides,
         'n_runs': len(samples)}

    if 'rollout' not in args.skip:
        print('\n[1/5] rollout ...')
        R['rollout'] = eval_rollout(model, ce, cfg, samples, strides,
                                    args.horizon, device)
    if 'context' not in args.skip:
        print('[2/5] context ablation ...')
        R['context'] = eval_context(model, ce, cfg, samples, strides, device)
    if 'stride' not in args.skip:
        print('[3/5] stride conditioning ...')
        R['stride'] = eval_stride(model, ce, cfg, samples, strides, device)
    if 'spectra' not in args.skip:
        print('[4/5] spectra ...')
        R['spectra'] = eval_spectra(model, ce, cfg, samples, strides[0],
                                    args.spectra_depths, device,
                                    refiner, sigma_d, args.refine_steps)
    if 'probe' not in args.skip:
        print('[5/5] context information ...')
        R['probe'] = eval_probe(model, ce, cfg, samples, strides, device)

    print_report(R, strides)
    with open(args.out / 'results.json', 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f'\nJSON → {args.out / "results.json"}')
    make_figures(R, args.out, strides)


if __name__ == '__main__':
    main()
