"""Audit the train/val split on a real dataset — run this ON the training machine.

Answers "can any validation data have been trained on?" with evidence, at three
levels of strictness:

  1. MECHANISM   — the split is one decision per run dir, so train/val run sets are
                   disjoint by construction.  Verified here, plus a check for
                   DUPLICATE run names (two dirs with the same basename hash to the
                   same side, but a name collision would mean the corpus has copies).
  2. CONTENT     — hashes the actual frame arrays and reports any val frame that is
                   byte-identical to a train frame.  This is the definitive test: it
                   catches duplicated/copied runs regardless of naming.
  3. CORRELATION — runs that are not identical but are not independent either:
                     * alternating segments of ONE trajectory (run_aNNN_sMM_*): a
                       later segment CONTINUES an earlier one, so a val segment whose
                       sibling is in train starts from a state the model has seen;
                     * grid runs sharing an initial condition (run_cNNN_iMMM_*): the
                       same IC recurs under every context by design, so a val run's
                       t=0 frame can be identical to a train run's;
                     * mesh geometry present on both sides (expected unless you
                       intended to hold out geometry).
                   These are reported as counts, not failures — whether they matter
                   depends on the claim you want the val number to support.

Usage (on Dawn):
    python scripts/check_val_split.py --data-dir /path/to/dataset --val-fraction 0.05
    python scripts/check_val_split.py --data-dir A --data-dir B --deep    # all frames

--deep hashes EVERY frame (slow, exact).  Default hashes each run's first and
middle frame, which already catches whole-run duplication.
"""

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hfm.data import is_val_run          # the real one training uses


def find_runs(roots):
    """Enumerate runs exactly as FVMDataModule.setup does: mesh dirs (a dir holding
    shared_mesh.pkl, or its subdirs), then every 'run*' subdirectory."""
    runs = []
    for root in roots:
        root = Path(root)
        mesh_dirs = []
        if (root / 'shared_mesh.pkl').exists():
            mesh_dirs.append(root)
        else:
            mesh_dirs += [p for p in sorted(root.iterdir())
                          if p.is_dir() and (p / 'shared_mesh.pkl').exists()]
        if not mesh_dirs:
            print(f'  [warn] no shared_mesh.pkl under {root}; treating subdirs as runs')
            mesh_dirs = [root]
        for mdir in mesh_dirs:
            runs += [p for p in sorted(mdir.iterdir())
                     if p.is_dir() and p.name.startswith('run')]
    return runs


def frame_hashes(run: Path, deep: bool):
    """Content hashes of a run's frames, as (hash, filename, is_uniform).

    Hashes the PHYSICAL field, not the stored array.  Frames are saved normalised
    per-frame (value - prim_mean) / prim_std, and a uniform field has prim_std = 0
    — so every run's t=0 frame stores the same degenerate array even when the
    physical states differ (rho=1.90 vs 1.81).  Hashing the stored array reports
    those as identical; de-normalising first compares what the run actually holds.

    `is_uniform` marks zero-variance frames: two runs sharing an initial condition
    legitimately match there, which is not evidence of duplication.
    """
    files = sorted([f for f in run.iterdir()
                    if f.name.startswith('t_') and f.name.endswith('.npz')],
                   key=lambda f: float(f.stem[2:]))
    if not files:
        return []
    picks = files if deep else [files[0]] + ([files[len(files) // 2]]
                                             if len(files) > 1 else [])
    out = []
    for f in picks:
        try:
            d = np.load(f)
            if 'cell_primatives' in d.files:                  # solver output
                arr = d['cell_primatives'].astype(np.float64)
                if 'prim_std' in d.files and 'prim_mean' in d.files:
                    arr = arr * d['prim_std'] + d['prim_mean']
            elif 'grid' in d.files:                           # viewer/pred format
                arr = d['grid'].astype(np.float64)
            else:
                continue
            arr = np.ascontiguousarray(np.round(arr, 6))      # kill fp16 round-trip noise
            uniform = bool(np.all(arr.std(axis=0) == 0)) if arr.ndim == 2 else False
            out.append((hashlib.md5(arr.tobytes()).hexdigest(), f.name, uniform))
        except Exception as e:
            print(f'  [warn] unreadable {f}: {type(e).__name__}')
    return out


def group_keys(run: Path):
    """Correlation keys for a run: (trajectory, initial-condition, mesh).

    Names come from the generators:
      alternating  run_a{traj}_s{seg}_{uid}   segments of one continuous rollout
      grid         run_c{ctx}_i{ic}_{uid}     context x initial-condition product
      legacy       run_{idx}_{uid}            independent draws
    """
    mesh = run.parent.name
    m_alt = re.match(r'run_a(\d+)_s(\d+)_', run.name)
    if m_alt:
        return (f'{mesh}/traj_a{m_alt.group(1)}', None, mesh)
    m_grid = re.match(r'run_c(\d+)_i(\d+)_', run.name)
    if m_grid:
        return (None, f'{mesh}/ic_{m_grid.group(2)}', mesh)
    return (None, None, mesh)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-dir', action='append', required=True,
                    help='dataset root (repeat for multiple, as in training)')
    ap.add_argument('--val-fraction', type=float, default=0.05,
                    help='must match the training value (default 0.05)')
    ap.add_argument('--deep', action='store_true',
                    help='hash every frame instead of first+middle (slow, exact)')
    ap.add_argument('--max-runs', type=int, default=None,
                    help='only hash this many runs (sampling, for a quick check)')
    args = ap.parse_args()

    runs = find_runs(args.data_dir)
    if not runs:
        raise SystemExit('no runs found')
    val = [r for r in runs if is_val_run(r, args.val_fraction)]
    train = [r for r in runs if not is_val_run(r, args.val_fraction)]
    print(f'\n=== 1. MECHANISM ===')
    print(f'runs: {len(runs)}   train: {len(train)}   val: {len(val)} '
          f'({100 * len(val) / len(runs):.1f}%, requested {100 * args.val_fraction:.1f}%)')
    if not val:
        print('  !! val set is EMPTY — every run is trained on. '
              'val_fraction too small for this corpus?')

    # A run cannot be in both (one decision, exclusive append) — assert it anyway.
    overlap = {r.resolve() for r in train} & {r.resolve() for r in val}
    print(f'train∩val run dirs: {len(overlap)}  '
          f'{"OK (disjoint)" if not overlap else "!! LEAK"}')

    by_name = defaultdict(list)
    for r in runs:
        by_name[r.name].append(r)
    dups = {n: v for n, v in by_name.items() if len(v) > 1}
    print(f'duplicate run names: {len(dups)}'
          + (f'  !! e.g. {list(dups)[:3]}' if dups else '  OK (all unique)'))

    # ---- 2. content ----
    print(f'\n=== 2. CONTENT (byte-identical frames across the split) ===')
    sample = slice(None) if args.max_runs is None else slice(0, args.max_runs)
    train_h, val_h, uni = {}, {}, set()
    for side, store in ((train[sample], train_h), (val[sample], val_h)):
        for r in side:
            for h, fn, is_uniform in frame_hashes(r, args.deep):
                store.setdefault(h, (r, fn))
                if is_uniform:
                    uni.add(h)
    shared = set(train_h) & set(val_h)
    hard, soft = sorted(shared - uni), sorted(shared & uni)
    print(f'hashed {len(train_h)} train / {len(val_h)} val frames '
          f'({"all frames" if args.deep else "first+middle per run"})')
    if hard:
        print(f'  !! {len(hard)} IDENTICAL non-uniform frame(s) in BOTH sets '
              f'— real duplication:')
        for h in hard[:5]:
            tr, tf = train_h[h]; va, vf = val_h[h]
            print(f'     train {tr.parent.name}/{tr.name}/{tf}')
            print(f'     val   {va.parent.name}/{va.name}/{vf}')
    else:
        print('  OK — no evolved val frame is identical to any train frame')
    if soft:
        print(f'  [info] {len(soft)} identical UNIFORM frame(s) across the split: runs '
              f'sharing an initial condition. Expected by design in grid corpora; '
              f'means the val run\'s t=0 state is not novel, not that data was copied.')

    # ---- 3. correlation ----
    print(f'\n=== 3. CORRELATION (not identical, but not independent) ===')
    for label, idx, note in (
        ('trajectory (alternating segments)', 0,
         'a val segment CONTINUES a trained one (same mesh, same freestream)'),
        ('initial condition (grid rows)', 1,
         'the val run starts from an IC that also appears in training'),
        ('mesh geometry', 2,
         'geometry seen in training; expected unless you hold out meshes'),
    ):
        sides = defaultdict(set)
        for r in runs:
            k = group_keys(r)[idx]
            if k is not None:
                sides[k].add('val' if is_val_run(r, args.val_fraction) else 'train')
        straddle = [k for k, s in sides.items() if len(s) > 1]
        if not sides:
            print(f'  {label}: n/a for this corpus')
        else:
            print(f'  {label}: {len(straddle)}/{len(sides)} groups span train AND val')
            if straddle:
                print(f'      -> {note}')
                print(f'      e.g. {straddle[:3]}')

    print('\nNote: normalisation stats (hfm_input_stats.json) are computed over ALL '
          'runs including val — a mild, conventional form of leakage that affects '
          'only the input scaling, not the targets.')


if __name__ == '__main__':
    main()
