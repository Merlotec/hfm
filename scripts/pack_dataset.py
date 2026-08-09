#!/usr/bin/env python3
"""
Pack each run's per-frame .npz files into one mmap-able array.

    python scripts/pack_dataset.py ../data/fvm_gen_alternating
    python scripts/pack_dataset.py ../data/fvm_gen_alternating --workers 16
    python scripts/pack_dataset.py ../data/fvm_gen_alternating --delete-npz

Why
---
A run holds ~31 frames of ~120 KB, so a corpus of 3205 runs is ~99,000 small
files.  Every training step opens 320 of them per rank (32 samples x 10 frames),
and on a parallel filesystem each of those is a metadata round trip.  That access
pattern -- many tiny files, random order, high concurrency -- is the one Lustre
handles worst, and it is what makes RDS reads feel slow.

Packing writes, per run:

    frames_fp16.npy   [T, n_cells, 4] float16   the payload, uncompressed
    frames_meta.npz   prim_mean/prim_std [T,1,4], times [T], names [T]

so a step becomes 32 opens of one contiguous array that the OS can page in
lazily, instead of 320 opens plus 320 zlib inflations.  Compression is dropped
deliberately: the payload is already fp16 and .npz was only buying ~3%, which is
not worth losing mmap for.

The dataset picks packs up automatically (scan_run records `packed`), so no
training flag is needed.  Runs that are not packed keep working unchanged, and a
packed run keeps its .npz files unless --delete-npz is passed.

Safe to re-run: a run whose pack is already present and consistent is skipped, so
this can be pointed at a growing dataset repeatedly.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfm.data import PACK_FILE, PACK_META, _atomic_write, _frame_files  # noqa: E402


def pack_run(run_dir: Path, force: bool = False, delete_npz: bool = False) -> tuple:
    """Returns (status, n_frames, bytes_written)."""
    files = _frame_files(run_dir)
    if not files:
        return 'empty', 0, 0

    pack, meta = run_dir / PACK_FILE, run_dir / PACK_META
    if pack.exists() and meta.exists() and not force:
        try:
            # Consistent means "same number of frames"; a run that grew since the
            # last pack is repacked rather than silently serving a truncated view.
            if np.load(meta)['names'].shape[0] == len(files):
                return 'skipped', len(files), 0
        except Exception:
            pass

    cells, means, stds, times = [], [], [], []
    for f in files:
        try:
            z = np.load(f)
        except Exception:
            # Stop at the first unreadable frame rather than skipping it: the
            # frames already packed are contiguous in time, and splicing across a
            # gap would hand the model a discontinuous sequence.
            break
        cells.append(z['cell_primatives'])           # fp16, as stored
        means.append(z['prim_mean'])
        stds.append(z['prim_std'])
        times.append(float(f.stem[2:]))
    if not cells:
        return 'unreadable', 0, 0

    # np.save/np.savez append .npy/.npz to a path that lacks the suffix, which
    # would defeat the atomic temp-then-rename (the temp name ends in .tmp.<pid>).
    # Writing through an open handle keeps the exact filename.
    def _write(fn):
        def inner(tmp: Path):
            with open(tmp, 'wb') as fh:
                fn(fh)
        return inner

    arr = np.stack(cells).astype(np.float16)         # [T, n_cells, 4]
    _atomic_write(pack, _write(lambda fh: np.save(fh, arr, allow_pickle=False)))
    _atomic_write(meta, _write(lambda fh: np.savez(
        fh,
        prim_mean=np.stack(means).astype(np.float32),
        prim_std=np.stack(stds).astype(np.float32),
        times=np.asarray(times, dtype=np.float64),
        names=np.asarray([f.name for f in files[:len(cells)]]),
    )))
    written = pack.stat().st_size + meta.stat().st_size

    if delete_npz:
        for f in files[:len(cells)]:
            f.unlink(missing_ok=True)
    return 'packed', len(cells), written


def find_runs(data_dir: Path) -> list:
    runs = []
    for mesh in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        runs.extend(sorted(p for p in mesh.iterdir()
                           if p.is_dir() and p.name.startswith('run')))
    if not runs:      # flat layout: runs directly under the data dir
        runs = sorted(p for p in data_dir.iterdir()
                      if p.is_dir() and p.name.startswith('run'))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data_dir', type=Path)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--force', action='store_true', help='repack even if up to date')
    ap.add_argument('--delete-npz', action='store_true',
                    help='remove the per-frame .npz files after a successful pack '
                         '(irreversible: the pack becomes the only copy)')
    args = ap.parse_args()

    runs = find_runs(args.data_dir)
    if not runs:
        raise SystemExit(f'No run_* directories found in {args.data_dir}')
    if args.delete_npz:
        print(f'!! --delete-npz: the {len(runs)} run(s) below will lose their '
              f'per-frame .npz files.  The pack becomes the only copy.')
        if input('   type "yes" to continue: ').strip() != 'yes':
            raise SystemExit('aborted')

    print(f'Packing {len(runs)} runs from {args.data_dir} ({args.workers} workers)')
    tally, frames, written = {}, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(pack_run, d, args.force, args.delete_npz): d for d in runs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status, n, b = fut.result()
            except Exception as e:
                status, n, b = f'error:{type(e).__name__}', 0, 0
            tally[status] = tally.get(status, 0) + 1
            frames += n
            written += b
            if i % 200 == 0 or i == len(runs):
                print(f'  {i}/{len(runs)}  ' +
                      '  '.join(f'{k}={v}' for k, v in sorted(tally.items())))

    print(f'\n{frames:,} frames  ->  {written / 1e9:.2f} GB in '
          f'{2 * tally.get("packed", 0):,} files')
    print('Training picks packs up automatically; no flag needed.')
    print('Delete hfm_run_index.json to force a re-index if runs were repacked.')


if __name__ == '__main__':
    main()
