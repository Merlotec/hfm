"""
Scan a dataset for unreadable / truncated frame files.

A simulation that crashes mid-write leaves a zero-byte or truncated t_*.npz.  During
training that surfaces as `EOFError: No data left in file` inside a DataLoader worker,
which kills the rank — and every other rank then dies in its next collective
("Connection closed by peer").  One bad file can end an 8-rank job hours in.

Usage
-----
    python scripts/check_frames.py ../data/fvm_gen_datasets
    python scripts/check_frames.py ../data/fvm_gen_datasets --delete   # remove bad files

By default it only reports.  --delete removes the unreadable files (the runs they
belong to keep their remaining frames; runs left with too few frames are simply
skipped by the dataset).
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def frame_files(root: Path) -> list[Path]:
    """All t_*.npz under root, whether laid out as root/run_* or root/mesh_*/run_*."""
    return sorted(root.glob('**/t_*.npz'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data_dir', type=Path)
    ap.add_argument('--delete', action='store_true',
                    help='delete unreadable files instead of only reporting them')
    ap.add_argument('--quick', action='store_true',
                    help='only check file size (fast); default also opens each file')
    args = ap.parse_args()

    files = frame_files(args.data_dir)
    if not files:
        raise SystemExit(f'No t_*.npz found under {args.data_dir}')
    print(f'Scanning {len(files)} frame files under {args.data_dir} ...')

    bad: list[tuple[Path, str]] = []
    for n, f in enumerate(files, 1):
        if n % 20000 == 0:
            print(f'  ... {n}/{len(files)}')
        try:
            if f.stat().st_size == 0:
                bad.append((f, 'zero-byte'))
                continue
            if not args.quick:
                with np.load(f) as d:            # catches truncated archives
                    _ = d['cell_primatives'].shape
        except Exception as e:
            bad.append((f, f'{type(e).__name__}: {e}'))

    print()
    if not bad:
        print(f'OK — all {len(files)} frame files readable.')
        return

    print(f'FOUND {len(bad)} unreadable file(s):')
    for f, why in bad[:40]:
        print(f'  {f}   [{why}]')
    if len(bad) > 40:
        print(f'  ... and {len(bad) - 40} more')

    runs = sorted({f.parent for f, _ in bad})
    print(f'\nAcross {len(runs)} run(s).')

    if args.delete:
        for f, _ in bad:
            f.unlink()
        print(f'\nDeleted {len(bad)} file(s).')
    else:
        print('\nRe-run with --delete to remove them.')
        sys.exit(1)


if __name__ == '__main__':
    main()
