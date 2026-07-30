"""
Backfill normalisation stats into an existing checkpoint.

Checkpoints written before norm_mean/norm_std were saved carry no normalisation, so
inference had to derive it from whatever data dir it was pointed at.  When that dir's
stats differ from the TRAINING stats (or the dir has none and the stale foundation
fallback is used), the model is fed off-distribution inputs and its delta is
de-normalised by the wrong scale — the fields come out pulled toward the mean and
grainy, even though training metrics were fine.

This writes the correct stats into the checkpoint so infer.py uses them directly.

Usage
-----
    python scripts/set_ckpt_stats.py checkpoints/train_step025000.pt \
        --data-dir ../data/fvm_gen_datasets

    # or point straight at the stats file, and/or write to a new checkpoint
    python scripts/set_ckpt_stats.py ckpt.pt --stats path/to/hfm_input_stats.json \
        --out ckpt_fixed.pt

IMPORTANT: pass the dir the model was TRAINED on, not the one you infer on.
"""
import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint', type=Path, help='.pt or .ckpt to update')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--data-dir', type=Path,
                     help='TRAINING data dir containing hfm_input_stats.json')
    src.add_argument('--stats', type=Path, help='path to an hfm_input_stats.json')
    ap.add_argument('--out', type=Path, default=None,
                    help='write here instead of updating in place')
    ap.add_argument('--force', action='store_true',
                    help='overwrite stats already present in the checkpoint')
    args = ap.parse_args()

    stats_path = args.stats if args.stats else args.data_dir / 'hfm_input_stats.json'
    if not stats_path.exists():
        raise SystemExit(f'No stats file at {stats_path}')
    with open(stats_path) as f:
        s = json.load(f)
    mean, std = [float(v) for v in s['mean']], [float(v) for v in s['std']]

    print(f'Checkpoint : {args.checkpoint}')
    print(f'Stats from : {stats_path}')
    print(f'  mean = {mean}')
    print(f'  std  = {std}')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit('Unexpected checkpoint format (not a dict).')

    existing = ckpt.get('norm_mean')
    if existing is not None and not args.force:
        print(f'\n  Checkpoint ALREADY has stats: mean={existing}')
        print('  Nothing written.  Pass --force to overwrite.')
        return

    ckpt['norm_mean'] = mean
    ckpt['norm_std'] = std

    out = args.out or args.checkpoint
    torch.save(ckpt, out)
    print(f'\n  Wrote norm_mean/norm_std → {out}')
    print('  infer.py will now use these instead of guessing from the data dir.')


if __name__ == '__main__':
    main()
