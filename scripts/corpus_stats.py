"""Corpus summary: runs, frames, meshes, save_t spread, per dataset root.

Reads the cached run index (hfm_run_index.json, the same cache training uses)
so it is instant and RDS-friendly; falls back to a directory walk for roots
that have never been indexed.  Counts are per root plus a grand total, which
is the number to quote in the report's dataset section.

Usage
-----
    python scripts/corpus_stats.py ../data/fvm_gen_alternating [more roots ...]
    python scripts/corpus_stats.py ../data/*          # every dataset at once
"""
import argparse
import json
import sys
from pathlib import Path

INDEX_FILE = 'hfm_run_index.json'


def run_dirs_for(root: Path) -> list[Path]:
    """run_* dirs, whether laid out as root/run_* or root/mesh_*/run_*."""
    direct = [d for d in root.iterdir() if d.is_dir() and d.name.startswith('run')]
    if direct:
        return sorted(direct)
    return sorted(d for m in root.iterdir() if m.is_dir()
                  for d in m.iterdir() if d.is_dir() and d.name.startswith('run'))


def stats_for(root: Path) -> "dict | None":
    idx_path = root / INDEX_FILE
    runs: dict = {}
    if idx_path.exists():
        try:
            runs = json.loads(idx_path.read_text()).get('runs', {})
        except Exception as e:
            print(f'  [warn] unreadable index in {root}: {e}', file=sys.stderr)
    if not runs:                      # never indexed: walk (slow on RDS)
        dirs = run_dirs_for(root) if root.is_dir() else []
        if not dirs:
            return None
        runs = {str(d.relative_to(root)):
                {'files': [f.name for f in sorted(d.glob('t_*.npz'))],
                 'save_t': None, 'cold': (d / 'ic.json').exists()}
                for d in dirs}
        print(f'  [note] {root.name}: no index cache, counted by walking '
              f'(train once to build the cache)', file=sys.stderr)

    frames  = [len(r.get('files', [])) for r in runs.values()]
    save_ts = sorted(r['save_t'] for r in runs.values() if r.get('save_t'))
    meshes  = {k.split('/')[0] for k in runs if '/' in k}
    return {
        'runs':    len(runs),
        'frames':  sum(frames),
        'meshes':  len(meshes) or 1,
        'min_f':   min(frames) if frames else 0,
        'max_f':   max(frames) if frames else 0,
        'save_lo': save_ts[0]  if save_ts else None,
        'save_hi': save_ts[-1] if save_ts else None,
        'cold':    sum(1 for r in runs.values() if r.get('cold')),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', type=Path, nargs='+')
    args = ap.parse_args()

    tot_runs = tot_frames = tot_meshes = 0
    print(f'{"dataset":<28} {"meshes":>6} {"runs":>6} {"frames":>9} '
          f'{"f/run":>9} {"save_t":>13} {"cold":>5}')
    for root in args.roots:
        s = stats_for(root)
        if s is None:
            print(f'{root.name:<28} {"-":>6} {"-":>6} {"-":>9}   (no runs found)')
            continue
        srange = (f'{s["save_lo"]:.3f}-{s["save_hi"]:.3f}'
                  if s['save_lo'] is not None else 'fixed/legacy')
        print(f'{root.name:<28} {s["meshes"]:>6} {s["runs"]:>6} {s["frames"]:>9} '
              f'{s["min_f"]:>4}-{s["max_f"]:<4} {srange:>13} {s["cold"]:>5}')
        tot_runs, tot_frames = tot_runs + s['runs'], tot_frames + s['frames']
        tot_meshes += s['meshes']
    if len(args.roots) > 1:
        print(f'{"TOTAL":<28} {tot_meshes:>6} {tot_runs:>6} {tot_frames:>9}')


if __name__ == '__main__':
    main()
