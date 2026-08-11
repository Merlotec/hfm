"""Detect corpus duplication from the fixed-seed generator bug.

Until 2026-08-10 the generators hard-seeded their RNGs, so parallel or repeated
invocations silently reproduced identical meshes AND identical trajectories
under different run uids -- and, because the val split hashes run NAMES, byte
copies of one trajectory could land on both sides of the split.  This script
answers "is my corpus actually distinct data?" three ways:

1. mesh check   : hash the mesh VERTICES of every mesh dir (the pickles of
                  identical meshes hash differently, so the geometry is hashed,
                  not the file).  Duplicate geometry across mesh dirs = cloned
                  meshes.
2. run check    : group runs by their sampled parameters (theta vector, class,
                  save_t, BC summary) straight from the hfm_run_index.json
                  cache.  These are continuous draws; exact equality across
                  runs means the same RNG stream, i.e. duplicated generation.
                  Instant, no file reads beyond the index.
3. frame check  : for every group the run check flags, md5 the first frame
                  file of each member to confirm byte-level duplication
                  (--frames to enable; only flagged groups are read).

Exit status is non-zero if any duplication is found, so it can gate a
training submission.

Usage
-----
    python scripts/check_dedup.py ../data/fvm_gen_alternating
    python scripts/check_dedup.py --frames root1 root2 ...   # cross-root too
"""
import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hfm.data  # noqa: F401  (injects the solver dir into sys.path so the
                 # mesh classes inside shared_mesh.pkl are unpicklable)

INDEX_FILE = 'hfm_run_index.json'


def mesh_dirs_for(root: Path) -> list[Path]:
    if (root / 'shared_mesh.pkl').exists():
        return [root]
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / 'shared_mesh.pkl').exists())


def mesh_hash(mesh_dir: Path) -> "str | None":
    try:
        with open(mesh_dir / 'shared_mesh.pkl', 'rb') as f:
            mesh = pickle.load(f)['mesh']
        verts = mesh.vertices.cpu().numpy()
        return hashlib.md5(verts.tobytes()).hexdigest()[:12]
    except Exception as e:
        print(f'  [warn] could not hash {mesh_dir}: {type(e).__name__}: {e}')
        return None


def run_key(meta: dict) -> "tuple | None":
    """Identity of a run's random draws.  None when the run carries none of
    them (very old corpora), i.e. it cannot be checked from metadata."""
    ctx, bc, st = meta.get('ctx'), meta.get('bc'), meta.get('save_t')
    if ctx is None and bc is None:
        return None
    # NaN components mean "label absent" (legacy corpora) and must compare
    # EQUAL across runs — raw NaNs never equal each other, which made every
    # legacy run key unique and hid genuine duplicates.
    e = lambda x: 'nan' if float(x) != float(x) else round(float(x), 12)
    mk = lambda v: None if v is None else tuple(e(x) for x in v)
    ctx_k, bc_k = mk(ctx), mk(bc)
    # All-absent ctx AND absent bc carries no sampling identity at all.
    if (ctx_k is None or set(ctx_k) == {'nan'}) and bc_k is None:
        return None
    return (meta.get('ctx_cls'), ctx_k, bc_k,
            None if st is None else e(st))


def first_frame_md5(root: Path, rel: str) -> str:
    run = root / rel
    frames = sorted(run.glob('t_*.npz'))
    if not frames:
        return '<no frames>'
    return hashlib.md5(frames[0].read_bytes()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', type=Path, nargs='+')
    ap.add_argument('--frames', action='store_true',
                    help='confirm flagged run groups by hashing their first frame file')
    args = ap.parse_args()

    bad = False

    # ---- 1. mesh geometry ----
    by_hash: dict = defaultdict(list)
    for root in args.roots:
        for md in mesh_dirs_for(root):
            h = mesh_hash(md)
            if h:
                by_hash[h].append(md)
    n_meshes = sum(len(v) for v in by_hash.values())
    dupes = {h: v for h, v in by_hash.items() if len(v) > 1}
    print(f'meshes: {n_meshes} checked, {len(by_hash)} distinct geometries')
    for h, dirs in sorted(dupes.items()):
        bad = True
        loc = ', '.join(str(d) for d in dirs)
        print(f'  DUPLICATE geometry {h}: {loc}')

    # ---- 2. run parameter draws (from the index cache) ----
    for root in args.roots:
        try:
            runs = json.loads((root / INDEX_FILE).read_text())['runs']
        except Exception:
            print(f'{root.name}: no readable {INDEX_FILE} — train once (or run '
                  f'corpus_stats.py) to build it; skipping run check')
            continue
        groups: dict = defaultdict(list)
        unverifiable = 0
        for rel, meta in runs.items():
            k = run_key(meta)
            if k is None:
                unverifiable += 1
            else:
                groups[k].append(rel)
        dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
        n_dup_runs = sum(len(v) for v in dup_groups.values())
        print(f'{root.name}: {len(runs)} runs, {len(groups)} distinct parameter '
              f'draws, {n_dup_runs} runs in {len(dup_groups)} duplicate group(s)'
              + (f', {unverifiable} unverifiable (no sampled params in index)'
                 if unverifiable else ''))
        for k, rels in sorted(dup_groups.items(), key=lambda kv: -len(kv[1])):
            bad = True
            print(f'  DUPLICATE draw x{len(rels)}: {", ".join(sorted(rels)[:4])}'
                  + (' ...' if len(rels) > 4 else ''))
            if args.frames:
                sigs = {rel: first_frame_md5(root, rel) for rel in rels}
                distinct = len(set(sigs.values()))
                verdict = ('CONFIRMED byte-identical' if distinct == 1
                           else f'{distinct} distinct frame contents (params '
                                f'collide but frames differ — investigate)')
                print(f'    frames: {verdict}')

    if bad:
        print('\nRESULT: duplication detected — dedup or regenerate before '
              'trusting in-corpus validation numbers.')
        sys.exit(1)
    print('\nRESULT: no duplication detected.')


if __name__ == '__main__':
    main()
