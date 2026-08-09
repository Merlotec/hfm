"""
FVM dataset for HFM training.

Each item is a [T, C, H, W] tensor of T consecutive normalised, rendered frames
from one simulation run.  T must be >= n_warmup_frames + 2 so the trainer has
enough frames for warmup + one training step.

The renderer (MeshRenderer) converts FVM cell-level primitives to a smooth
pixel grid via barycentric interpolation.  It is built once per dataset and
cached to disk alongside the data as renderer_cache_{H}x{W}.pt.
"""

import hashlib
import json
import time
import os
import shutil
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler

# ---- inject solver path so MeshRenderer is importable ----
_FLSIM_ROOT = Path(__file__).resolve().parents[2]  # .../flsim
_SOLVER_DIR = _FLSIM_ROOT / 'fvm_solver'
_GEN_DIR    = _FLSIM_ROOT / 'fvm_model' / 'fvm_gen'
for _p in (_SOLVER_DIR, _GEN_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import MeshRenderer  # noqa: E402  (needs path injection above)


# ---------------------------------------------------------------------------
# Cache writes
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, write_fn) -> None:
    """Write a cache file so no reader can ever observe a partial one.

    Every rank runs setup() and every rank builds the same caches, so 8 processes
    used to torch.save() over the same path at once.  A reader landing mid-write
    sees a truncated file, which surfaces as whichever of EOFError / OSError
    (Errno 22) / UnpicklingError('Unsupported operand') the bytes happen to hit
    -- exactly the three different errors the ranks reported.

    write_fn writes to a private per-process temp path in the SAME directory;
    os.replace then swaps it in atomically, so a concurrent reader gets either
    the whole old file or the whole new one, never a mixture.
    """
    tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    try:
        write_fn(tmp)
        os.replace(tmp, path)          # atomic within a directory
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Renderer factory
# ---------------------------------------------------------------------------

def build_renderer(dataset_dir: Path, resolution: tuple[int, int],
                   device: str = 'cpu') -> MeshRenderer:
    """Load renderer from cache if available and consistent, otherwise build and cache it."""
    H, W = resolution
    cache = dataset_dir / f'renderer_cache_{H}x{W}.pt'

    mesh_pkl = dataset_dir / 'shared_mesh.pkl'
    if not mesh_pkl.exists():
        raise FileNotFoundError(f'shared_mesh.pkl not found in {dataset_dir}')
    with open(mesh_pkl, 'rb') as f:
        mesh_dict = pickle.load(f)
    fvm_mesh = mesh_dict['mesh']
    verts = fvm_mesh.vertices.cpu().numpy()
    n_cells = int(fvm_mesh.cells.shape[0])
    x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())

    if cache.exists():
        # A read failure is treated as a stale cache, not a fatal error: the cache
        # is derived data we can always rebuild, and a corrupt one (interrupted
        # write, killed job, full quota) should not take down the run.
        try:
            renderer = MeshRenderer.from_cache(str(cache), device=device)
            eps = 1e-3
            cache_ok = (
                renderer._c2v_tri.max().item() + 1 == n_cells
                and abs(renderer.xlim[0] - x0) < eps and abs(renderer.xlim[1] - x1) < eps
                and abs(renderer.ylim[0] - y0) < eps and abs(renderer.ylim[1] - y1) < eps
            )
            if cache_ok:
                return renderer
            print('  Renderer cache stale (mesh mismatch), rebuilding...')
        except Exception as e:
            print(f'  Renderer cache unreadable ({type(e).__name__}: {e}), rebuilding...')

    renderer = MeshRenderer(
        verts,
        fvm_mesh.cells.cpu().numpy(),
        resolution=resolution,
        device=device,
    )
    # Atomic: do NOT unlink the old cache first -- another rank may be reading it.
    _atomic_write(cache, lambda tmp: renderer.save_cache(str(tmp)))
    return renderer


# ---------------------------------------------------------------------------
# Pixel mask (fluid vs. hole)
# ---------------------------------------------------------------------------

def mesh_dirs_for(data_dir) -> "list[Path]":
    """Geometry dirs in a DETERMINISTIC (sorted) order.

    mesh_id indexes THIS list, so any consumer that builds a parallel mask table
    must use the same order.  The scripts used bare iterdir() (filesystem order)
    while the datamodule enumerated its own list -- two independent unsorted scans,
    so a mismatch would silently pair samples with another geometry's mask.
    """
    data_dir = Path(data_dir)
    if (data_dir / 'shared_mesh.pkl').exists():
        return [data_dir]
    return sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (p / 'shared_mesh.pkl').exists())


def load_mesh_masks(data_dir, resolution) -> torch.Tensor:
    """[n_mesh, 1, H, W] fluid masks stacked in mesh_id order."""
    dirs = mesh_dirs_for(data_dir)
    return torch.stack([
        load_pixel_mask(d, build_renderer(d, resolution), resolution)[0] for d in dirs
    ])


def build_pixel_mask(renderer: MeshRenderer, resolution: tuple[int, int]) -> torch.Tensor:
    """Boolean (1, 1, H, W) mask — True for pixels inside the fluid mesh."""
    H, W = resolution
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[renderer._interior_idx] = True
    return mask.view(1, 1, H, W)


def load_pixel_mask(dataset_dir: Path, renderer: MeshRenderer,
                    resolution: tuple[int, int]) -> torch.Tensor:
    """Return cached pixel mask, building and saving it if not yet cached."""
    H, W = resolution
    cache = dataset_dir / f'pixel_mask_{H}x{W}.pt'
    n_interior = len(renderer._interior_idx)
    if cache.exists():
        try:
            m = torch.load(cache, weights_only=True)
            if (m.shape == (1, 1, H, W)
                    and int(m.sum()) == n_interior
                    and m.view(-1)[renderer._interior_idx].all()):
                return m
            print('  Pixel mask stale (renderer mismatch), rebuilding...')
        except Exception as e:
            print(f'  Pixel mask unreadable ({type(e).__name__}: {e}), rebuilding...')
    mask = build_pixel_mask(renderer, resolution)
    _atomic_write(cache, lambda tmp: torch.save(mask, tmp))
    print(f'  Pixel mask saved — {mask.sum().item()} fluid / {mask.numel()} total pixels')
    return mask


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_normalisation_stats(
    runs: list[tuple[Path, MeshRenderer]],
    n_samples: int = 300,
    first_frame: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-channel mean and std by sampling frames across all runs."""
    all_files: list[tuple[Path, MeshRenderer]] = []
    for d, renderer in runs:
        fs = _frame_files(d)
        all_files.extend([(f, renderer) for f in fs[first_frame:]])

    n = min(n_samples, len(all_files))
    idx = torch.randperm(len(all_files))[:n].tolist()

    C = 4
    s1 = torch.zeros(C)
    s2 = torch.zeros(C)
    cnt = torch.zeros(C)
    n_bad = 0
    for i in idx:
        fpath, renderer = all_files[i]
        # Skip truncated/zero-byte frames rather than aborting the whole run at setup:
        # these are statistics over a random sample, so a few dropped frames are
        # statistically irrelevant.
        try:
            d = np.load(fpath)
            vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
            frame = renderer.render_cell_smooth(vals)   # [C, H, W]
        except Exception:
            n_bad += 1
            continue
        for c in range(C):
            px = frame[c]
            fin = px[torch.isfinite(px)]
            s1[c]  += fin.sum()
            s2[c]  += (fin ** 2).sum()
            cnt[c] += fin.numel()
    if n_bad:
        print(f'  [warn] stats: skipped {n_bad}/{n} unreadable frame(s) '
              '(run scripts/check_frames.py to find them)')

    mean = s1 / cnt
    std  = ((s2 / cnt) - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    return mean, std


# ---------------------------------------------------------------------------
# Truncated runs
# ---------------------------------------------------------------------------

class TruncatedRunError(RuntimeError):
    """A run directory that could not yield a single readable sequence."""

    def __init__(self, sim_dir: Path, reason: str):
        super().__init__(f'{sim_dir}: {reason} — this run is truncated')
        self.sim_dir = Path(sim_dir)
        self.reason = reason


# Runs this PROCESS has already dealt with, so a dead run is not re-attempted (or
# re-deleted) on every draw.  Per-process by design: workers are forked copies and
# the on-disk delete is what they actually share.
_quarantined: set = set()


def prune_enabled() -> bool:
    """Delete truncated runs on sight.  HFM_PRUNE_BAD_RUNS=0 quarantines without
    deleting, which is the setting to use on a dataset you cannot regenerate."""
    return os.environ.get('HFM_PRUNE_BAD_RUNS', '1') not in ('0', 'false', 'False')


def quarantine_run(sim_dir: Path, reason: str) -> None:
    """Take a truncated run out of circulation, deleting it when pruning is on.

    Guarded deliberately: this removes data irreversibly, so it refuses anything
    that is not shaped like a solver run directory (name starts with 'run',
    carries params.json, sits under a mesh dir).  A path that fails the guard is
    only skipped, never deleted — a bug in the caller must not be able to take
    out a mesh directory or the dataset root.
    """
    sim_dir = Path(sim_dir)
    if sim_dir in _quarantined:
        return
    _quarantined.add(sim_dir)

    # Already gone: another rank deleted it and this process is just catching up
    # (its in-memory dataset still references the dir).  Nothing to do, and
    # warning about it every epoch from every worker is pure noise.
    if not sim_dir.exists():
        return

    looks_like_run = (
        sim_dir.is_dir()
        and sim_dir.name.startswith('run')
        and (sim_dir / 'params.json').exists()
        and sim_dir.parent != sim_dir
    )
    if not looks_like_run:
        print(f'  [warn] refusing to delete {sim_dir}: does not look like a run '
              f'directory; skipping it for this process only')
        return
    if not prune_enabled():
        print(f'  [warn] truncated run {sim_dir} ({reason}); skipping '
              f'(HFM_PRUNE_BAD_RUNS=0, not deleting)')
        return

    n_frames = len(list(sim_dir.glob('t_*.npz')))
    try:
        shutil.rmtree(sim_dir)
    except Exception as e:
        print(f'  [warn] could not delete {sim_dir}: {type(e).__name__}: {e}')
        return
    print(f'  [prune] deleted truncated run {sim_dir} ({reason}, {n_frames} frames)')
    # Append-only record so a pruned dataset can still be accounted for later.
    # One small O_APPEND write per line, which is atomic across ranks.
    try:
        line = (f'{datetime.now().isoformat(timespec="seconds")}\t{sim_dir.name}\t'
                f'{n_frames} frames\t{reason}\n')
        with open(sim_dir.parent / 'deleted_runs.log', 'a') as fh:
            fh.write(line)
    except Exception:
        pass


class ResilientConcat(Dataset):
    """ConcatDataset that cannot be killed by one bad run.

    A truncated run used to raise out of the DataLoader worker, which killed the
    rank, which stranded every other rank in its next collective ("Connection
    closed by peer").  One unreadable file took down all 8 ranks hours in.  Here
    the bad run is quarantined (and deleted, see quarantine_run) and the draw is
    retried at a different random index, so training simply continues.
    """

    def __init__(self, dataset: ConcatDataset, max_retries: int = 8):
        self.dataset = dataset
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        n = len(self.dataset)
        for _ in range(self.max_retries):
            try:
                return self.dataset[idx]
            except TruncatedRunError as e:
                quarantine_run(e.sim_dir, e.reason)
            except Exception as e:
                print(f'  [warn] sample {idx} failed ({type(e).__name__}: {e}); redrawing')
            idx = int(torch.randint(n, (1,)).item())
        raise RuntimeError(
            f'{self.max_retries} consecutive unreadable samples — the dataset is '
            'likely mostly truncated; check it with scripts/check_frames.py')


# ---------------------------------------------------------------------------
# Single-run dataset
# ---------------------------------------------------------------------------

def is_val_run(sim_dir: Path, val_fraction: float) -> bool:
    """Deterministic held-out split, decided per RUN.

    Split at the RUN level, never at the window level: windows slide by one frame,
    so two windows from the same run share up to seq_len-1 frames and a window-level
    split would put near-copies of the training data in the validation set.

    The decision is a stable md5 of the run's directory name, NOT Python's hash()
    (salted per process, so the 8 ranks would each hold out a different subset) and
    NOT an index into a sorted list (which reshuffles the whole split whenever a run
    is added or pruned).  Same run name, same side of the split, on every rank and
    across restarts.
    """
    if val_fraction <= 0.0:
        return False
    h = hashlib.md5(sim_dir.name.encode()).digest()
    return (int.from_bytes(h[:4], 'big') % 10_000) < val_fraction * 10_000


def _frame_files(sim_dir: Path) -> "list[Path]":
    """A run's t_*.npz frames in simulation-time order."""
    return sorted(
        [f for f in sim_dir.iterdir()
         if f.name.startswith('t_') and f.name.endswith('.npz')],
        key=lambda f: float(f.stem[2:]),
    )


# The per-segment physics context.  gen_alternating resamples every one of these
# for each segment and writes them into that segment's params.json; the freestream
# (ic_param_specs) is fixed per TRAJECTORY, so these are the only labels that
# actually distinguish one segment from the next.  Loguniform-sampled parameters
# are regressed in log space, which is both the space they were drawn from and the
# one where a factor-of-two error costs the same at either end of the range.
CTX_PARAM_KEYS = (
    ('visc_n',           False),
    ('visc_min_factor',  False),
    ('visc_gamma_scale', True),
    ('viscosity',        True),
    ('visc_bulk',        True),
    ('thermal_cond',     True),
    ('C_v',              False),
    ('gamma',            False),
)
CTX_MODEL_CHOICES = ('Newtonian', 'PowerLaw', 'Carreau', 'HerschelBulkley')


def _is_cold_start(sim_dir: Path) -> bool:
    """True when this run begins from an initial condition rather than continuing
    a developed field.

    gen_alternating chains segments: segment 0 starts from the IC, and segments
    1..n-1 hand off the previous segment's final state, so only segment 0 carries
    a startup transient.  ic.json records segment_id; runs without it (run_gen's
    independent trajectories) are always cold starts."""
    try:
        return int(json.loads((sim_dir / 'ic.json').read_text())['segment_id']) == 0
    except Exception:
        return True


def _settle_skip(files: "list[Path]", settle_time: float, cold: bool) -> int:
    """Frames to drop from the front of a cold-start run.

    Measured decay of the one-step persistence loss after an impulsive start
    (dt=0.01 data, per unit sim time):

        0.00-0.05 s  1.635      0.40-0.70 s  0.077
        0.05-0.10 s  1.221      0.70-1.00 s  0.012
        0.10-0.20 s  0.829      1.00-1.50 s  0.003
        0.20-0.40 s  0.301      2.50-4.00 s  0.0008

    so the field settles by roughly 0.7 s, a 140x drop.  The cut is in SIM TIME,
    not frames, because a segment spans steps_per_segment * save_t: at
    save_t=0.01 all 30 frames sit inside the transient, while at save_t=0.2 only
    the first four do.  A frame-count cut would be wrong at both ends.
    """
    if settle_time <= 0.0 or not cold:
        return 0
    for i, f in enumerate(files):
        try:
            if float(f.stem[2:]) >= settle_time:
                return i
        except ValueError:
            return 0
    return len(files)          # whole run is transient


def _context_params_from(rec: dict) -> "tuple[Optional[torch.Tensor], int]":
    """(continuous context vector, viscosity-model class) for one run.

    Missing or unparseable values become NaN / -1, which the probe loss masks per
    sample, so a dataset that predates these fields still trains (it just has no
    context head to speak of)."""
    vals = []
    for k, is_log in CTX_PARAM_KEYS:
        v = rec.get(k)
        try:
            v = float(v)                                    # type: ignore[arg-type]
        except (TypeError, ValueError):
            vals.append(float('nan')); continue
        vals.append(float(np.log(v)) if is_log and v > 0 else
                    (float('nan') if is_log else v))
    m = rec.get('visc_model')
    cls = CTX_MODEL_CHOICES.index(m) if m in CTX_MODEL_CHOICES else -1
    return torch.tensor(vals, dtype=torch.float32), cls


# ---------------------------------------------------------------------------
# Run index
# ---------------------------------------------------------------------------

# Packed runs.  One .npy per run instead of one .npz per frame: the corpus is
# ~99k files of ~120 KB, and each training step opens 320 of them per rank (32
# samples x 10 frames), which is exactly the access pattern a parallel filesystem
# is worst at.  Packing makes that 32 opens of one contiguous, mmap-able array.
# Compression buys ~3% here (the payload is already fp16), so the pack is
# uncompressed and can be paged in lazily rather than decompressed whole.
PACK_FILE = 'frames_fp16.npy'
PACK_META = 'frames_meta.npz'

INDEX_FILE = 'hfm_run_index.json'
# Bump when the fields scan_run produces change, so a stale index is rebuilt
# rather than silently feeding old values into a new label space.
INDEX_VERSION = 2


def _bc_summary(files: "list[Path]") -> "Optional[list]":
    """Physical boundary-condition summary [mean(C), std(C)] for a run."""
    fallback = None
    for f in files[:6]:
        try:
            z = np.load(f)
            # De-normalise, exactly as the cell path and the solver's own loader do
            # (fvm_solver/.../saving.py).  The .npz stores bc_vals z-scored by THAT
            # FRAME's cell statistics, so reading it raw gives "how far the boundary
            # sits from this frame's own field mean, in units of its own field std"
            # -- a frame-relative number that mixes the forcing with the field
            # statistics and is partly computable from the frame the encoder already
            # sees.  That is the opposite of what this target is for.
            bc = (z['bc_primatives'].astype(np.float32)
                  * z['prim_std'] + z['prim_mean'])            # [N, C], physical
            with np.errstate(invalid='ignore'):
                summ = np.concatenate([np.nanmean(bc, 0), np.nanstd(bc, 0)])
            if not np.isfinite(summ).all():
                continue
            if fallback is None:
                fallback = summ
            # Skip a uniform frame: at t=0 a cold-start segment has a constant
            # field, so the std half of the summary is all zeros.  Only s00
            # segments start uniform, so accepting it would hand those runs a
            # differently-shaped label from every continuation segment.
            if not np.any(summ[bc.shape[1]:] > 0):
                continue
            return [float(x) for x in summ]
        except Exception:
            continue
    return None if fallback is None else [float(x) for x in fallback]


def scan_run(sim_dir: Path) -> dict:
    """Everything about a run that does not depend on seq_len, in ONE pass.

    This is the only place that touches a run's files at setup time, so the
    result can be cached in the dataset index and reused by every rank and every
    later job.  params.json used to be opened twice (save_t, then the context
    parameters); it is read once here.
    """
    files = _frame_files(sim_dir)
    try:
        rec = json.loads((sim_dir / 'params.json').read_text())
    except Exception:
        rec = {}
    ctx, cls = _context_params_from(rec)
    return {
        'files':   [f.name for f in files],
        'packed':  (sim_dir / PACK_FILE).exists() and (sim_dir / PACK_META).exists(),
        'save_t':  _save_t_from(rec, files),
        # Distinguishes gen_alternating output (explicit per-segment save_t) from
        # legacy fixed-interval corpora, which get temporal-stride augmentation.
        'has_save_t': bool(rec.get('save_t')),
        'cold':    _is_cold_start(sim_dir),
        'bc':      _bc_summary(files),
        'ctx':     None if ctx is None else [float(x) for x in ctx],
        'ctx_cls': cls,
    }


def load_run_index(data_dir: Path, run_dirs: "list[Path]") -> dict:
    """Scanned metadata for every run, reusing a cached index where possible.

    Warm start costs one file read plus one directory listing per mesh, instead
    of ~5 filesystem operations per run on every rank.  Only runs that are new
    since the index was written are scanned, and entries for runs that have gone
    (pruned by quarantine_run, say) are dropped, so the index self-heals without
    ever needing a full rebuild.
    """
    path = data_dir / INDEX_FILE
    cached: dict = {}
    try:
        blob = json.loads(path.read_text())
        if blob.get('version') == INDEX_VERSION:
            cached = blob['runs']
    except Exception:
        pass

    keys = {d: str(d.relative_to(data_dir)) for d in run_dirs}
    index, missing = {}, []
    for d, k in keys.items():
        if k in cached:
            index[k] = cached[k]
        else:
            missing.append((d, k))
    if missing:
        print(f'  Indexing {len(missing)} new run(s)'
              + (f' (of {len(run_dirs)})' if len(missing) != len(run_dirs) else ''))
        for d, k in missing:
            index[k] = scan_run(d)
    if missing or len(index) != len(cached):
        try:
            _atomic_write(path, lambda tmp: tmp.write_text(json.dumps(
                {'version': INDEX_VERSION, 'runs': index})))
        except Exception as e:
            print(f'  [warn] could not write {path.name}: {type(e).__name__}: {e}')
    return {d: index[k] for d, k in keys.items()}


def _save_t_from(rec: dict, files: "list[Path]") -> float:
    """Frame interval of a run, in simulation time.

    Prefers params.json['save_t'] (written per segment by gen_alternating);
    otherwise infers it from consecutive frame timestamps, which is exact for
    any evenly-sampled run; otherwise the legacy constant."""
    try:
        v = rec.get('save_t')
        if v:
            return float(v)
    except Exception:
        pass
    try:
        if len(files) >= 2:
            dt = float(files[1].stem[2:]) - float(files[0].stem[2:])
            if dt > 0:
                return dt
    except Exception:
        pass
    return 0.01


class FVMSequenceDataset(Dataset):
    """
    Sliding-window sequences of `seq_len` consecutive rendered frames from one
    simulation run.

    Returns [seq_len, C, H, W] float32 tensors, normalised per channel.
    """

    def __init__(
        self,
        sim_dir:     Path,
        renderer:    MeshRenderer,
        seq_len:     int,
        mean:        torch.Tensor,
        std:         torch.Tensor,
        first_frame: int = 0,
        frame_cache: Optional[list[torch.Tensor]] = None,
        mesh_id:     Optional[int] = None,
        paths_override: "Optional[list[Path]]" = None,
        n_context:   Optional[int] = None,
        ctx_random:  bool = True,
        settle_time: float = 0.0,
        dt_stride_max: int = 1,
        meta:        Optional[dict] = None,
    ):
        # `meta` is a pre-scanned record from the dataset index (see scan_run).
        # Without it this constructor touches the filesystem five times per run --
        # one directory listing, params.json, ic.json and one or two .npz loads --
        # which every rank repeats for every run at startup.  On a shared
        # parallel filesystem that is the dominant cost of setup().
        if meta is None:
            meta = scan_run(sim_dir)

        if paths_override is not None:
            files = paths_override
            offset = 0
        else:
            all_files = [sim_dir / n for n in meta['files']]
            skip = first_frame
            skip += _settle_skip(all_files[skip:], settle_time, meta['cold'])
            files = all_files[skip:]
            offset = skip
        self.paths      = files
        # Packed-run access.  The memmap is opened lazily in _get_frame because a
        # DataLoader worker is forked after construction and an inherited mmap is
        # not safe to share; each worker gets its own handle on first use.
        self.sim_dir     = sim_dir
        self._packed     = bool(meta.get('packed')) and paths_override is None
        self._pack_off   = offset
        self._pack: Optional[np.ndarray] = None
        self._pack_mean: Optional[np.ndarray] = None
        self._pack_std:  Optional[np.ndarray] = None
        self.renderer   = renderer
        self.seq_len    = seq_len
        # Index into FVMDataModule.mesh_masks.  A batch mixes geometries freely
        # (ConcatDataset + shuffle), so the mask must be gathered PER SAMPLE.
        # Training used to apply ONE mask from mesh_dirs[0] to every sample, which
        # is wrong for every other geometry -- masks differ by ~13% of the frame
        # between meshes, corrupting the loss mask AND the model's mask channel.
        self.mesh_id    = mesh_id
        self.mean       = mean.view(-1, 1, 1)   # [C, 1, 1] for broadcasting
        self.std        = std.view(-1, 1, 1)
        self._cache     = frame_cache            # optional pre-rendered cache

        self.bc_raw  = (torch.tensor(meta['bc'], dtype=torch.float32)
                        if meta['bc'] is not None else None)
        self.bc      = self.bc_raw               # normalised via set_bc_norm()
        self.save_t  = meta['save_t']
        self.ctx_raw = (torch.tensor(meta['ctx'], dtype=torch.float32)
                        if meta['ctx'] is not None else None)
        self.ctx_cls = meta['ctx_cls']
        self.ctx_vec = self.ctx_raw              # normalised via set_ctx_norm()

        # Where the context frames come from.  With n_context set and ctx_random on,
        # the context block is drawn from a RANDOM position earlier in the same run
        # rather than always sitting immediately before the prediction.
        #
        # Always taking the preceding frames lets the context double as "a few extra
        # recent frames": the model can read x_{t-1} out of it and finite-difference
        # its way forward, instead of using the context for what it is meant to
        # carry, the run's physics (dt, BCs, the hidden regime).  It also pins one
        # context to each prediction point, so a run offers |windows| distinct
        # (context, target) pairs; drawing the context freely gives ~|windows| times
        # more, which is real augmentation against memorising specific triples.
        #
        # The block is constrained to end at or before the INPUT frame, so no target
        # frame can ever appear in the context.  Without that the model could read
        # the answer straight out of its own conditioning.
        self.n_context  = n_context
        self.ctx_random = bool(ctx_random and n_context is not None)

        # Temporal-stride augmentation for LEGACY fixed-interval corpora.  A
        # gen_alternating segment carries its dt in save_t and is used at stride
        # 1; an old-style run is stuck at save_t=0.01, so each sample draws a
        # stride s in [1, dt_stride_max] and uses every s-th frame, making its
        # effective timestep s * save_t.  The dt LABEL returned with the sample
        # is that effective value, so the probe target and everything downstream
        # stay exact, and the trainer needs no striding of its own.
        self._rand_stride = bool(ctx_random)     # deterministic in validation
        self.dt_stride_max = max(1, int(dt_stride_max))

    def set_bc_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """z-score the BC summary with dataset-level stats (regression target)."""
        if self.bc_raw is not None:
            self.bc = (self.bc_raw - mean) / std.clamp(min=1e-6)

    def set_ctx_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """z-score the physics-context vector, so all nine parameters carry equal
        weight in the regression regardless of their native units (thermal_cond
        spans 5e-7..5e-3, gamma spans 1.1..1.67)."""
        if self.ctx_raw is not None:
            self.ctx_vec = (self.ctx_raw - mean) / std.clamp(min=1e-6)

    def __len__(self) -> int:
        return max(0, len(self.paths) - self.seq_len + 1)

    # Warn at most once per worker process about unreadable frames, so a handful of
    # bad files don't flood the log.
    _warned_corrupt = False

    def __getitem__(self, idx: int):
        # A truncated / zero-byte .npz (left behind by a simulation that crashed
        # mid-write) makes np.load raise EOFError.  Letting that propagate kills the
        # DataLoader worker, which kills the rank, which strands every OTHER rank in
        # its next collective ("Connection closed by peer") — i.e. one bad file takes
        # down the whole job hours in.  Skip to a different window instead, and return
        # a wholly valid sequence rather than splicing around the bad frame.
        n = max(1, len(self))
        for _ in range(4):
            try:
                idxs, stride = self._window(idx)
                frames = torch.stack([self._get_frame(i) for i in idxs])
                if self.mesh_id is not None:
                    # + geometry id, + BC summary, + EFFECTIVE frame interval,
                    # + the hidden physics context (vector, viscosity-model class)
                    bc = self.bc if self.bc is not None \
                        else torch.full((8,), float('nan'))
                    cx = self.ctx_vec if self.ctx_vec is not None \
                        else torch.full((len(CTX_PARAM_KEYS),), float('nan'))
                    return (frames, self.mesh_id, bc, self.save_t * stride,
                            cx, self.ctx_cls)
                return frames   # [T, C, H, W]
            except Exception as e:                       # unreadable/corrupt frame
                if not FVMSequenceDataset._warned_corrupt:
                    bad = self.paths[min(idx, len(self.paths) - 1)]
                    print(f'  [warn] unreadable frame near {bad}: {type(e).__name__}: {e}')
                    print('         Skipping this sample.  Find bad files with:')
                    print('         python scripts/check_frames.py <data_dir>')
                    FVMSequenceDataset._warned_corrupt = True
                idx = (idx + self.seq_len) % n
        raise TruncatedRunError(self.paths[0].parent,
                                'no valid sequence after 4 attempts')

    def _window(self, idx: int) -> "tuple[list[int], int]":
        """(frame indices, stride) for sample `idx`: context block then prediction.

        Layout matches what the trainer expects, frames[:n_context] is the context
        and frames[n_context:] is the input frame plus its targets, so nothing
        downstream changes.  Both blocks share one stride s, so the sequence is a
        faithful sampling of the run at effective timestep s * save_t.

        Stride: 1 unless dt_stride_max > 1 (legacy corpora), where s is drawn
        uniformly when ctx_random is on and cycles deterministically (1 + idx mod
        max) in validation, clamped so the window fits inside the run.

        Context block: starts at c, drawn uniformly from [0, idx + s] when
        ctx_random is on (c = idx is the contiguous case; c = idx + s ends the
        context ON the input frame, which the model already sees).  Its last
        frame c + (nc-1)s never exceeds the input frame p = idx + nc*s, so no
        target can leak into the conditioning.
        """
        nc = self.n_context
        span = self.seq_len - 1
        s_fit = max(1, (len(self.paths) - 1 - idx) // max(1, span))
        if self.dt_stride_max > 1:
            if self._rand_stride:
                s = 1 + int(torch.randint(min(self.dt_stride_max, s_fit), (1,)).item())
                s = min(s, s_fit)
            else:
                s = 1 + idx % self.dt_stride_max
                s = min(s, s_fit)
        else:
            s = 1
        if not self.ctx_random or nc is None:
            return list(range(idx, idx + (span + 1) * s, s))[:self.seq_len], s
        p = idx + nc * s
        c = int(torch.randint(idx + s + 1, (1,)).item())      # [0, idx + s]
        return ([c + k * s for k in range(nc)]
                + [p + k * s for k in range(self.seq_len - nc)]), s

    def _open_pack(self) -> None:
        self._pack = np.load(self.sim_dir / PACK_FILE, mmap_mode='r')
        m = np.load(self.sim_dir / PACK_META)
        self._pack_mean, self._pack_std = m['prim_mean'], m['prim_std']

    def _get_frame(self, i: int) -> torch.Tensor:
        if self._cache is not None:
            return self._cache[i]
        if self._packed:
            if self._pack is None:
                self._open_pack()
            j = self._pack_off + i
            vals = (np.asarray(self._pack[j], dtype=np.float32)     # type: ignore[index]
                    * self._pack_std[j] + self._pack_mean[j])       # type: ignore[index]
            return (self.renderer.render_cell_smooth(vals) - self.mean) / self.std
        d    = np.load(self.paths[i])
        vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        raw  = self.renderer.render_cell_smooth(vals)   # [C, H, W]
        return (raw - self.mean) / self.std

    @classmethod
    def with_cache(cls, sim_dir: Path, renderer: MeshRenderer, seq_len: int,
                   mean: torch.Tensor, std: torch.Tensor,
                   first_frame: int = 0,
                   mesh_id: Optional[int] = None,
                   n_context: Optional[int] = None,
                   ctx_random: bool = True,
                   settle_time: float = 0.0,
                   meta: Optional[dict] = None) -> "FVMSequenceDataset":
        """Pre-render and cache all frames in memory for fast repeated access."""
        if meta is None:
            meta = scan_run(sim_dir)
        files = [sim_dir / n for n in meta['files']][first_frame:]
        files = files[_settle_skip(files, settle_time, meta['cold']):]
        m = mean.view(-1, 1, 1)
        s = std.view(-1, 1, 1)
        cache = []
        for i, path in enumerate(files):
            try:
                d    = np.load(path)
                vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
                raw  = renderer.render_cell_smooth(vals)
            except Exception as e:
                # Stop at the first unreadable frame rather than skipping it: the
                # frames already cached are contiguous in time, and splicing across
                # a gap would silently hand the model a discontinuous sequence.
                print(f'  [warn] {sim_dir.name}: unreadable frame {path.name} '
                      f'({type(e).__name__}); keeping the first {i} frame(s)')
                files = files[:i]
                break
            cache.append((raw - m) / s)
        if len(cache) < seq_len:
            raise TruncatedRunError(
                sim_dir, f'only {len(cache)} readable frames, need {seq_len}')
        return cls(sim_dir, renderer, seq_len, mean, std, first_frame,
                   frame_cache=cache, mesh_id=mesh_id, paths_override=files,
                   n_context=n_context, ctx_random=ctx_random,
                   settle_time=settle_time, meta=meta)


# ---------------------------------------------------------------------------
# Multi-run data module (plain PyTorch, no Lightning dependency)
# ---------------------------------------------------------------------------

class FVMDataModule:
    """
    Scans a dataset directory for simulation subdirectories, builds a renderer,
    computes normalisation statistics, and exposes a DataLoader.

    Usage
    -----
    dm = FVMDataModule(data_dir, seq_len=7, batch_size=4)
    dm.setup()
    for batch in dm.train_dataloader():
        frames = [batch[:, t] for t in range(batch.shape[1])]  # list of [B,C,H,W]
    """

    STATS_FILE = 'hfm_input_stats.json'

    def __init__(
        self,
        data_dir:    Path,
        seq_len:     int,
        resolution:  tuple[int, int] = (256, 256),
        batch_size:  int = 4,
        num_workers: int = 4,
        # Off by default.  It was 20, to skip the cold-start transient of
        # run_gen's ~1000-frame trajectories.  gen_alternating segments continue
        # an already-developed field and are only 30 frames long, so there is no
        # transient to skip and a nonzero value just throws away training data.
        first_frame: int = 0,
        # data_dir may be one directory or a LIST: every root's meshes and runs
        # are pooled into one dataset (mesh_ids number across all roots, so the
        # mask table stays consistent).  Runs whose params.json carries save_t
        # (gen_alternating) train at stride 1; legacy fixed-interval runs get
        # per-sample stride augmentation 1..legacy_stride_max, covering effective
        # dt = save_t .. legacy_stride_max * save_t.
        # Fraction of RUNS held out for validation, drawn from the training corpus
        # itself so train and val share a distribution.  A separate directory of
        # old-solver runs does not: it was all save_t=0.01, i.e. the 5th percentile
        # of what training now sees, so val measured one corner of the dt axis.
        val_fraction: float = 0.05,
        # Frames the context encoder sees.  Set it to draw the context block from a
        # random causal position in the run instead of always the frames directly
        # before the prediction (see FVMSequenceDataset._window).
        n_context: Optional[int] = None,
        # Sim-time to discard from the front of COLD-START runs only (0 = keep
        # everything).  Unlike first_frame this is a physical criterion applied
        # only where there is actually a transient, so continuation segments keep
        # all 30 of their frames.
        settle_time: float = 0.0,
        # Stride-augmentation ceiling for legacy corpora (1 disables).  10 at
        # save_t=0.01 spans effective dt 0.01..0.10, meeting the alternating
        # corpus's U(0.01, 0.2) across most of its range.
        legacy_stride_max: int = 10,
        cache_frames: bool = False,
        mean: Optional[torch.Tensor] = None,
        std:  Optional[torch.Tensor] = None,
        return_mesh_id: bool = False,
    ):
        dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
        self.data_dirs    = [Path(d) for d in dirs]
        self.data_dir     = self.data_dirs[0]     # primary: stats + index home
        self.seq_len      = seq_len
        self.resolution   = resolution
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.first_frame  = first_frame
        self.cache_frames = cache_frames
        self._dataset: Optional[ConcatDataset] = None
        self.mean: Optional[torch.Tensor] = mean
        self.std:  Optional[torch.Tensor] = std
        self.return_mesh_id = return_mesh_id
        self.val_fraction = val_fraction
        self.n_context = n_context
        self.settle_time = settle_time
        self.legacy_stride_max = max(1, int(legacy_stride_max))
        self.mesh_masks: Optional[torch.Tensor] = None   # [n_mesh, 1, H, W]
        self._val_dataset: Optional[Dataset] = None

    def setup(self, recompute_stats: bool = False):
        mesh_dirs = []
        for root in self.data_dirs:
            found = mesh_dirs_for(root)
            if not found:
                raise RuntimeError(f'No shared_mesh.pkl found in {root} or its subdirectories')
            mesh_dirs.extend(found)

        runs: list[tuple[Path, MeshRenderer, int]] = []
        mesh_masks = []
        for mi, mdir in enumerate(mesh_dirs):
            renderer = build_renderer(mdir, self.resolution)
            if self.return_mesh_id:
                mesh_masks.append(load_pixel_mask(mdir, renderer, self.resolution)[0])
            sim_dirs = sorted([p for p in mdir.iterdir()
                               if p.is_dir() and p.name.startswith('run')])
            for sdir in sim_dirs:
                runs.append((sdir, renderer, mi))
        if self.return_mesh_id:
            self.mesh_masks = torch.stack(mesh_masks)     # [n_mesh, 1, H, W]

        if not runs:
            raise RuntimeError(f'No simulation subdirectories found in {self.data_dir}')

        # One cached pass over every run's metadata, instead of ~5 filesystem
        # operations per run per rank.  See load_run_index.
        _t0 = time.perf_counter()
        meta = {}
        for root in self.data_dirs:
            in_root = [d for d, _r, _mi in runs
                       if root == d.parents[1] or root == d.parents[0]]
            if in_root:
                meta.update(load_run_index(root, in_root))
        print(f'  Run index: {len(meta)} runs in {time.perf_counter() - _t0:.1f}s')

        # Load or compute normalisation stats (skip if already provided externally)
        if self.mean is None or self.std is None:
            stats_path = self.data_dir / self.STATS_FILE
            stats = None
            if stats_path.exists() and not recompute_stats:
                try:
                    stats = json.loads(stats_path.read_text())
                except Exception as e:
                    print(f'  Stats file unreadable ({type(e).__name__}: {e}), recomputing...')
            if stats is not None:
                self.mean = torch.tensor(stats['mean'])
                self.std  = torch.tensor(stats['std'])
            else:
                print('Computing normalisation stats...')
                self.mean, self.std = compute_normalisation_stats(
                    [(d, r) for d, r, _mi in runs], first_frame=self.first_frame)
                _atomic_write(stats_path, lambda tmp: tmp.write_text(json.dumps(
                    {'mean': self.mean.tolist(), 'std': self.std.tolist()})))
                print(f'Stats saved to {stats_path}')

        builder = FVMSequenceDataset.with_cache if self.cache_frames else (
            lambda *a, **kw: FVMSequenceDataset(*a, **kw)
        )
        datasets, val_datasets = [], []
        for d, renderer, mi in runs:
            try:
                is_val = is_val_run(d, self.val_fraction)
                ds = builder(
                    d, renderer, self.seq_len, self.mean, self.std, self.first_frame,
                    mesh_id=(mi if self.return_mesh_id else None),
                    n_context=self.n_context,
                    settle_time=self.settle_time,
                    meta=meta[d],
                    # Legacy fixed-interval runs get stride augmentation; runs
                    # with a real per-segment save_t train at stride 1.
                    dt_stride_max=(1 if meta[d].get('has_save_t')
                                   else self.legacy_stride_max),
                    # Validation stays DETERMINISTIC: a random context would make
                    # the val metric a different measurement every epoch, and the
                    # whole point of the ratio is that its movement means something.
                    ctx_random=not is_val)
            except TruncatedRunError as e:
                # Setup must not die on a bad run either: quarantine and carry on.
                quarantine_run(e.sim_dir, e.reason)
                continue
            (val_datasets if is_val else datasets).append(ds)
        n_short = sum(1 for ds in datasets if len(ds) == 0)
        datasets = [ds for ds in datasets if len(ds) > 0]
        val_datasets = [ds for ds in val_datasets if len(ds) > 0]
        if not datasets:
            raise RuntimeError(
                f'No usable sequences found: all {n_short} run(s) hold fewer than '
                f'seq_len={self.seq_len} frames after first_frame={self.first_frame}. '
                'gen_alternating writes short segments (30 frames by default), so '
                'lower max(time_strides), rollout_horizon or first_frame.')
        if n_short:
            # Loud, because a silent drop looks identical to a smaller dataset:
            # one over-large stride can discard most of the corpus unnoticed.
            print(f'  [WARN] dropped {n_short}/{n_short + len(datasets)} run(s) with '
                  f'fewer than seq_len={self.seq_len} usable frames')

        # z-score the per-run BC summaries across runs so the probe regression
        # target is unit-scale.  Runs with no readable BCs keep NaN targets, which
        # the probe loss masks per sample.
        # Fitted on the TRAIN runs only, then applied to both: fitting across the
        # validation runs too would leak their statistics into the training target.
        bcs = [ds.bc_raw for ds in datasets if ds.bc_raw is not None
               and bool(torch.isfinite(ds.bc_raw).all())]
        if bcs:
            B = torch.stack(bcs)
            self.bc_mean, self.bc_std = B.mean(0), B.std(0)
            for ds in datasets + val_datasets:
                ds.set_bc_norm(self.bc_mean, self.bc_std)
            self.bc_dim = B.shape[1]
        else:
            self.bc_mean = self.bc_std = None
            self.bc_dim = 0

        # Same treatment for the physics context, fitted on TRAIN runs only.
        cxs = [ds.ctx_raw for ds in datasets if ds.ctx_raw is not None
               and bool(torch.isfinite(ds.ctx_raw).all())]
        if cxs:
            C = torch.stack(cxs)
            self.ctx_mean, self.ctx_std = C.mean(0), C.std(0)
            for ds in datasets + val_datasets:
                ds.set_ctx_norm(self.ctx_mean, self.ctx_std)
            self.ctx_dim = C.shape[1]
        else:
            self.ctx_mean = self.ctx_std = None
            self.ctx_dim = 0
        # Viscosity-model classes actually present, so the head is not sized for
        # choices this dataset never uses.
        seen_cls = {ds.ctx_cls for ds in datasets if ds.ctx_cls >= 0}
        self.n_visc_models = len(CTX_MODEL_CHOICES) if seen_cls else 0

        # Balance the two corpus types so training genuinely alternates between
        # them.  Legacy runs are ~700 frames against alternating's ~31, so raw
        # window counts would let the legacy corpus drown out the alternating one
        # ~20:1.  Every k-th window of each legacy run is kept (windows overlap by
        # seq_len-1 frames, so thinning them loses variety, not coverage), which
        # brings the two types to roughly equal draw probability under shuffle.
        legacy = [ds for ds in datasets if ds.dt_stride_max > 1]
        modern = [ds for ds in datasets if ds.dt_stride_max == 1]
        n_leg, n_mod = sum(len(d) for d in legacy), sum(len(d) for d in modern)
        train_parts: list = list(datasets)
        if legacy and modern and n_leg > n_mod:
            k = max(1, round(n_leg / n_mod))
            train_parts = modern + [
                torch.utils.data.Subset(ds, range(0, len(ds), k)) for ds in legacy]
            n_leg = sum(len(t) for t in train_parts) - n_mod
            print(f'  corpus balance: legacy thinned x{k} -> '
                  f'{n_leg} legacy vs {n_mod} alternating sequences')
        elif legacy:
            print(f'  corpora: {n_leg} legacy (stride-augmented 1..'
                  f'{self.legacy_stride_max}) / {n_mod} alternating sequences')

        self._dataset = ResilientConcat(ConcatDataset(train_parts))
        self._val_dataset = (ResilientConcat(ConcatDataset(val_datasets))
                             if val_datasets else None)
        # print the MESH count too: without it a multi-mesh run looks identical to a
        # single-mesh one, which is how the shared-mask bug stayed invisible.
        print(f'Dataset ready: {len(self._dataset)} sequences across {len(datasets)} '
              f'usable runs (of {len(runs)} found) / {len(mesh_dirs)} mesh(es)')
        # Frame-interval spread across runs: with gen_alternating this should span
        # roughly 0.01..0.2, and a single value means the dataset predates save_t
        # (or params.json is missing), so the dt probe has nothing to regress.
        st = sorted(ds.save_t for ds in datasets)
        self.save_t_min, self.save_t_max = st[0], st[-1]
        print(f'  save_t: min {st[0]:.4g}  median {st[len(st)//2]:.4g}  max {st[-1]:.4g}'
              + ('   [WARN] constant — dt probe target is degenerate'
                 if st[-1] - st[0] < 1e-9 else ''))
        if self.ctx_dim:
            print(f'  physics context: {self.ctx_dim} continuous params '
                  f'+ {len(seen_cls)}/{len(CTX_MODEL_CHOICES)} viscosity models present')
        else:
            print('  [WARN] no per-segment physics context in params.json — the '
                  'context head is disabled (dataset predates gen_alternating?)')
        if self._val_dataset is not None:
            vt = sorted(ds.save_t for ds in val_datasets)
            print(f'  val split: {len(self._val_dataset)} sequences from '
                  f'{len(val_datasets)} held-out runs '
                  f'({100 * len(val_datasets) / (len(datasets) + len(val_datasets)):.1f}% '
                  f'of runs), save_t median {vt[len(vt)//2]:.4g}')
        if len(mesh_dirs) > 1 and not self.return_mesh_id:
            print(f'  [WARN] {len(mesh_dirs)} geometries but return_mesh_id=False -- the '
                  f'caller will apply ONE mask to all of them, which is wrong for '
                  f'{len(mesh_dirs)-1} of them.')

    def train_dataloader(self) -> DataLoader:
        assert self._dataset is not None, 'Call setup() first'
        # Under DDP each rank must see a DISJOINT shard of the data, otherwise every
        # rank trains on the same samples and the gradient averaging buys nothing.
        # sampler and shuffle are mutually exclusive, so shuffling moves into the
        # sampler (train.py calls set_epoch each epoch to reshuffle).
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self._dataset, shuffle=True, drop_last=True)
        return DataLoader(
            self._dataset,
            batch_size         = self.batch_size,
            shuffle            = (sampler is None),
            sampler            = sampler,
            num_workers        = self.num_workers,
            pin_memory         = True,
            persistent_workers = self.num_workers > 0,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """The held-out runs.  None when val_fraction is 0."""
        assert self._dataset is not None, 'Call setup() first'
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = True,
            persistent_workers = self.num_workers > 0,
        )
