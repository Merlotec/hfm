"""
FVM dataset for HFM training.

Each item is a [T, C, H, W] tensor of T consecutive normalised, rendered frames
from one simulation run.  T must be >= n_warmup_frames + 2 so the trainer has
enough frames for warmup + one training step.

The renderer (MeshRenderer) converts FVM cell-level primitives to a smooth
pixel grid via barycentric interpolation.  It is built once per dataset and
cached to disk alongside the data as renderer_cache_{H}x{W}.pt.
"""

import json
import os
import sys
import pickle
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
        cache.unlink()

    renderer = MeshRenderer(
        verts,
        fvm_mesh.cells.cpu().numpy(),
        resolution=resolution,
        device=device,
    )
    renderer.save_cache(str(cache))
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
        m = torch.load(cache, weights_only=True)
        if (m.shape == (1, 1, H, W)
                and int(m.sum()) == n_interior
                and m.view(-1)[renderer._interior_idx].all()):
            return m
        print('  Pixel mask stale (renderer mismatch), rebuilding...')
        cache.unlink()
    mask = build_pixel_mask(renderer, resolution)
    torch.save(mask, cache)
    print(f'  Pixel mask saved — {mask.sum().item()} fluid / {mask.numel()} total pixels')
    return mask


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_normalisation_stats(
    runs: list[tuple[Path, MeshRenderer]],
    n_samples: int = 300,
    first_frame: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-channel mean and std by sampling frames across all runs."""
    all_files: list[tuple[Path, MeshRenderer]] = []
    for d, renderer in runs:
        all_files.extend([(f, renderer) for f in sorted(
            [f for f in d.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
            key=lambda f: float(f.stem[2:]),
        )[first_frame:]])

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
# Single-run dataset
# ---------------------------------------------------------------------------

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
        first_frame: int = 20,
        frame_cache: Optional[list[torch.Tensor]] = None,
        mesh_id:     Optional[int] = None,
    ):
        files = sorted(
            [f for f in sim_dir.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
            key=lambda f: float(f.stem[2:]),
        )[first_frame:]
        self.paths      = files
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
                frames = torch.stack(
                    [self._get_frame(idx + i) for i in range(self.seq_len)])
                if self.mesh_id is not None:
                    return frames, self.mesh_id  # + which geometry this sample is
                return frames   # [T, C, H, W]
            except Exception as e:                       # unreadable/corrupt frame
                if not FVMSequenceDataset._warned_corrupt:
                    bad = self.paths[min(idx, len(self.paths) - 1)]
                    print(f'  [warn] unreadable frame near {bad}: {type(e).__name__}: {e}')
                    print('         Skipping this sample.  Find bad files with:')
                    print('         python scripts/check_frames.py <data_dir>')
                    FVMSequenceDataset._warned_corrupt = True
                idx = (idx + self.seq_len) % n
        raise RuntimeError(
            f'{self.paths[0].parent}: could not read a valid sequence after 4 attempts '
            '— this run is likely truncated; remove or repair it.')

    def _get_frame(self, i: int) -> torch.Tensor:
        if self._cache is not None:
            return self._cache[i]
        d    = np.load(self.paths[i])
        vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        raw  = self.renderer.render_cell_smooth(vals)   # [C, H, W]
        return (raw - self.mean) / self.std

    @classmethod
    def with_cache(cls, sim_dir: Path, renderer: MeshRenderer, seq_len: int,
                   mean: torch.Tensor, std: torch.Tensor,
                   first_frame: int = 20,
                   mesh_id: Optional[int] = None) -> "FVMSequenceDataset":
        """Pre-render and cache all frames in memory for fast repeated access."""
        files = sorted(
            [f for f in sim_dir.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
            key=lambda f: float(f.stem[2:]),
        )[first_frame:]
        m = mean.view(-1, 1, 1)
        s = std.view(-1, 1, 1)
        cache = []
        for path in files:
            d    = np.load(path)
            vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
            raw  = renderer.render_cell_smooth(vals)
            cache.append((raw - m) / s)
        return cls(sim_dir, renderer, seq_len, mean, std, first_frame, frame_cache=cache)


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
        first_frame: int = 20,
        cache_frames: bool = False,
        mean: Optional[torch.Tensor] = None,
        std:  Optional[torch.Tensor] = None,
        return_mesh_id: bool = False,
    ):
        self.data_dir     = Path(data_dir)
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
        self.mesh_masks: Optional[torch.Tensor] = None   # [n_mesh, 1, H, W]

    def setup(self, recompute_stats: bool = False):
        mesh_dirs = mesh_dirs_for(self.data_dir)
        if not mesh_dirs:
            raise RuntimeError(f'No shared_mesh.pkl found in {self.data_dir} or its subdirectories')

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

        # Load or compute normalisation stats (skip if already provided externally)
        if self.mean is None or self.std is None:
            stats_path = self.data_dir / self.STATS_FILE
            if stats_path.exists() and not recompute_stats:
                with open(stats_path) as f:
                    s = json.load(f)
                self.mean = torch.tensor(s['mean'])
                self.std  = torch.tensor(s['std'])
            else:
                print('Computing normalisation stats...')
                self.mean, self.std = compute_normalisation_stats(
                    [(d, r) for d, r, _mi in runs], first_frame=self.first_frame)
                with open(stats_path, 'w') as f:
                    json.dump({'mean': self.mean.tolist(), 'std': self.std.tolist()}, f)
                print(f'Stats saved to {stats_path}')

        builder = FVMSequenceDataset.with_cache if self.cache_frames else (
            lambda *a, **kw: FVMSequenceDataset(*a, **kw)
        )
        datasets = [
            builder(d, renderer, self.seq_len, self.mean, self.std, self.first_frame,
                    mesh_id=(mi if self.return_mesh_id else None))
            for d, renderer, mi in runs
        ]
        datasets = [ds for ds in datasets if len(ds) > 0]
        if not datasets:
            raise RuntimeError('No usable sequences found — try reducing seq_len or first_frame')
        self._dataset = ConcatDataset(datasets)
        # print the MESH count too: without it a multi-mesh run looks identical to a
        # single-mesh one, which is how the shared-mask bug stayed invisible.
        print(f'Dataset ready: {len(self._dataset)} sequences across {len(runs)} runs '
              f'/ {len(mesh_dirs)} mesh(es)')
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

    def val_dataloader(self) -> DataLoader:
        assert self._dataset is not None, 'Call setup() first'
        return DataLoader(
            self._dataset,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = True,
            persistent_workers = self.num_workers > 0,
        )
