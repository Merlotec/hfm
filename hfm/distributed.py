"""Multi-device / multi-node helpers.

Ported from NOMAD (nomad/nomad/distributed.py), which already runs on Dawn.
Lightning cannot be used on Dawn: its accelerator registry is
{cpu, cuda, mps, tpu} — there is no XPU accelerator — so `accelerator='auto'`
silently selects CPU on a PVC node.  This module drives raw torch DDP instead
and selects the device explicitly.

Supports NVIDIA (CUDA / NCCL), Intel Data Center GPU Max — the "1550" / Ponte
Vecchio — (XPU / CCL), Apple MPS, and CPU.  HFM is pure PyTorch with no
vendor-specific ops, so only *device selection* and the *distributed backend*
differ across clusters.

Launch is env-driven and covers torchrun, MPI (Intel/Open MPI) and plain srun:
rank/size/local-rank are read from whichever variables the launcher set.  On a
single process (world_size == 1) nothing distributed is initialised, so local
runs are unaffected.
"""
from __future__ import annotations

import os

import torch

# Importing these registers the `xpu` device and the `ccl` distributed backend.
# Guarded so nothing breaks on CUDA / MPS / CPU machines that lack the Intel stack.
# The failure reason is KEPT: silently swallowing it is how a job ends up training
# on CPU for 24h on a GPU allocation with no clue why (see xpu_status()).
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
    _IPEX_ERR = None
except Exception as e:                                     # pragma: no cover
    ipex = None
    _IPEX_ERR = e
try:
    import oneccl_bindings_for_pytorch  # noqa: F401  (registers the 'ccl' backend)
    _CCL_ERR = None
except Exception as e:                                     # pragma: no cover
    _CCL_ERR = e


def _has_xpu() -> bool:
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def xpu_status() -> str:
    """Why XPU is (or is not) usable — printed when a multi-rank job lands on CPU."""
    lines = [f'torch={torch.__version__}']
    lines.append(f'intel_extension_for_pytorch: '
                 + (f'OK {getattr(ipex, "__version__", "")}' if ipex is not None
                    else f'IMPORT FAILED -> {type(_IPEX_ERR).__name__}: {_IPEX_ERR}'))
    lines.append('oneccl_bindings_for_pytorch: '
                 + ('OK' if _CCL_ERR is None
                    else f'IMPORT FAILED -> {type(_CCL_ERR).__name__}: {_CCL_ERR}'))
    if hasattr(torch, 'xpu'):
        try:
            lines.append(f'torch.xpu.is_available()={torch.xpu.is_available()} '
                         f'device_count={torch.xpu.device_count()}')
        except Exception as e:
            lines.append(f'torch.xpu probe raised {type(e).__name__}: {e}')
    else:
        lines.append('torch.xpu attribute ABSENT (this torch build has no XPU support)')
    return '\n  '.join(lines)


def _env_int(*names, default: int = 0) -> int:
    for n in names:
        if n in os.environ:
            return int(os.environ[n])
    return default


def pick_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    if _has_xpu():
        # Under ZE_AFFINITY_MASK each rank sees exactly ONE tile, always index 0 —
        # indexing by local_rank would be out of range for every rank but 0.
        n = torch.xpu.device_count()
        return torch.device(f'xpu:{local_rank if local_rank < n else 0}')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _ddp_backend(device: torch.device) -> str:
    override = os.environ.get('HFM_DDP_BACKEND')
    if override:
        return override
    if device.type == 'cuda':
        return 'nccl'
    if device.type == 'xpu':
        # torch >= 2.7 ships a native XCCL backend for XPU — no oneccl_bindings
        # needed.  Fall back to the 'ccl' bindings on older stacks.
        try:
            if torch.distributed.is_xccl_available():   # type: ignore[attr-defined]
                return 'xccl'
        except Exception:
            pass
        return 'ccl'          # oneCCL via oneccl_bindings_for_pytorch (torch-ccl)
    return 'gloo'


def init_distributed():
    """Initialise (or no-op) the process group.  Returns (rank, world, local, device)."""
    rank = _env_int('RANK', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'SLURM_PROCID', default=0)
    world = _env_int('WORLD_SIZE', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'SLURM_NTASKS', default=1)
    local = _env_int('LOCAL_RANK', 'MPI_LOCALRANKID', 'PALS_LOCAL_RANKID',
                     'OMPI_COMM_WORLD_LOCAL_RANK', 'SLURM_LOCALID', default=0)
    device = pick_device(local)

    # A multi-rank job on CPU is almost always a broken accelerator env, not intent:
    # it means hours of cluster time at ~1% throughput.  Refuse, and say why.
    # Set HFM_ALLOW_CPU=1 for a deliberate CPU-parallel run.
    if world > 1 and device.type == 'cpu' and os.environ.get('HFM_ALLOW_CPU') != '1':
        raise RuntimeError(
            f'Distributed run (world_size={world}) resolved to device=cpu — no '
            f'accelerator was found, so this would train on CPU.\n'
            f'  {xpu_status()}\n'
            'Fix the environment (activate the XPU venv / `pip install '
            'intel_extension_for_pytorch oneccl_bind_pt`), or set HFM_ALLOW_CPU=1 '
            'to run on CPU deliberately.  Run scripts/check_xpu.py for detail.'
        )

    if world > 1:
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world)
        os.environ['LOCAL_RANK'] = str(local)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        elif device.type == 'xpu':
            torch.xpu.set_device(device)
        torch.distributed.init_process_group(
            backend=_ddp_backend(device), rank=rank, world_size=world)
    return rank, world, local, device


def wrap_ddp(module: torch.nn.Module, device: torch.device,
             find_unused_parameters: bool = False) -> torch.nn.Module:
    """DistributedDataParallel wrap (no-op if not distributed).

    Wrapping reuses the same Parameter objects, so an optimizer built over the
    module *before* wrapping stays valid — DDP just adds gradient-averaging hooks.

    find_unused_parameters is needed for the GAN path: the generator step does not
    touch every discriminator parameter (and vice versa), so the reducer would
    otherwise wait forever for grads that never arrive.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return module
    if device.type in ('cuda', 'xpu'):
        return torch.nn.parallel.DistributedDataParallel(
            module, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=find_unused_parameters)
    return torch.nn.parallel.DistributedDataParallel(
        module, find_unused_parameters=find_unused_parameters)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """The underlying module, whether or not it is DDP-wrapped."""
    return getattr(module, 'module', module)


def host_grad_sync_enabled() -> bool:
    """Manual host-side gradient averaging instead of DDP (HFM_HOST_GRAD_SYNC=1).

    For stacks where NO device collective backend works (Dawn: oneCCL's ZE path
    faults on this driver, and XCCL cannot run with ZE disabled — 'ze_data was
    not initialized').  The process group is gloo over host memory; compute stays
    on the XPU; gradients cross rank boundaries via allreduce_grads() below."""
    return os.environ.get('HFM_HOST_GRAD_SYNC') == '1'


def allreduce_grads(modules) -> None:
    """Average .grad across ranks on the HOST (works with a gloo process group).

    Call between loss.backward() and optimizer.step().  No-op when not
    distributed.  All ranks must call this with the same modules and the same
    set of grad-bearing params (keep any conditional backward globally
    consistent — see allreduce_stats).  ~4B/param of host traffic per call.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    world = torch.distributed.get_world_size()
    grads = [p.grad for m in modules for p in m.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.detach().reshape(-1).to('cpu', torch.float32) for g in grads])
    torch.distributed.all_reduce(flat)
    flat.div_(world)
    off = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[off:off + n].view_as(g).to(g.device, g.dtype))
        off += n


def allreduce_stats(*vals: float):
    """Sum a few scalars across ranks (host-side).  Returns the summed list, or
    the inputs unchanged when not distributed.  Use it to make control-flow
    decisions (health gates, NaN bails) IDENTICAL on every rank — a per-rank
    branch around a backward/step desyncs both DDP and manual grad averaging."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return list(vals)
    t = torch.tensor(vals, dtype=torch.float64)
    torch.distributed.all_reduce(t)
    return t.tolist()


def is_main() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def cleanup():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
