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
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
except Exception:
    ipex = None
try:
    import oneccl_bindings_for_pytorch  # noqa: F401  (registers the 'ccl' backend)
except Exception:
    pass


def _has_xpu() -> bool:
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def _env_int(*names, default: int = 0) -> int:
    for n in names:
        if n in os.environ:
            return int(os.environ[n])
    return default


def pick_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    if _has_xpu():
        return torch.device(f'xpu:{local_rank}')
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
        return 'ccl'          # oneCCL via oneccl_bindings_for_pytorch (torch-ccl)
    return 'gloo'


def init_distributed():
    """Initialise (or no-op) the process group.  Returns (rank, world, local, device)."""
    rank = _env_int('RANK', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'SLURM_PROCID', default=0)
    world = _env_int('WORLD_SIZE', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'SLURM_NTASKS', default=1)
    local = _env_int('LOCAL_RANK', 'MPI_LOCALRANKID', 'PALS_LOCAL_RANKID',
                     'OMPI_COMM_WORLD_LOCAL_RANK', 'SLURM_LOCALID', default=0)
    device = pick_device(local)

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
