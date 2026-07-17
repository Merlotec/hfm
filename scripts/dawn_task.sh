#!/bin/bash -l
# Per-rank launcher for the Dawn job — invoked by `srun` from dawn_train.slurm.
#
# The oneAPI modules are loaded HERE, inside each task, NOT in the batch script.

# ---- oneAPI runtime: PIP, not modules --------------------------------------
# There are two coherent ways to get the SYCL/UR/MKL/CCL runtime, and MIXING THEM
# IS WHAT BREAKS:
#   (A) all-pip     — the `torch ... --index-url .../whl/xpu` wheel bundles its own
#                     oneAPI runtime into the venv (libsycl.so.8, libur_loader, MKL).
#   (B) all-module  — module-provided oneAPI + a torch built against that version.
#
# We use (A).  Loading intel-oneapi-compilers puts Dawn's 2025.0.3 runtime AHEAD of
# the venv on LD_LIBRARY_PATH, so the wheel's newer libsycl.so.8 resolves against
# the module's older libur_loader.so.0 and import torch dies with:
#   ImportError: .../libur_loader.so.0: version `LIBUR_LOADER_0.11' not found
# So: load ONLY the base/driver module (level-zero comes from the system driver),
# and let the venv supply everything else.  Do not add intel-oneapi-{compilers,mkl,
# ccl} back unless you also switch to strategy (B) wholesale.
module purge
module load default-dawn                       # base env + GPU (level-zero) driver

# venv with the XPU build of torch (must be e.g. 2.8.0+xpu — NOT +cu128):
source "$HOME/rds/hpc-work/venvs/flsim-xpu/bin/activate"   # EDIT to your env

# Make the venv's bundled oneAPI runtime win over anything the base module adds.
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib64:${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"

# ---- XPU / oneCCL configuration --------------------------------------------
export ZE_FLAT_DEVICE_HIERARCHY=FLAT           # expose each PVC tile as its own XPU
                                               # (2 tiles/card -> xpu:0..7 per node)
export CCL_ZE_IPC_EXCHANGE=sockets             # robust IPC handle exchange on SLURM
export CCL_ATL_TRANSPORT=ofi                   # oneCCL over libfabric (srun launch)

# oneCCL spawns worker THREADS and pins them to cores.  Under SLURM each rank is
# confined to a cgroup cpuset; if oneCCL picks a core outside it, the pin fails and
# you get:  oneCCL: exec.cpp:122 start_workers: EXCEPTION: failed to start worker # 0
# Give it exactly one worker and reserve a core for it by leaving OMP one short,
# then pin that worker to a core we know is inside this rank's allocation.
export CCL_WORKER_COUNT=1
_CPT="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS=$(( _CPT > 1 ? _CPT - 1 : 1 ))   # leave 1 core for the CCL worker
# Last core of THIS rank's slice (ranks are laid out contiguously by local id).
_LOCALID="${SLURM_LOCALID:-0}"
export CCL_WORKER_AFFINITY=$(( _LOCALID * _CPT + _CPT - 1 ))

# oneCCL bring-up diagnostics.  `failed to start worker # 0` is a generic message;
# this prints the actual reason.  Set CCL_DEBUG=0 in the environment to silence.
export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-info}"

# Backend override: torch>=2.7 with an XPU build ships a NATIVE 'xccl' backend that
# does not go through oneccl_bindings_for_pytorch at all.  Set HFM_DDP_BACKEND=xccl
# to force it (hfm/distributed.py auto-detects, but only if is_xccl_available()).
# export HFM_DDP_BACKEND=xccl

# ---- fail fast, with the reason, instead of silently training on CPU --------
# rank 0 prints what torch can actually see; init_distributed() hard-errors if a
# multi-rank job resolves to CPU (override: HFM_ALLOW_CPU=1).
if [ "${SLURM_PROCID:-0}" = "0" ]; then
  echo "=== rank0 env ==="
  echo "  nodes=${SLURM_NNODES:-?} ntasks=${SLURM_NTASKS:-?} localid=${SLURM_LOCALID:-?} cpus/task=${SLURM_CPUS_PER_TASK:-?}"
  echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS CCL_WORKER_COUNT=$CCL_WORKER_COUNT CCL_WORKER_AFFINITY=$CCL_WORKER_AFFINITY"
  echo "  CCL_ATL_TRANSPORT=$CCL_ATL_TRANSPORT HFM_DDP_BACKEND=${HFM_DDP_BACKEND:-<auto>}"
  echo "  cpus visible to this rank: $(nproc) | affinity: $(taskset -pc $$ 2>/dev/null || echo n/a)"
  python - <<'PY' || true
import torch, torch.distributed as d
print('  torch', torch.__version__,
      '| xpu', torch.xpu.is_available() if hasattr(torch, 'xpu') else 'ABSENT',
      '| count', torch.xpu.device_count() if hasattr(torch, 'xpu') else 0)
for b in ('xccl', 'ccl', 'mpi', 'gloo'):
    fn = getattr(d, f'is_{b}_available', None)
    print(f'   backend {b:5s}: ', fn() if fn else 'no probe')
PY
  echo "================="
fi

# NOTE: we do NOT use Lightning on Dawn.  Lightning's accelerator registry is
# {cpu, cuda, mps, tpu} — there is no XPU accelerator — so accelerator='auto'
# silently falls back to CPU on a PVC node (and 'bf16-true' then dies on a
# fp32-input/bf16-weight mismatch).  scripts/train.py uses raw torch DDP via
# hfm/distributed.py, which selects xpu:<SLURM_LOCALID> and uses the native
# 'xccl' backend (torch>=2.7) or oneCCL 'ccl'.  Rank/size come from SLURM_* vars.
cd "$SLURM_SUBMIT_DIR"
exec python scripts/train.py
