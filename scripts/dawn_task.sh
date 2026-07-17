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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ---- fail fast, with the reason, instead of silently training on CPU --------
# rank 0 prints what torch can actually see; init_distributed() hard-errors if a
# multi-rank job resolves to CPU (override: HFM_ALLOW_CPU=1).
if [ "${SLURM_PROCID:-0}" = "0" ]; then
  python -c "import torch; print('torch', torch.__version__, '| xpu',
        torch.xpu.is_available() if hasattr(torch,'xpu') else 'ABSENT')" || true
fi

# NOTE: we do NOT use Lightning on Dawn.  Lightning's accelerator registry is
# {cpu, cuda, mps, tpu} — there is no XPU accelerator — so accelerator='auto'
# silently falls back to CPU on a PVC node (and 'bf16-true' then dies on a
# fp32-input/bf16-weight mismatch).  scripts/train.py uses raw torch DDP via
# hfm/distributed.py, which selects xpu:<SLURM_LOCALID> and uses the native
# 'xccl' backend (torch>=2.7) or oneCCL 'ccl'.  Rank/size come from SLURM_* vars.
cd "$SLURM_SUBMIT_DIR"
exec python scripts/train.py
