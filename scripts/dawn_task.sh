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
# NOTE: ZE_AFFINITY_MASK (one tile per rank) was tried and made things WORSE —
# host SIGSEGV in the DDP-wrap allgather, i.e. it crashed EARLIER than the
# unmasked layout.  Leave it unset: all ranks see all 8 tiles, each computes on
# xpu:<SLURM_LOCALID>, and collectives are host-staged (CCL_ZE_ENABLE=0 below),
# which sidesteps the peer-mapping faults entirely.
export CCL_ZE_IPC_EXCHANGE=sockets             # robust IPC handle exchange on SLURM
export CCL_ATL_TRANSPORT=ofi                   # oneCCL over libfabric (srun launch)

# libfabric provider.  We deliberately do NOT load the oneAPI modules (see above),
# so FI_PROVIDER_PATH points at the PIP-BUNDLED providers in the venv, which ship
# little more than tcp/shm.  oneCCL therefore selects the *tcp* provider, and
# multi-node allreduce over tcp falls over under load with:
#   atl_ofi.cpp: prov_ep_handle_cq_err: fi_cq_readerr: err: 265,
#                prov_err: Resource temporarily unavailable(11)     <- EAGAIN
#   recv_reduce_entry.hpp:88 update: RECV_REDUCE entry failed
# Best fix is to not cross the network at all: run 8 ranks on ONE node so every
# collective stays intra-node (shm / Xe Link).  If you must go multi-node, point
# FI_PROVIDER_PATH at the SYSTEM libfabric (which has verbs/psm3 for Dawn's fabric)
# and set FI_PROVIDER accordingly — check `fi_info -l` on a compute node first.
# export FI_PROVIDER=tcp                       # explicit; slow but functional
# export FI_PROVIDER_PATH=/usr/lib64/libfabric # system providers, if available

# oneCCL spawns worker THREADS and pins them to cores.  If it pins to a core outside
# this rank's cgroup cpuset, pthread_create fails with EINVAL(22) and you get:
#   CCL_ERROR| base_thread.cpp:22 start: pthread_create returns 22
#   oneCCL: exec.cpp:122 start_workers: failed to start worker # 0
#
# Do NOT compute the core id from SLURM_LOCALID/CPUS_PER_TASK: Dawn does not hand out
# contiguous blocks.  A real rank's mask looks like
#   0,2,4,6,8,10,12,14,16,18,20,22          (even cores only — HT siblings)
# so arithmetic like localid*cpus+cpus-1 lands on an odd core that is NOT in the set.
# oneCCL's default ('auto') is just as wrong — it picks high cores like 95.
# Ask the kernel instead, and give EACH local rank a DISTINCT worker core.
export CCL_WORKER_COUNT=1
_NCORES=$(python -c 'import os; print(len(os.sched_getaffinity(0)))')
_LOCALN="${SLURM_NTASKS_PER_NODE:-${SLURM_NTASKS:-1}}"
# oneCCL parses CCL_WORKER_AFFINITY as a NODE-WIDE list of length
# local_proc_count * CCL_WORKER_COUNT and gives local process i slot i.  A bare
# integer parses only when local_proc_count == 1; with 8 ranks/node it dies with
#   env.cpp:1363 env_2_worker_affinity: failed to parse worker affinity
# We build the full list from THIS rank's own cpuset: slot j -> the (j+1)-th core
# counting DOWN from the top.  Every slot is inside our cpuset (so the slot we read
# never triggers EINVAL(22)); and when ranks SHARE a cpuset — the case that made the
# old "fill every slot with our one last core" collapse all 8 workers onto core 22 —
# the per-rank slots now land on different cores.  Workers take the HIGH cores; the
# DataLoader / OMP get the low ones.
export CCL_WORKER_AFFINITY=$(python - "$_LOCALN" <<'PY'
import os, sys
cores = sorted(os.sched_getaffinity(0)); n = len(cores)
N = int(sys.argv[1])
print(','.join(str(cores[-(1 + (j % n))]) for j in range(N)))
PY
)
# Compute runs on the XPU; host cores mostly feed the DataLoader (num_workers in
# hyperparams.json — renderer processes share this cpuset) and drive the host-staged
# allreduce (a threaded cat/div/copy that does benefit from a few cores).  Give the
# main process a modest OMP pool without swamping the loaders or the CCL worker.
# Tunable: dial back if `nproc` per rank is small, or raise with more cpus-per-task.
_OMP=$(( _NCORES > 2 ? _NCORES - 2 : 1 ))
export OMP_NUM_THREADS=$(( _OMP > 6 ? 6 : _OMP ))

# --- oneCCL topology checks: LEAVE ON with 8 ranks/node -----------------------
# History, because every one of these lines was paid for in failed jobs:
#  * 1 rank/node: topo discovery segfaulted in build_fabric_connectivity_matrix
#    (a degenerate layout with no local peers), so we disabled the port/fabric
#    checks to get past it.
#  * 8 ranks/node WITH those checks disabled: first allreduce ran direct SYCL
#    peer reads over links oneCCL never verified -> GPU page fault
#    ("Segmentation fault from GPU ... NotPresent (PDE)") in the compute runtime.
# 8 ranks/node is the layout the topology code expects; let it actually probe.
#
# COLLECTIVES: NO oneCCL AT ALL.  Every configuration of oneCCL's ZE path fails
# on this driver (compute-runtime 25.18):
#   * topo checks off            -> GPU PDE page fault, first backward allreduce
#   * topo checks on             -> same GPU PDE page fault
#   * + ZE_AFFINITY_MASK         -> host SIGSEGV in the DDP-wrap allgather
#   * CCL_ZE_ENABLE=0            -> "ze_data was not initialized": XCCL cannot
#                                   build a device communicator without ZE, so
#                                   oneCCL is simply unusable here.
# Bypass the layer entirely: gloo process group on the HOST + manual gradient
# averaging (hfm/distributed.py: allreduce_grads, called in train_step_gan).
# Compute stays on the XPUs; only gradients cross rank boundaries, D2H -> gloo
# shm allreduce -> H2D (~170MB/step for this model => tens of ms).
# The CCL_* variables above are now inert; kept for the day oneCCL works again.
# Still worth a Dawn support ticket: "torch 2.8+xpu native XCCL on pvc9,
# 8 ranks/node: GPU NotPresent(PDE) fault on first allreduce; SIGSEGV in comm
# init under ZE_AFFINITY_MASK — is there a supported oneCCL config?"
#
# RE-TEST RESULT (2026-08-09, compute-runtime 25.18.33578.77): native xccl with
# ZE enabled was cleanly tried and FAILS with the same GPU fault as oneCCL:
#   Segmentation fault from GPU ... type: 0 (NotPresent), level: 1 (PDE)
#   at drm_neo.cpp:288, on the FIRST reducer allreduce inside backward.
# Everything before it worked (init via cpu:gloo,xpu:xccl, DDP param broadcast,
# forwards, host-side collectives), so the fault is in the driver/compute
# runtime, not in any one collectives library — every route to device
# collectives on this driver hits it.  Host-staged gloo stays the floor.
# Do NOT re-run HFM_TRY_XCCL=1 until the compute runtime moves past 25.18
# (check: clinfo | grep -i driver, or the loaded intel module version).  The
# support-ticket ask is now sharper: "GPU PDE NotPresent fault on first device
# allreduce, reproduced on BOTH oneCCL-ZE and torch-native xccl, 8 ranks/node,
# torch 2.8+xpu, compute-runtime 25.18.33578.77 — is a runtime fix available?"
if [ "${HFM_TRY_XCCL:-0}" = "1" ]; then
  export HFM_DDP_BACKEND=xccl
  unset  HFM_HOST_GRAD_SYNC              # DDP device collectives, not host staging
  echo "[collective] HFM_TRY_XCCL=1 -> native xccl device collectives (ZE enabled)"
else
  export HFM_DDP_BACKEND=gloo
  export HFM_HOST_GRAD_SYNC=1
fi

# oneCCL bring-up diagnostics.  `failed to start worker # 0` is a generic message;
# this prints the actual reason.  Set CCL_LOG_LEVEL=warn once it works — info is
# hundreds of lines per rank.
export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-info}"

# ---- fail fast, with the reason, instead of silently training on CPU --------
# rank 0 prints what torch can actually see; init_distributed() hard-errors if a
# multi-rank job resolves to CPU (override: HFM_ALLOW_CPU=1).
if [ "${SLURM_PROCID:-0}" = "0" ]; then
  echo "=== rank0 env ==="
  echo "  nodes=${SLURM_NNODES:-?} ntasks=${SLURM_NTASKS:-?} localid=${SLURM_LOCALID:-?} cpus/task=${SLURM_CPUS_PER_TASK:-?}"
  echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS CCL_WORKER_COUNT=$CCL_WORKER_COUNT CCL_WORKER_AFFINITY=$CCL_WORKER_AFFINITY"
  echo "  CCL_ATL_TRANSPORT=$CCL_ATL_TRANSPORT HFM_DDP_BACKEND=${HFM_DDP_BACKEND:-<auto>} HFM_HOST_GRAD_SYNC=${HFM_HOST_GRAD_SYNC:-0}"
  echo "  tuning: HFM_TRY_XCCL=${HFM_TRY_XCCL:-0} HFM_ACCUM_STEPS=${HFM_ACCUM_STEPS:-<hp>} HFM_GRAD_ALLREDUCE_BF16=${HFM_GRAD_ALLREDUCE_BF16:-<hp>}"
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
