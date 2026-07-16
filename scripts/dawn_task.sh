#!/bin/bash -l
# Per-rank launcher for the Dawn job — invoked by `srun` from dawn_train.slurm.
#
# The oneAPI modules are loaded HERE, inside each task, NOT in the batch script.

module purge
module load default-dawn                       # Dawn base environment
module load intel-oneapi-mpi intel-oneapi-compilers   # `module avail intel` if renamed
# venv with torch + intel_extension_for_pytorch + oneccl_bindings_for_pytorch (XPU):
source "$HOME/rds/hpc-work/venvs/flsim-xpu/bin/activate"   # EDIT to your env

# ---- XPU / oneCCL configuration --------------------------------------------
export ZE_FLAT_DEVICE_HIERARCHY=FLAT           # expose each PVC tile as its own XPU
                                               # (2 tiles/card -> xpu:0..7 per node)
export CCL_ZE_IPC_EXCHANGE=sockets             # robust IPC handle exchange on SLURM
export CCL_ATL_TRANSPORT=ofi                   # oneCCL over libfabric (srun launch)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# PyTorch Lightning reads SLURM env vars automatically
cd "$SLURM_SUBMIT_DIR"
exec python scripts/train_lightning.py --devices 8 --nodes ${SLURM_NNODES:-1}
