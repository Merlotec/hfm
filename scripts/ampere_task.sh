#!/bin/bash -l
# Per-rank launcher for the Ampere job — invoked by `srun` from ampere_train.slurm.
#
# Modules are loaded HERE, inside each task, NOT in the batch script to prevent 
# SLURM client failures. `#!/bin/bash -l` (login shell) makes `module` available.

# Standard modules for CUDA/A100 (adjust for your cluster's Ampere environment)
# Example for CSD3 Ampere nodes:
# module load rhel8/default-amp
# module load cuda/11.8 cudnn/8.9_cuda-11.8

# venv with torch (CUDA):
# EDIT this path to point to your CUDA virtual environment
source "$HOME/rds/hpc-work/flsim/flsim-env/bin/activate"   

# ---- CUDA / NCCL configuration --------------------------------------------
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# PyTorch Lightning reads SLURM env vars automatically
cd "$SLURM_SUBMIT_DIR"
exec python scripts/train_lightning.py --devices 4 --nodes ${SLURM_NNODES:-1}
