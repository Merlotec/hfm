#!/bin/bash
# HFM Lightning training on Dawn (Cambridge) — Intel Data Centre GPU
# Max 1550 (PVC) nodes, 4 GPUs x 2 tiles per node => 8 XPU ranks per node.
#
#   sbatch scripts/dawn_train.slurm
#
# The training stack uses PyTorch Lightning which detects SLURM's env vars
# and configures DDP for oneCCL backend on XPU.
#
# ---------------------------------------------------------------------------
#SBATCH --job-name=hfm-dawn
#SBATCH --account=AIRR-P67-DAWN-GPU  
#SBATCH --partition=pvc9                    # Dawn PVC (Max 1550) partition
#SBATCH --nodes=4                           
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8                 # 1 rank per GPU tile (2 tiles/card)
#SBATCH --cpus-per-task=12                  # 96 cores/node / 8 ranks
#SBATCH --time=24:00:00
#SBATCH --output=logs/hfm_dawn.out
#SBATCH --error=logs/hfm_dawn.err
#SBATCH --export=NONE                       # Do not export loaded modules from the submission shell

set -euo pipefail
mkdir -p logs

# IMPORTANT: do NOT load the oneAPI modules here.  They pull in glibc/2.28 and
# mangle the library env, which breaks SLURM's own tools. The batch script stays 
# in the clean system env so `srun` works; the modules + venv are loaded per-rank 
# inside scripts/dawn_task.sh (which srun launches).

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# ---- sanity echo ------------------------------------------------------------
RANKS_PER_NODE=8
TOTAL_RANKS=$(( SLURM_NNODES * RANKS_PER_NODE ))
echo "nodes=$SLURM_NNODES ranks=$TOTAL_RANKS master=$MASTER_ADDR"

# ---- launch ------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
srun --ntasks="$TOTAL_RANKS" --ntasks-per-node="$RANKS_PER_NODE" \
    --export=ALL scripts/dawn_task.sh
