#!/bin/bash
# HFM Lightning training on Ampere (NVIDIA A100)
# 4 GPUs per node => 4 ranks per node.
#
#   sbatch scripts/ampere_train.slurm
#
# The training stack uses PyTorch Lightning which detects SLURM's env vars
# and configures DDP.
#
# ---------------------------------------------------------------------------
#SBATCH --job-name=hfm-ampere
#SBATCH --account=MPHIL-NIKIFORAKIS-NK624-SL2-GPU
#SBATCH --partition=ampere                  # Ampere A100 partition
#SBATCH --nodes=1                           
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4                 # 1 rank per GPU
#SBATCH --cpus-per-task=16                  # EDIT: adjust based on cores available on Ampere nodes
#SBATCH --time=24:00:00
#SBATCH --output=logs/hfm_ampere.out
#SBATCH --error=logs/hfm_ampere.err
#SBATCH --export=NONE                       # Do not export loaded modules from the submission shell

set -euo pipefail
mkdir -p logs

# torch.distributed rendezvous. This script runs on the first node of the
# allocation, so its hostname IS the rendezvous master.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# ---- sanity echo ------------------------------------------------------------
RANKS_PER_NODE=4
TOTAL_RANKS=$(( SLURM_NNODES * RANKS_PER_NODE ))
echo "nodes=$SLURM_NNODES ranks=$TOTAL_RANKS master=$MASTER_ADDR"

# ---- launch ------------------------------------------------------------------
# srun runs in the clean env and spawns one task per GPU; each task runs
# ampere_task.sh, which loads the modules + venv and execs python.
cd "$SLURM_SUBMIT_DIR"
srun --ntasks="$TOTAL_RANKS" --ntasks-per-node="$RANKS_PER_NODE" \
    --export=ALL scripts/ampere_task.sh
