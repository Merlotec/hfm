#!/bin/bash
# Flow-matching REFINER training on Dawn — single process, single XPU tile.
#
#   sbatch --export=ALL,HFM_TRAIN_ARGS="--checkpoint checkpoints/dynamics_a.ckpt \
#          --data /path/to/dataset" scripts/dawn_refiner.slurm
#
# Unlike dawn_train.slurm this is a ONE-RANK job: scripts/train_refiner.py is
# single-process by design (the ~8.6M-param refiner trains against a FROZEN base,
# whose forwards are no-grad), so none of the oneCCL/DDP machinery applies.  It
# reuses scripts/dawn_task.sh purely for the environment (module purge +
# default-dawn, venv activation, LD_LIBRARY_PATH ordering, the import-torch
# preflight that catches flapping RDS mounts); the collective env it also sets is
# inert at 1 rank.
#
# ---------------------------------------------------------------------------
#SBATCH --job-name=hfm-refiner
#SBATCH --account=AIRR-P67-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # ONE rank: the refiner is single-device
#SBATCH --cpus-per-task=32                  # no rank contention -> give the loader cores
#SBATCH --time=24:00:00
#SBATCH --output=logs/hfm_refiner.out
#SBATCH --error=logs/hfm_refiner.err
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs

# Same rule as dawn_train.slurm: do NOT load oneAPI modules here — they pull in
# glibc/2.28 and break srun.  dawn_task.sh loads them per-rank.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501                    # distinct from dawn_train.slurm

export HFM_ENTRY=scripts/train_refiner.py
: "${HFM_TRAIN_ARGS:?set --checkpoint (frozen base) via --export=ALL,HFM_TRAIN_ARGS=...}"
echo "refiner args: $HFM_TRAIN_ARGS"

cd "$SLURM_SUBMIT_DIR"
srun --ntasks=1 --ntasks-per-node=1 --export=ALL scripts/dawn_task.sh
