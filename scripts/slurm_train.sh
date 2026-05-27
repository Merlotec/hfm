#!/bin/bash
#SBATCH --job-name=hfm_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # one Lightning process per node; it manages GPUs internally
#SBATCH --gpus-per-node=4          # adjust to match hardware (Dawn: 4× Max 1550 per node)
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=tmp/slurm_%j.out
#SBATCH --error=tmp/slurm_%j.err
# Uncomment for Dawn / Intel oneAPI:
# #SBATCH --partition=gpu
# module load oneapi intel-extension-for-pytorch

# For CUDA clusters (current A100 setup):
# module load cuda/12.x pytorch

set -euo pipefail

# ---- Environment ----
source /data/phy-thetis/nk624/flsim/.venv/bin/activate
cd /data/phy-thetis/nk624/flsim/hfm

# Lightning reads these automatically on SLURM — no torchrun needed
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Nodes:       $SLURM_NNODES"
echo "Master addr: $SLURM_NODELIST"
echo "Master port: $MASTER_PORT"
echo "Job ID:      $SLURM_JOB_ID"

srun python scripts/train_lightning.py \
    --devices 4 \
    --nodes "$SLURM_NNODES" \
    --data /data/phy-thetis/nk624/flsim/data/fvm_gen_datasets \
    --resume latest \
    --workers 8
