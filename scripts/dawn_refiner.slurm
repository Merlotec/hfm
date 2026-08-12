#!/bin/bash
# Flow-matching REFINER training on Dawn — single process, single XPU tile.
#
#   sbatch scripts/dawn_refiner.slurm
#
# EDIT THE SETTINGS BLOCK BELOW — the arguments live in this file on purpose.
# Passing them through `sbatch --export=ALL,HFM_TRAIN_ARGS="..."` does not work
# reliably: --export splits its value on COMMAS, so any comma in the argument
# string is parsed as the start of another VAR=VALUE pair.
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

# ============================ SETTINGS — EDIT ME ============================
# Frozen base to refine against.  A refiner is only valid for ONE base: the path,
# step and normalisation stats are recorded in every refiner checkpoint and
# checked by infer.py --refine, which warns loudly on a mismatch.
BASE_CKPT="checkpoints/dynamics_a.ckpt"

# Dataset root (same corpus the base was trained on).
DATA="$HOME/rds/rds-airr-p67-Z6yB993aUHo/nk624/data/fvm_gen_alternating_gpu"

BATCH_SIZE=16
NUM_WORKERS=16          # 1 rank owns the node's cores; 4 (the script default) idles the XPU
EPOCHS=50
MAX_STEPS=0             # 0 = unlimited (the --time limit stops it)
CKPT_EVERY=1000

# Set to a refiner checkpoint to continue, e.g. checkpoints/refiner_step004000.pt
RESUME=""

# Anything else to append verbatim (e.g. --cache-frames --lr 1e-4).
EXTRA=""
# ============================================================================

REFINER_ARGS="--checkpoint $BASE_CKPT --data $DATA"
REFINER_ARGS="$REFINER_ARGS --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS"
REFINER_ARGS="$REFINER_ARGS --epochs $EPOCHS --max-steps $MAX_STEPS --ckpt-every $CKPT_EVERY"
[ -n "$RESUME" ] && REFINER_ARGS="$REFINER_ARGS --resume $RESUME"
[ -n "$EXTRA" ]  && REFINER_ARGS="$REFINER_ARGS $EXTRA"

cd "$SLURM_SUBMIT_DIR"

# Fail here, with the path, rather than 10 minutes into node setup.
[ -f "$BASE_CKPT" ] || { echo "[fatal] base checkpoint not found: $BASE_CKPT" >&2; exit 1; }
[ -d "$DATA" ]      || { echo "[fatal] data dir not found: $DATA" >&2; exit 1; }

# Same rule as dawn_train.slurm: do NOT load oneAPI modules here — they pull in
# glibc/2.28 and break srun.  dawn_task.sh loads them per-rank.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501                    # distinct from dawn_train.slurm

export HFM_ENTRY=scripts/train_refiner.py
export HFM_TRAIN_ARGS="$REFINER_ARGS"
echo "refiner: python $HFM_ENTRY $HFM_TRAIN_ARGS"

srun --ntasks=1 --ntasks-per-node=1 --export=ALL scripts/dawn_task.sh
