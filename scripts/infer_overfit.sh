#!/usr/bin/env bash
# Run inference on the overfit dataset using the latest stage2 checkpoint.
# Run from the hfm/ root: bash scripts/infer_overfit.sh

set -e
cd "$(dirname "$0")/.."

DATA_DIR="$(pwd)/../data/fvm_gen_alternating"
OUT_DIR="$(pwd)/out/infer"
# CKPT="$(pwd)/checkpoints/ckpt-step037500.ckpt"
CKPT="$(pwd)/checkpoints/train_step004500.pt"
DET_CKPT="$(pwd)/checkpoints/refiner_step020000.pt"

echo "Checkpoint : $CKPT"
echo "Data       : $DATA_DIR"
echo "Output     : $OUT_DIR"
echo ""

# fvm_gen_alternating here is LOCALLY generated: fresh trajectories on a fresh
# mesh from the same generator, so every run is out-of-sample by construction
# and no filtering is needed.  Native frame spacing carries the real save_t, so
# the rollout ratio printed per run is directly comparable to the training
# log's val_ratio.  (If this dir ever points at a copy of the TRAINING corpus
# instead, add --val-only to restrict to the held-out 5%.)
#
# For OLD fixed-interval corpora (fvm_validation, save_t=0.01) add
# --time-stride 10: measured there, the model's dt readout saturates near its
# training median (~0.1), losing to persistence at stride 1 (ratio ~2.4) and
# winning at stride 10 (~0.77).
python infer.py \
    --checkpoint "$CKPT" \
    --data-dir   "$DATA_DIR" \
    --out-dir    "$OUT_DIR" \
    --seq-start 0 \
    --n-predict  25 
    # --refine "$DET_CKPT" \
    # --teacher-forcing
