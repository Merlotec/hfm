#!/usr/bin/env bash
# Run inference on the overfit dataset using the latest stage2 checkpoint.
# Run from the hfm/ root: bash scripts/infer_overfit.sh

set -e
cd "$(dirname "$0")/.."

DATA_DIR="$(pwd)/../data/fvm_validation"
OUT_DIR="$(pwd)/out/infer"
# CKPT="$(pwd)/checkpoints/ckpt-step037500.ckpt"
CKPT="$(pwd)/checkpoints/train_step001500.pt"
DET_CKPT="$(pwd)/checkpoints/refiner_step020000.pt"

echo "Checkpoint : $CKPT"
echo "Data       : $DATA_DIR"
echo "Output     : $OUT_DIR"
echo ""

python infer.py \
    --checkpoint "$CKPT" \
    --data-dir   "$DATA_DIR" \
    --out-dir    "$OUT_DIR" \
    --n-predict  30 
    # --refine "$DET_CKPT" \
    # --teacher-forcing
