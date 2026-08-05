#!/usr/bin/env bash
# Run inference on the overfit dataset using the latest stage2 checkpoint.
# Run from the hfm/ root: bash scripts/infer_overfit.sh

set -e
cd "$(dirname "$0")/.."

DATA_DIR="$(pwd)/../data/fvm_validation"
OUT_DIR="$(pwd)/out/infer"
CKPT="$(pwd)/checkpoints/ckpt-step003000.ckpt"

echo "Checkpoint : $CKPT"
echo "Data       : $DATA_DIR"
echo "Output     : $OUT_DIR"
echo ""

python infer.py \
    --checkpoint "$CKPT" \
    --data-dir   "$DATA_DIR" \
    --out-dir    "$OUT_DIR" \
    --n-predict  30 
    # --teacher-forcing
