#!/usr/bin/env bash
# Run inference on the overfit dataset using the latest stage2 checkpoint.
# Run from the hfm/ root: bash scripts/infer_overfit.sh

set -e
cd "$(dirname "$0")/.."

DATA_DIR="$(pwd)/../data/fvm_gen_overfit"
OUT_DIR="$(pwd)/out/overfit"
CKPT_DIR="$(pwd)/checkpoints"

# Pick the latest stage2 checkpoint (fallback to any checkpoint if none found)
CKPT=$(ls -t "$CKPT_DIR"/overfit_stage2_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    CKPT=$(ls -t "$CKPT_DIR"/overfit_*.pt 2>/dev/null | head -1)
fi
if [ -z "$CKPT" ]; then
    echo "No checkpoint found in $CKPT_DIR — run scripts/overfit.py first."
    exit 1
fi

echo "Checkpoint : $CKPT"
echo "Data       : $DATA_DIR"
echo "Output     : $OUT_DIR"
echo ""

python infer.py \
    --checkpoint "$CKPT" \
    --data-dir   "$DATA_DIR" \
    --out-dir    "$OUT_DIR" \
    --n-warmup   3 \
    --n-predict  20
