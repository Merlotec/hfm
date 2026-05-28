#!/usr/bin/env bash
# Launch the fvm_viewer comparison server: real overfit data vs HFM predictions.
# Run from the hfm/ root: bash scripts/comp_server.sh [port]
#
# The viewer shows real frames on top and HFM predictions below.
# Warmup frames appear with a blue border (is_seed=True); predictions are neutral.
#
# Prerequisites:
#   1. Run scripts/overfit.py to train a checkpoint.
#   2. Run scripts/infer_overfit.sh to generate out/overfit/viewer/ frames.

set -e
cd "$(dirname "$0")/.."

REAL_DIR="$(pwd)/../fvm_model/data/fvm_gen_overfit"
GEN_DIR="$(pwd)/out/overfit/viewer"
VIEWER="$(pwd)/../fvm_model/fvm_viewer/viewer.py"
PORT="${1:-8051}"

if [ ! -d "$GEN_DIR" ]; then
    echo "Viewer output not found at $GEN_DIR"
    echo "Run scripts/infer_overfit.sh first to generate predictions."
    exit 1
fi

echo "Real data : $REAL_DIR"
echo "HFM preds : $GEN_DIR"
echo "Port      : $PORT"
echo ""
echo "Open http://localhost:$PORT in your browser."
echo ""

python "$VIEWER" "$REAL_DIR" -c "$GEN_DIR" --port "$PORT"
