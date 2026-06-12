#!/usr/bin/env bash
# Stage 2/3: make sure the native CLaTr evaluation backend has a checkpoint
# (src/test.py needs it to compute the CLaTr-based metrics) and export it
# as CLATR_NATIVE_CHECKPOINT_PATH.
#
# Skipped when CLATR_NATIVE_CHECKPOINT_PATH already points at an existing file.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [ -z "${CLATR_NATIVE_CHECKPOINT_PATH:-}" ] || [ ! -f "${CLATR_NATIVE_CHECKPOINT_PATH:-}" ]; then
    echo "=== [2/3] Training native CLaTr (evaluation backend) ==="
    export CLATR_OUTPUT_DIR="$OUTPUT_DIR/clatr_$RUN_STAMP"
    python src/train_clatr.py 2>&1 | tee "$LOG_DIR_RUN/train_clatr.log"

    CLATR_NATIVE_CHECKPOINT_PATH="$(newest "$CLATR_OUTPUT_DIR" 'clatr-best-*.ckpt')"
    if [ -z "$CLATR_NATIVE_CHECKPOINT_PATH" ]; then
        CLATR_NATIVE_CHECKPOINT_PATH="$CLATR_OUTPUT_DIR/last.ckpt"
    fi
    if [ ! -f "$CLATR_NATIVE_CHECKPOINT_PATH" ]; then
        echo "ERROR: no CLaTr checkpoint found under $CLATR_OUTPUT_DIR." >&2
        exit 1
    fi
else
    echo "=== [2/3] Using existing CLaTr checkpoint ==="
fi
export CLATR_NATIVE_CHECKPOINT_PATH
echo "CLaTr checkpoint: $CLATR_NATIVE_CHECKPOINT_PATH"
