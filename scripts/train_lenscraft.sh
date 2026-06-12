#!/usr/bin/env bash
# Stage 1/3: train LensCraft on the simulation dataset (the default
# dataset) and export TEST_CHECKPOINT_PATH pointing at the best checkpoint.
#
# Skipped when TEST_CHECKPOINT_PATH already points at an existing file.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [ -z "${TEST_CHECKPOINT_PATH:-}" ] || [ ! -f "${TEST_CHECKPOINT_PATH:-}" ]; then
    echo "=== [1/3] Training LensCraft on the simulation dataset ==="
    python src/train.py 2>&1 | tee "$LOG_DIR_RUN/train_lenscraft.log"

    # Hydra chdirs into $OUTPUT_DIR/<date>/<time>, where Lightning's
    # ModelCheckpoint writes lightning_logs/version_*/checkpoints/best-val-model-*.ckpt
    TEST_CHECKPOINT_PATH="$(newest "$OUTPUT_DIR" 'best-val-model-*.ckpt')"
    if [ -z "$TEST_CHECKPOINT_PATH" ]; then
        echo "ERROR: no best-val-model-*.ckpt found under $OUTPUT_DIR after training." >&2
        exit 1
    fi
else
    echo "=== [1/3] Skipping training, using existing checkpoint ==="
fi
export TEST_CHECKPOINT_PATH
echo "LensCraft checkpoint: $TEST_CHECKPOINT_PATH"
