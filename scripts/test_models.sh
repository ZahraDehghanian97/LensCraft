#!/usr/bin/env bash
# Stage 3/3: evaluate all 4 models (LensCraft, CCDM, E.T., GenDoP) with
# src/test.py and print a summary.
#
# Baselines (CCDM / E.T. / GenDoP) use the trained LensCraft checkpoint as
# ref_model via TEST_CHECKPOINT_PATH; CCDM and E.T. auto-download their own
# pretrained weights on first run.
#
# Requires TEST_CHECKPOINT_PATH and CLATR_NATIVE_CHECKPOINT_PATH to be set
# (exported by train_lenscraft.sh / train_clatr.sh, or set manually).
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

echo "=== [3/3] Testing all 4 models ==="
FAILED=()

run_test() {
    local name="$1"; shift
    echo ""
    echo "--- Testing: $name ---"
    if python src/test.py "$@" 2>&1 | tee "$LOG_DIR_RUN/test_${name}.log"; then
        echo "--- $name: done (log: $LOG_DIR_RUN/test_${name}.log) ---"
    else
        echo "--- $name: FAILED (see $LOG_DIR_RUN/test_${name}.log) ---" >&2
        FAILED+=("$name")
    fi
}

# 1) LensCraft on the simulation test split
run_test lenscraft

# 2) CCDM baseline on the CCDM dataset (matching 300-frame format)
run_test ccdm training/model=ccdm data/dataset=ccdm

# 3) E.T. baseline on the E.T. dataset
run_test et training/model=et data/dataset=et

# 4) GenDoP baseline on the simulation dataset (generates one prompt at a
#    time, so keep the batch size small)
run_test gendop training/model=gendop data.batch_size=4

echo ""
echo "=== Summary ==="
echo "LensCraft checkpoint: $TEST_CHECKPOINT_PATH"
echo "Logs: $LOG_DIR_RUN"
grep -H "Final Metrics" "$LOG_DIR_RUN"/test_*.log 2>/dev/null || true

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED tests: ${FAILED[*]}" >&2
    exit 1
fi
echo "All 4 model evaluations completed."
