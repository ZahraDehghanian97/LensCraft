set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS_DIR/common.sh"

source "$SCRIPTS_DIR/train_lenscraft.sh"
source "$SCRIPTS_DIR/train_clatr.sh"

# Freeze one full-model evaluator for every table row in this sweep.
export SEMANTIC_EVALUATOR_CHECKPOINT_PATH="${SEMANTIC_EVALUATOR_CHECKPOINT_PATH:-$TEST_CHECKPOINT_PATH}"
export SEMANTIC_EVALUATOR_CONFIG_PATH="${SEMANTIC_EVALUATOR_CONFIG_PATH:-${TEST_CONFIG_PATH:-}}"

export TEST_OUTPUT_DIR="$PROJECT_DIR/table_results"
mkdir -p "$TEST_OUTPUT_DIR"

STATIC_TYPES='[static]'
DYNAMIC_TYPES='[circular,zigzag,linear,spiral,figureEight,wave,pendulum,orbital,bounce]'

EVAL_BS="${EVAL_BS:-128}"
TEST_FRACTION="${TEST_FRACTION:-1.0}"

run_eval() {
    local name="$1" eset="$2" amt="$3"; shift 3
    echo ""
    echo "--- eval: $name ($eset) ---"
    python src/test.py \
        "+eval_set=$eset" \
        "test_movement_types=$amt" \
        "test_fraction=$TEST_FRACTION" \
        caption_top1_metric=false \
        "data.batch_size=$EVAL_BS" \
        "$@" 2>&1 | tee "$LOG_DIR_RUN/test_${name}_${eset}.log"
}

for eset in static dynamic; do
    if [ "$eset" = static ]; then amt="$STATIC_TYPES"; else amt="$DYNAMIC_TYPES"; fi

    run_eval lens_craft "$eset" "$amt"

    run_eval ccdm   "$eset" "$amt" baseline_norm_ablation=true training/model=ccdm
    run_eval et     "$eset" "$amt" baseline_norm_ablation=true training/model=et
    run_eval gendop "$eset" "$amt" baseline_norm_ablation=true training/model=gendop
done

echo ""
echo "--- efficiency ---"
python src/efficiency.py                                   2>&1 | tee "$LOG_DIR_RUN/eff_lens_craft.log"
python src/efficiency.py training/model=ccdm               2>&1 | tee "$LOG_DIR_RUN/eff_ccdm.log"
python src/efficiency.py training/model=et                 2>&1 | tee "$LOG_DIR_RUN/eff_et.log"
python src/efficiency.py training/model=gendop             2>&1 | tee "$LOG_DIR_RUN/eff_gendop.log"


python src/aggregate_results.py --results-dir "$TEST_OUTPUT_DIR"
echo ""
echo "Tables written under $TEST_OUTPUT_DIR/tables/ (and all_metrics.csv)."
