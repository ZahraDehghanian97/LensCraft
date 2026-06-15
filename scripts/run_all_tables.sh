set -uo pipefail
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS_DIR/common.sh"

source "$SCRIPTS_DIR/train_lenscraft.sh"
source "$SCRIPTS_DIR/train_clatr.sh"

export TEST_OUTPUT_DIR="$PROJECT_DIR/table_results"
mkdir -p "$TEST_OUTPUT_DIR"

STATIC_TYPES='[static]'
DYNAMIC_TYPES='[circular,zigzag,linear,spiral,figureEight,wave,pendulum,orbital,bounce]'

run_eval() {
    local name="$1" eset="$2" amt="$3"; shift 3
    echo ""
    echo "--- eval: $name ($eset) ---"
    python src/test.py \
        "+eval_set=$eset" \
        "+data.dataset.config.allowed_movement_types=$amt" \
        caption_top1_metric=true \
        "$@" 2>&1 | tee "$LOG_DIR_RUN/test_${name}_${eset}.log"
}

for eset in static dynamic; do
    if [ "$eset" = static ]; then amt="$STATIC_TYPES"; else amt="$DYNAMIC_TYPES"; fi

    run_eval lens_craft "$eset" "$amt"

    run_eval ccdm   "$eset" "$amt" training/model=ccdm
    run_eval et     "$eset" "$amt" training/model=et
    run_eval gendop "$eset" "$amt" training/model=gendop data.batch_size=4
done

echo ""
echo "--- efficiency ---"
python src/efficiency.py                                   2>&1 | tee "$LOG_DIR_RUN/eff_lens_craft.log"
python src/efficiency.py training/model=ccdm               2>&1 | tee "$LOG_DIR_RUN/eff_ccdm.log"
python src/efficiency.py training/model=et                 2>&1 | tee "$LOG_DIR_RUN/eff_et.log"
python src/efficiency.py training/model=gendop data.batch_size=4 2>&1 | tee "$LOG_DIR_RUN/eff_gendop.log"


python src/aggregate_results.py --results-dir "$TEST_OUTPUT_DIR"
echo ""
echo "Tables written under $TEST_OUTPUT_DIR/tables/ (and all_metrics.csv)."