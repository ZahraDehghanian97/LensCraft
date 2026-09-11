#!/usr/bin/env bash
# Train LensCraft on the simulation dataset, then evaluate all 4 models
# (LensCraft, CCDM, E.T., GenDoP) with src/test.py.
#
# The work is split into one file per stage in this directory. Each stage
# can also be run on its own:
#   train_lenscraft.sh   [1/3] train LensCraft -> TEST_CHECKPOINT_PATH
#   train_clatr.sh       [2/3] train the CLaTr evaluation backend
#   test_models.sh       [3/3] evaluate the 4 models, print summary
#
# Usage:
#   export SEMANTIC_EVALUATOR_CHECKPOINT_PATH=/runs/fixed-evaluator/best.ckpt
#   bash scripts/run_train_and_test.sh
#
# Optional environment overrides (otherwise taken from .env / defaults):
#   TEST_CHECKPOINT_PATH          skip training, evaluate this LensCraft ckpt
#   CLATR_NATIVE_CHECKPOINT_PATH  skip CLaTr training, use this ckpt
set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source (not exec) the stages so the checkpoint paths discovered and
# exported by one stage are visible to the next.
source "$SCRIPTS_DIR/train_lenscraft.sh"
source "$SCRIPTS_DIR/train_clatr.sh"
source "$SCRIPTS_DIR/test_models.sh"
