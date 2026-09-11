#!/usr/bin/env bash
# Pre-flight check: is this machine ready to run LensCraft?
# Verifies GPU + CUDA PyTorch, key Python deps, git submodules and that the
# paths declared in .env actually exist. Run from anywhere:
#   bash scripts/check_ready.sh
# Exit code is non-zero if any required check fails.

set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

red=$'\e[31m'; grn=$'\e[32m'; ylw=$'\e[33m'; rst=$'\e[0m'
fail=0
ok()   { printf "  ${grn}OK${rst}   %s\n" "$1"; }
bad()  { printf "  ${red}FAIL${rst} %s\n" "$1"; fail=1; }
warn() { printf "  ${ylw}WARN${rst} %s\n" "$1"; }

echo "== .env =="
if [ -f .env ]; then
    set -a; source .env; set +a
    ok ".env loaded"
else
    bad ".env missing (copy from .env-sample)"
fi

echo "== GPU / driver =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
    ok "nvidia-smi present"
else
    bad "nvidia-smi not found (no NVIDIA driver?)"
fi

echo "== Python / PyTorch / CUDA =="
python - <<'PY'
import importlib, sys
def check(mod):
    try:
        m = importlib.import_module(mod)
        print(f"  OK   {mod} {getattr(m,'__version__','?')}")
        return True
    except Exception as e:
        print(f"  FAIL {mod}: {e}")
        return False

allok = True
allok &= check("torch")
for m in ("torchvision","lightning","hydra","transformers","numpy","clip","msgpack","dotenv"):
    allok &= check(m)

try:
    import torch
    if torch.cuda.is_available():
        print(f"  OK   torch.cuda.is_available() -> True ({torch.cuda.device_count()} dev, {torch.version.cuda})")
    else:
        print("  FAIL torch.cuda.is_available() -> False")
        allok = False
except Exception as e:
    print(f"  FAIL torch cuda check: {e}")
    allok = False
sys.exit(0 if allok else 1)
PY
[ $? -ne 0 ] && fail=1

echo "== git submodules (third_parties) =="
for sub in third_parties/DIRECTOR third_parties/Camera-control third_parties/GenDoP; do
    if [ -n "$(ls -A "$sub" 2>/dev/null)" ]; then ok "$sub populated"; else bad "$sub empty -> git submodule update --init --recursive"; fi
done

echo "== data / checkpoint paths from .env =="
# required dir, required file, optional
check_dir()  { [ -d "${!1:-}" ] && ok "$1=${!1}" || bad "$1 missing or unset: '${!1:-}'"; }
check_file() { [ -f "${!1:-}" ] && ok "$1=${!1}" || bad "$1 missing or unset: '${!1:-}'"; }
opt_file()   { [ -z "${!1:-}" ] && warn "$1 unset (only needed for that baseline/feature)" || { [ -f "${!1}" ] && ok "$1=${!1}" || warn "$1 set but not found: ${!1}"; }; }

check_dir SIMULATION_DATA_PATH
check_dir ET_DATA_DIR
check_dir CCDM_DATA_DIR
check_dir DIRECTOR_PROJECT_DIR
check_file ET_CIN_LANG_PATH
# Baseline / eval-only — not needed for LensCraft training:
opt_file CCDM_CHECKPOINT_PATH   # only for training/model=ccdm baseline
opt_file GENDOP_CHECKPOINT_PATH # only for training/model=gendop baseline
opt_file TEST_CHECKPOINT_PATH   # only for src/test.py
opt_file SEMANTIC_EVALUATOR_CHECKPOINT_PATH # required for semantic evaluation
opt_file SEMANTIC_EVALUATOR_CONFIG_PATH     # optional fixed evaluator architecture

echo "== writable output / cache dirs =="
for v in OUTPUT_DIR CLIP_EMBEDDINGS_CACHE_DIR LOG_DIR; do
    d="${!v:-}"
    [ -z "$d" ] && { warn "$v unset"; continue; }
    mkdir -p "$d" 2>/dev/null && [ -w "$d" ] && ok "$v=$d (writable)" || bad "$v=$d not writable"
done

echo
if [ "$fail" -eq 0 ]; then
    printf "${grn}ALL REQUIRED CHECKS PASSED — server is ready.${rst}\n"
else
    printf "${red}Some required checks FAILED — see above.${rst}\n"
fi
exit $fail
