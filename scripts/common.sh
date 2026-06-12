#!/usr/bin/env bash
# Shared setup for the stage scripts in this directory: cd to the project
# root, load .env, create the per-run log directory and define helpers.
# Meant to be sourced; safe to source more than once.

[ -n "${LENSCRAFT_COMMON_LOADED:-}" ] && return 0
LENSCRAFT_COMMON_LOADED=1

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Export everything from .env (the python entry points also load it via
# dotenv, but the shell needs OUTPUT_DIR etc. for checkpoint discovery).
set -a
source .env
set +a

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR_RUN="${LOG_DIR_RUN:-$OUTPUT_DIR/run_$RUN_STAMP}"
mkdir -p "$LOG_DIR_RUN"

# Returns the newest file matching $2 (a find -name pattern) under $1.
newest() {
    find "$1" -name "$2" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
}
