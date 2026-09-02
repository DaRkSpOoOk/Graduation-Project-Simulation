#!/usr/bin/env bash
# Small A100 compatibility smoke test.  Use a deterministic 18--24-row
# manifest in TASK008A_SMOKE_MANIFEST when available.  The default manifest
# plus --limit is only a scheduling smoke test; it is not the official pilot
# contract unless the selected rows are explicitly documented.

#SBATCH --job-name=karsl-smoke
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --array=0-0
#SBATCH --output=task008a-smoke-%A_%a.out
#SBATCH --error=task008a-smoke-%A_%a.err

set -euo pipefail

REPO_ROOT="${TASK008A_REPO_ROOT:-$PWD}"
DATA_ROOT="${KARSL_DATA_ROOT:?Set KARSL_DATA_ROOT}"
RUN_ROOT="${TASK008A_SMOKE_RUN_ROOT:-${TASK008A_RUN_ROOT:?Set TASK008A_RUN_ROOT or TASK008A_SMOKE_RUN_ROOT}/smoke}"
MANIFEST="${TASK008A_SMOKE_MANIFEST:-$REPO_ROOT/datasets/manifests/karsl_core28.csv}"
LIMIT="${TASK008A_SMOKE_LIMIT:-18}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/provenance" "$RUN_ROOT/state" "$RUN_ROOT/failures"
if [[ -n "${CUDA_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
fi
if [[ -n "${TASK008A_ENV_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "$TASK008A_ENV_ACTIVATE"
fi

python scripts/run_task008a_karsl_core28.py \
    --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" \
    --run-root "$RUN_ROOT" \
    --num-shards 1 \
    --shard-index 0 \
    --limit "$LIMIT" \
    --resume
