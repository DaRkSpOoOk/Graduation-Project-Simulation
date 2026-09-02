#!/usr/bin/env bash
# TASK-008A production worker.  The array and resource directives are safe
# defaults for Ibex; override partition/account/QOS/output paths at sbatch
# time when the allocation requires them.
#
# Example (after creating the log directory):
#   sbatch --output="$TASK008A_RUN_ROOT/logs/core28-%A_%a.out" \
#          --error="$TASK008A_RUN_ROOT/logs/core28-%A_%a.err" \
#          slurm/task008a_core28_a100.slurm
#
# The full extraction is intentionally not launched by repository tests.

#SBATCH --job-name=karsl-core28
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --array=0-15%4
#SBATCH --output=task008a-core28-%A_%a.out
#SBATCH --error=task008a-core28-%A_%a.err

set -euo pipefail

REPO_ROOT="${TASK008A_REPO_ROOT:-$PWD}"
DATA_ROOT="${KARSL_DATA_ROOT:?Set KARSL_DATA_ROOT to an external Ibex data root}"
RUN_ROOT="${TASK008A_RUN_ROOT:?Set TASK008A_RUN_ROOT to an external run root}"
MANIFEST="${TASK008A_MANIFEST:-$REPO_ROOT/datasets/manifests/karsl_core28.csv}"
NUM_SHARDS="${TASK008A_NUM_SHARDS:-16}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/provenance" "$RUN_ROOT/state" "$RUN_ROOT/failures"

if [[ -n "${CUDA_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
fi
if [[ -n "${TASK008A_ENV_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "$TASK008A_ENV_ACTIVATE"
fi

echo "TASK-008A worker startup: $(date --iso-8601=seconds)"
echo "hostname=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
echo "repo=$REPO_ROOT manifest=$MANIFEST data_root=$DATA_ROOT run_root=$RUN_ROOT"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

python scripts/run_task008a_karsl_core28.py \
    --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" \
    --run-root "$RUN_ROOT" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "${SLURM_ARRAY_TASK_ID:?SLURM array index is required}" \
    --resume \
    ${TASK008A_RETRY_FAILED:+--retry-failed}
