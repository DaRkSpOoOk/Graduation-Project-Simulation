#!/usr/bin/env bash
# CPU-only final QA for a completed/resumed extraction.  Submit this after
# the array with --dependency=afterok:<ARRAY_JOB_ID> when appropriate.

#SBATCH --job-name=karsl-final-qa
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=task008a-final-qa-%j.out
#SBATCH --error=task008a-final-qa-%j.err

set -euo pipefail

REPO_ROOT="${TASK008A_REPO_ROOT:-$PWD}"
DATA_ROOT="${KARSL_DATA_ROOT:?Set KARSL_DATA_ROOT}"
RUN_ROOT="${TASK008A_RUN_ROOT:?Set TASK008A_RUN_ROOT}"
MANIFEST="${TASK008A_MANIFEST:-$REPO_ROOT/datasets/manifests/karsl_core28.csv}"
QA_ROOT="${TASK008A_QA_ROOT:-$RUN_ROOT/qa}"
EXPORT_ROOT="${TASK008A_EXPORT_ROOT:-$RUN_ROOT/export}"

cd "$REPO_ROOT"
mkdir -p "$QA_ROOT" "$EXPORT_ROOT"

# Raw-video validation is read-only with respect to the committed manifest;
# its compact observation JSON is stored under the external data root.
python scripts/download_karsl_core28.py \
    --data-root "$DATA_ROOT" \
    --manifest "$MANIFEST" \
    --verify

python scripts/validate_karsl_core28.py \
    --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" \
    --run-root "$RUN_ROOT" \
    --output-json "$QA_ROOT/karsl_core28_qa.json"

python scripts/build_karsl_core28_index.py \
    --manifest "$MANIFEST" \
    --run-root "$RUN_ROOT" \
    --output "$EXPORT_ROOT/karsl_core28_virtual_glove.csv"
