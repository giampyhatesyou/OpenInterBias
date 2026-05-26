#!/usr/bin/env bash
#SBATCH --job-name=ob_inter
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err
#SBATCH --partition=TODO_cpu_partition_if_available
#SBATCH --account=TODO_account
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
# =============================================================================
# Stage 5 — Intersectional (post-hoc) analysis
#
# ⚠️  PLACEHOLDER: the python entry point referenced below
#     (intersectional/run_analysis.py) does NOT exist yet. This sbatch
#     describes the *shape* of the future submission so the cluster
#     setup is ready when the increment lands.
#
# Pre-requisites:
#   - Stages 1-4 have completed and produced vqa_answers.json
#   - intersectional/run_analysis.py is implemented (per intersectional/SCHEMA.md
#     and docs/SCHEMA_DECISION.md once Q1-Q5 are answered)
#   - A run folder exists at ${OPENBIAS_RUN_DIR} with
#     config.snapshot.yaml = configs/intersectional.template.yaml (filled)
#
# This stage is CPU-only (no --gres=gpu above) — it reads cached JSONs
# and computes joint distributions / metrics. Cheap.
# =============================================================================

set -euo pipefail

if [ -f cluster/_common.local.env ]; then
  # shellcheck source=/dev/null
  source cluster/_common.local.env
else
  echo "ERROR: cluster/_common.local.env not found. Copy cluster/_common.env first."
  exit 1
fi

cd "${OPENBIAS_REPO}"
# shellcheck source=/dev/null
source "${OPENBIAS_VENV}/bin/activate"
bash tools/snapshot_run.sh "${OPENBIAS_RUN_DIR}" "intersectional_analysis"

# Guardrail: refuse to run if the analysis script doesn't exist yet
if [ ! -f intersectional/run_analysis.py ]; then
  echo "ERROR: intersectional/run_analysis.py does not exist yet."
  echo "This sbatch is a PLACEHOLDER waiting for the increment to be implemented."
  echo "See docs/SCHEMA_DECISION.md and intersectional/SCHEMA.md for the design."
  exit 2
fi

# --- the actual stage (CPU-only) ---
# Expected CLI mirrors run_VQA.py for consistency with the baseline.
python -m intersectional.run_analysis \
    --config "${OPENBIAS_RUN_DIR}/config.snapshot.yaml" \
    --dataset coco \
    --generator sd-xl \
    --mode generated \
    --vqa_model llava-1.5-13b \
    2>&1 | tee "${OPENBIAS_RUN_DIR}/logs/05_intersectional.out"

echo "DONE: stage 5 intersectional analysis"
echo "Expected output: results/intersectional/.../intersectional_results.json"
