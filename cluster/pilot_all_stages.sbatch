#!/usr/bin/env bash
#SBATCH --job-name=ob_pilot
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err
#SBATCH --partition=TODO_partition
#SBATCH --account=TODO_account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
# =============================================================================
# Pilot smoke-test: runs all 4 stages back-to-back on a single allocation.
# Use only for the first end-to-end validation on baldo.
# =============================================================================

set -euo pipefail

if [ -f cluster/_common.local.env ]; then
  # shellcheck source=/dev/null
  source cluster/_common.local.env
else
  echo "ERROR: cluster/_common.local.env not found."
  exit 1
fi

cd "${OPENBIAS_REPO}"
# shellcheck source=/dev/null
source "${OPENBIAS_VENV}/bin/activate"
bash tools/snapshot_run.sh "${OPENBIAS_RUN_DIR}" "pilot_all"

LOG_DIR="${OPENBIAS_RUN_DIR}/logs"

echo "=== Stage 1: bias_proposals ==="
python bias_proposals.py --workers 6 --dataset coco \
    2>&1 | tee "${LOG_DIR}/01_bias_proposal.out"

echo "=== Stage 2: generate_images ==="
python generate_images.py --dataset coco --generator sd-xl \
    2>&1 | tee "${LOG_DIR}/02_generate_images.out"

echo "=== Stage 3: run_VQA ==="
python run_VQA.py --vqa_model llava-1.5-13b --workers 4 \
    --dataset coco --mode generated --generator sd-xl \
    2>&1 | tee "${LOG_DIR}/03_run_vqa.out"

echo "=== Stage 4: make_plots ==="
python make_plots.py --generator sd-xl --dataset coco --mode generated \
    2>&1 | tee "${LOG_DIR}/04_make_plots.out"

cp -v results/VQA/coco/generated/sd-xl/llava-1.5-13b/context_aware.png \
      "${OPENBIAS_RUN_DIR}/outputs/" || true
cp -v results/VQA/coco/generated/sd-xl/llava-1.5-13b/context_free.png \
      "${OPENBIAS_RUN_DIR}/outputs/" || true

echo "DONE: pilot all stages"
