#!/usr/bin/env bash
#SBATCH --job-name=ob_plots
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err
#SBATCH --partition=TODO_cpu_partition_if_available
#SBATCH --account=TODO_account
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --export=NONE

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
bash tools/snapshot_run.sh "${OPENBIAS_RUN_DIR}" "make_plots"

# --- the actual stage (CPU-only; no --gres=gpu line above) ---
python make_plots.py \
    --generator sd-xl \
    --dataset coco \
    --mode generated \
    2>&1 | tee "${OPENBIAS_RUN_DIR}/logs/04_make_plots.out"

# Copy plots into the run folder for archival
mkdir -p "${OPENBIAS_RUN_DIR}/outputs"
cp -v results/VQA/coco/generated/sd-xl/llava-1.5-13b/context_aware.png \
      "${OPENBIAS_RUN_DIR}/outputs/" || true
cp -v results/VQA/coco/generated/sd-xl/llava-1.5-13b/context_free.png \
      "${OPENBIAS_RUN_DIR}/outputs/" || true

echo "DONE: stage 4 plots — copies in ${OPENBIAS_RUN_DIR}/outputs/"
