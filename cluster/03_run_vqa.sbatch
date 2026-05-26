#!/usr/bin/env bash
#SBATCH --job-name=ob_vqa
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err
#SBATCH --partition=TODO_partition
#SBATCH --account=TODO_account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
bash tools/snapshot_run.sh "${OPENBIAS_RUN_DIR}" "run_vqa"

# --- the actual stage ---
# Mirrors steps.sh line 8 of upstream.
python run_VQA.py \
    --vqa_model llava-1.5-13b \
    --workers 4 \
    --dataset coco \
    --mode generated \
    --generator sd-xl \
    2>&1 | tee "${OPENBIAS_RUN_DIR}/logs/03_run_vqa.out"

echo "DONE: stage 3 VQA — outputs under results/VQA/coco/generated/sd-xl/llava-1.5-13b/"
