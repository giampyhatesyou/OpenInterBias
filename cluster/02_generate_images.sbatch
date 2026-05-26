#!/usr/bin/env bash
#SBATCH --job-name=ob_genimg
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err
#SBATCH --partition=TODO_partition
#SBATCH --account=TODO_account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
bash tools/snapshot_run.sh "${OPENBIAS_RUN_DIR}" "generate_images"

# --- the actual stage ---
# Mirrors steps.sh line 5 of upstream.
# For multi-GPU, set CUDA_VISIBLE_DEVICES=0,1,... in _common.local.env
python generate_images.py \
    --dataset coco \
    --generator sd-xl \
    2>&1 | tee "${OPENBIAS_RUN_DIR}/logs/02_generate_images.out"

echo "DONE: stage 2 image generation — outputs under sd_generated_dataset/coco/train/sd-xl/"
