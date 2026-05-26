#!/usr/bin/env bash
#SBATCH --job-name=ob_propose
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err
#SBATCH --partition=TODO_partition
#SBATCH --account=TODO_account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --export=NONE

set -euo pipefail

# Load shared env
if [ -f cluster/_common.local.env ]; then
  # shellcheck source=/dev/null
  source cluster/_common.local.env
else
  echo "ERROR: cluster/_common.local.env not found. Copy cluster/_common.env first."
  exit 1
fi

cd "${OPENBIAS_REPO}"

# Activate venv
# shellcheck source=/dev/null
source "${OPENBIAS_VENV}/bin/activate"

# Snapshot git state into the run folder
bash tools/snapshot_run.sh "${OPENBIAS_RUN_DIR}" "bias_proposal"

# --- the actual stage ---
# Args mirror steps.sh line 2 of upstream
python bias_proposals.py \
    --workers 6 \
    --dataset coco \
    2>&1 | tee "${OPENBIAS_RUN_DIR}/logs/01_bias_proposal.out"

echo "DONE: stage 1 bias proposal — outputs under proposed_biases/coco/3/"
