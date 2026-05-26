#!/usr/bin/env bash
# Snapshot git state, env, and Python version into a run folder.
#
# Usage:
#   bash tools/snapshot_run.sh <run_dir> <stage_label>
#
# Example:
#   bash tools/snapshot_run.sh runs/2026-05-26_baseline_v1 bias_proposal

set -euo pipefail

RUN_DIR="${1:-}"
STAGE="${2:-stage}"

if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: snapshot_run.sh: missing run_dir argument" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

# 1. Git commit + status (porcelain so it's diffable)
{
  echo "# Captured: $(date -u +%FT%TZ)"
  echo "# Stage: ${STAGE}"
  echo
  echo "## git rev-parse HEAD"
  git rev-parse HEAD 2>/dev/null || echo "NOT_A_GIT_REPO"
  echo
  echo "## git status --porcelain"
  git status --porcelain 2>/dev/null || true
  echo
  echo "## git log -1"
  git log -1 --format='%H%n%an <%ae>%n%ad%n%s' 2>/dev/null || true
} > "${RUN_DIR}/git.${STAGE}.txt"

# 2. Python + pip freeze (only if we're inside a venv)
{
  echo "# Captured: $(date -u +%FT%TZ)"
  echo
  echo "## python --version"
  python --version 2>&1 || true
  echo
  echo "## which python"
  which python || true
  echo
  echo "## pip freeze"
  pip freeze 2>/dev/null || echo "pip not available"
} > "${RUN_DIR}/env.${STAGE}.txt"

# 3. The literal command that's about to be executed (caller can append more)
echo "# Stage ${STAGE} snapshot complete." >> "${RUN_DIR}/command.${STAGE}.sh"
echo "# Add the launching command here for reproducibility." >> "${RUN_DIR}/command.${STAGE}.sh"

echo "snapshot_run.sh: wrote git.${STAGE}.txt, env.${STAGE}.txt, command.${STAGE}.sh in ${RUN_DIR}"
