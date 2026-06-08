#!/usr/bin/env bash
# =============================================================================
# ONE command to launch the Stage-5 demo scale-up. Run on the baldo LOGIN node,
# from the repo root:
#
#     bash cluster/run_demo.sh smoke      # 50-image test (~10-15 min incl model load)
#     bash cluster/run_demo.sh 6k         # the real run (~5.8 h on 2 GPU)
#     bash cluster/run_demo.sh full       # everything (~24 h)
#
# It patches utils/config.py, runs the dry-run gate (aborts if the count is wrong),
# submits the GPU job (generation + VQA), and tells you how to watch it.
# The job auto-reverts utils/config.py when it finishes.
# =============================================================================
set -euo pipefail

TIER="${1:-}"
case "$TIER" in
  smoke) FILE=coco_train_demo_smoke.json; PART=edu-medium ;;
  6k)    FILE=coco_train_demo6k.json;     PART=edu-long ;;
  full)  FILE=coco_train_demo.json;       PART=edu-long ;;
  *) echo "usage: bash cluster/run_demo.sh <smoke|6k|full>"; exit 1 ;;
esac
PY=~/openbias/bin/python
unset HF_HOME TRANSFORMERS_CACHE   # the dry-run's clustering step loads SBERT -> use the populated default HF cache

echo ">> [1/3] reset + patch utils/config.py  (proposed file: $FILE)"
git checkout utils/config.py 2>/dev/null || true
$PY intersectional/apply_demo_config.py "$FILE"

echo ">> [2/3] dry-run gate (counting images; also detects a ConceptNet hang)..."
OUT=$(PYTHONHASHSEED=0 $PY intersectional/dryrun_count.py 2>&1 || true)
N=$(printf '%s\n' "$OUT" | grep -oE 'IMAGES=[0-9]+' | grep -oE '[0-9]+' | tail -1 || true)
if [ -z "${N:-}" ]; then
  echo "$OUT" | tail -20
  echo "   !! dry-run did not produce a count -> reverting + aborting (see error above)."
  git checkout utils/config.py; exit 1
fi
echo "   images to generate: $N"
if [ "$N" -lt 1 ] || [ "$N" -gt 40000 ]; then
  echo "   !! count looks wrong -> reverting + aborting. Re-check the demo file / edits."
  git checkout utils/config.py; exit 1
fi

echo ">> [3/3] submitting GPU job on partition $PART ..."
JOB=$(sbatch --partition="$PART" cluster/ob_demo.sbatch | awk '{print $NF}')
echo ""
echo "   submitted job $JOB   ($N images: generation + VQA on 2 GPU)"
echo "   ------------------------------------------------------------------"
echo "   watch the queue:   squeue -u \$USER          (PD=queued, R=running)"
echo "   watch the log:     tail -f ob_demo_$JOB.out  (Ctrl-C just stops watching)"
echo "   when you see '=== DONE ===' the results are ready; on your laptop run:"
echo "       bash cluster/pull_from_baldo.sh"
echo "   (utils/config.py auto-reverts at the end of the job.)"
