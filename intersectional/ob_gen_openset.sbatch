#!/bin/bash
#SBATCH --job-name=ob_openset
#SBATCH --account=foundation.models25
#SBATCH --qos=gpuedu
#SBATCH --partition=edu-long
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=%x_%j.out

# Open-set all-pairs run, resumable across 8h sessions. Just resubmit the same file until done:
#  - generation auto-resumes (populated caption folders are skipped), so a wall-clock kill
#    during generation loses nothing;
#  - run_VQA writes its results once at the end, so VQA only starts when generation is
#    complete AND enough wall time remains to finish it — otherwise the job exits cleanly
#    and the next submission picks it up.
# HF cache: weights live in the DEFAULT ~/.cache/huggingface — do not override HF_HOME.
set -euo pipefail
cd ~/OpenInterBias
export PYTHONHASHSEED=0
export PATH="$HOME/openbias/bin:$PATH"

GPUS=$(nvidia-smi -L | wc -l)
echo ">> $GPUS GPU(s) visible"

PHASE=gen bash intersectional/run_openset.sh

# generation complete? (count caption dirs vs the dry-run's exact expectation)
EXPECTED=$(python -c "import json;print(json.load(open('results/intersectional/openset_dryrun_pairs.json'))['n_captions'])")
HAVE=$(ls sd_generated_dataset/coco/train_openset/sd-xl | wc -l)
if [ "$HAVE" -lt "$EXPECTED" ]; then
  echo ">> generation incomplete ($HAVE/$EXPECTED) — wall clock will stop us; RESUBMIT this job tomorrow."
  exit 0
fi

# enough time left for the all-or-nothing VQA? (~20k questions ≈ 2.5-3h on 1 GPU)
REMAINING=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
REM_S=$(python - "$REMAINING" <<'EOF'
import sys
t = sys.argv[1].strip()
d, t = t.split("-") if "-" in t else ("0", t)
p = [int(x) for x in t.split(":")]
while len(p) < 3:
    p.insert(0, 0)
print(int(d) * 86400 + p[0] * 3600 + p[1] * 60 + p[2])
EOF
)
NEED=$(( 12000 / GPUS ))
echo ">> remaining ${REM_S}s, VQA needs ~${NEED}s with $GPUS worker(s)"
if [ "$REM_S" -lt "$NEED" ]; then
  echo ">> not enough time for the all-or-nothing VQA — RESUBMIT this job; it will go straight to VQA."
  exit 0
fi

WORKERS=$GPUS PHASE=vqa bash intersectional/run_openset.sh
echo ">> ALL DONE — results in results/intersectional/coco_openset{,clean}"
