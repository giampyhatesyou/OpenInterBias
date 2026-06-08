#!/usr/bin/env bash
# =============================================================================
# Stage-5 demo scale-up: generation + VQA in one job. Self-contained (no
# _common.local.env needed). Submit from the repo root on baldo:  sbatch cluster/ob_demo.sbatch
#
# PRE-REQS (do these on the baldo LOGIN node first):
#   1) demo proposals copied to ~/OpenInterBias/proposed_biases/coco/3/
#   2) python intersectional/apply_demo_config.py coco_train_demo_smoke.json   (or _demo6k.json)
#   3) the dry-run gate printed a sane image count
# REVERT AFTER:  git checkout utils/config.py
# =============================================================================
#SBATCH --job-name=ob_demo
#SBATCH --account=foundation.models25
#SBATCH --qos=gpuedu
#SBATCH --partition=edu-long          # smoke run: edu-medium (2h) schedules faster & is plenty
#SBATCH --nodes=1
#SBATCH --ntasks=1                     # REQUIRED (sbatch refuses without it)
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00                # a max, not a reservation; a 50-img smoke just finishes early
#SBATCH --output=ob_demo_%j.out
#SBATCH --error=ob_demo_%j.err

set -euo pipefail
cd ~/OpenInterBias

# weights live in the DEFAULT HF cache; the repo's .hf_cache is empty -> never point HF there
unset HF_HOME TRANSFORMERS_CACHE
export PYTHONHASHSEED=0                 # VQA merge_class_clusters determinism (else intermittent KeyError)
export TOKENIZERS_PARALLELISM=false

PY=~/openbias/bin/python               # use the venv python directly (activate from repo fails)

# peft breaks SDXL loading (needs accelerate>=0.34); the baseline doesn't use it
if $PY -m pip show peft >/dev/null 2>&1; then
  echo ">> removing peft (SDXL load crash); reinstall later with: pip install peft==0.17.1"
  $PY -m pip uninstall -y peft
fi

echo "=== which config is active ==="
$PY - <<'EOF'
from utils.config import GEN_SETTING, VQA_SETTING
print("proposed_biases:", GEN_SETTING['coco']['proposed_biases_path'])
print("gen save_path   :", GEN_SETTING['save_path'], "/", GEN_SETTING['coco']['subfolder'])
print("vqa save_path   :", VQA_SETTING['save_path'])
print("max_prompts     :", GEN_SETTING['max_prompts_per_bias'])
EOF

echo "=== GENERATION ($(date)) ==="
$PY generate_images.py --dataset coco --generator sd-xl

echo "=== VQA ($(date)) ==="
$PY run_VQA.py --vqa_model llava-1.5-13b --workers 4 --dataset coco --mode generated --generator sd-xl

echo "=== VQA done; reverting config ($(date)) ==="
git checkout utils/config.py || echo "WARN: could not revert utils/config.py (do it by hand)"

echo "=== DONE ($(date)) ==="
echo "Output: results/VQA_demo/coco/generated/sd-xl/llava-1.5-13b/vqa_answers.json"
