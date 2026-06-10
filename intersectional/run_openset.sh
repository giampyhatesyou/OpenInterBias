#!/usr/bin/env bash
# Open-set all-pairs scale-up, end to end: link the reusable images, generate the new ones,
# run the uncapped VQA, scan ALL same-cluster pairs with FDR control. Run from the repo root
# on a machine with a GPU and the model weights cached (made to be the body of an sbatch job).
#
#   bash intersectional/run_openset.sh
#
# Prerequisites (CPU, can be done beforehand):
#   python intersectional/openset_select.py          # writes coco_train_openset.json + report
#   PYTHONHASHSEED=0 python intersectional/dryrun_pairs.py   # verify the plan, no GPU
set -euo pipefail

# PHASE=gen   -> only image generation (auto-resumes; safe to kill at the wall clock)
# PHASE=vqa   -> only VQA + scoring (run_VQA writes results ONCE AT THE END: it must fit
#                inside the remaining wall time or all its work is lost)
# PHASE=all   -> both (default; for a single long session)
PHASE="${PHASE:-all}"
WORKERS="${WORKERS:-4}"
export PYTHONHASHSEED=0
FILE=coco_train_openset.json
TAG=openset
PROP="proposed_biases/coco/3/$FILE"

# Point the config at the open-set subset; --max-prompts must exceed the caption count or the
# per-bias caption capping silently starves the VQA pair support. Skip if already patched.
if ! grep -q "results/VQA_${TAG}" utils/config.py; then
  python intersectional/apply_demo_config.py "$FILE" --tag "$TAG" --max-prompts 1000000
fi
trap 'git checkout utils/config.py' EXIT

# Reuse every already-generated caption: symlinks make generate_images auto-skip them.
python intersectional/link_reused_images.py

if [ "$PHASE" != "vqa" ]; then
  echo ">> captions in this run (generated + reused):"
  python intersectional/dryrun_count.py
  python generate_images.py --dataset coco --generator sd-xl
fi

if [ "$PHASE" = "gen" ]; then
  git checkout utils/config.py; trap - EXIT
  echo ">> PHASE=gen done ($(ls sd_generated_dataset/coco/train_${TAG}/sd-xl | wc -l) caption dirs)."
  exit 0
fi

python run_VQA.py --vqa_model llava-1.5-13b --workers "$WORKERS" --dataset coco --mode generated --generator sd-xl

git checkout utils/config.py; trap - EXIT

# Score: full all-pairs scan (no cluster filter), BH-FDR, raw + prompt-filtered variants.
VQADIR="results/VQA_$TAG"
DS="coco_$TAG"
rm -rf "results/VQA/$DS"; cp -r "$VQADIR/coco" "results/VQA/$DS"
python intersectional/run_analysis.py --dataset "$DS" --generator sd-xl --vqa_model llava-1.5-13b \
    --mode generated --min_support 30 --proposed_biases "$PROP" --fdr_q 0.05
rm -rf "results/VQA/${DS}clean"; cp -r "$VQADIR/coco" "results/VQA/${DS}clean"
python intersectional/run_analysis.py --dataset "${DS}clean" --generator sd-xl --vqa_model llava-1.5-13b \
    --mode generated --min_support 30 --proposed_biases "$PROP" --fdr_q 0.05 --exclude_leaky
echo ">> done. Results in results/intersectional/${DS} (raw) and ${DS}clean (prompt-filtered)."
