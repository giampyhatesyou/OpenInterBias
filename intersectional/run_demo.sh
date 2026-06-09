#!/usr/bin/env bash
# Demographic scale-up, end to end: generate images, run the VQA, score the intersectional pairs.
# Run from the repo root on a machine with a GPU and the model weights cached.
#
#   bash intersectional/run_demo.sh smoke   # 50 images, quick sanity check
#   bash intersectional/run_demo.sh 6k       # ~3000 demographic images, n-images=1 (context-free)
#   bash intersectional/run_demo.sh ctx      # 300 captions x 10 images (context-aware)
set -euo pipefail

TIER="${1:-6k}"
case "$TIER" in
  smoke) FILE=coco_train_demo_smoke.json; NIMG=1;  TAG=demo ;;
  6k)    FILE=coco_train_demo6k.json;     NIMG=1;  TAG=demo ;;
  full)  FILE=coco_train_demo.json;       NIMG=1;  TAG=demo ;;
  ctx)   FILE=coco_train_ctxaware.json;   NIMG=10; TAG=ctxaware ;;
  *) echo "usage: bash intersectional/run_demo.sh [smoke|6k|full|ctx]"; exit 1 ;;
esac
export PYTHONHASHSEED=0
PROP="proposed_biases/coco/3/$FILE"

# Point the generation config at the subset and separate output dirs; restore it on exit.
python intersectional/apply_demo_config.py "$FILE" --n-images "$NIMG" --tag "$TAG"
trap 'git checkout utils/config.py' EXIT

echo ">> images to generate:"
python intersectional/dryrun_count.py

python generate_images.py --dataset coco --generator sd-xl
python run_VQA.py --vqa_model llava-1.5-13b --workers 4 --dataset coco --mode generated --generator sd-xl

git checkout utils/config.py; trap - EXIT

# Score. Wire the VQA output as a dataset so run_analysis finds it.
VQADIR="results/VQA_$TAG"
DS="coco_$TAG"
rm -rf "results/VQA/$DS"; cp -r "$VQADIR/coco" "results/VQA/$DS"
python intersectional/run_analysis.py --dataset "$DS" --generator sd-xl --vqa_model llava-1.5-13b \
    --mode generated --cluster person --min_support 30 --proposed_biases "$PROP"
python intersectional/make_plots.py --dataset "$DS" --generator sd-xl --vqa_model llava-1.5-13b \
    --mode generated --cluster person --min_support 5

# For context-free runs, also produce the prompt-quality-filtered variant.
if [ "$NIMG" -eq 1 ]; then
  rm -rf "results/VQA/${DS}clean"; cp -r "$VQADIR/coco" "results/VQA/${DS}clean"
  python intersectional/run_analysis.py --dataset "${DS}clean" --generator sd-xl --vqa_model llava-1.5-13b \
      --mode generated --cluster person --min_support 30 --proposed_biases "$PROP" --exclude_leaky
  echo ">> done. Results in results/intersectional/${DS} (raw) and ${DS}clean (prompt-filtered)."
else
  echo ">> done. Context-aware results in results/intersectional/${DS} (see the Context-aware table)."
fi
