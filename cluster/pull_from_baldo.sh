#!/usr/bin/env bash
# =============================================================================
# Run on your LAPTOP, from the repo root, AFTER the job printed "=== DONE ===":
#     bash cluster/pull_from_baldo.sh
# Downloads the scaled-up VQA answers and re-runs Stage 5 on them locally.
# =============================================================================
set -euo pipefail
REMOTE='baldo:OpenInterBias/results/VQA_demo/coco/generated/sd-xl/llava-1.5-13b'
L='results/VQA_demo/coco/generated/sd-xl/llava-1.5-13b'

echo ">> pulling demo VQA results from baldo ..."
mkdir -p "$L"
scp "$REMOTE/vqa_answers.json" "$REMOTE/data_counts.json" "$L/"

echo ">> wiring the demo tree as dataset 'coco_demo' (so run_analysis finds it) ..."
rm -rf results/VQA/coco_demo
cp -r results/VQA_demo/coco results/VQA/coco_demo

echo ">> running Stage 5 on the scaled-up data ..."
PYTHONHASHSEED=0 python3 intersectional/run_analysis.py --dataset coco_demo --generator sd-xl \
    --vqa_model llava-1.5-13b --mode generated --cluster person --min_support 30
python3 intersectional/make_plots.py --dataset coco_demo --generator sd-xl \
    --vqa_model llava-1.5-13b --mode generated --cluster person --min_support 5

echo ">> done."
echo "   results: results/intersectional/coco_demo/generated/sd-xl/llava-1.5-13b/"
echo "   open:    intersectional_results.md  +  intersectional_nmi_person.png"
