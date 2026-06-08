#!/usr/bin/env bash
# =============================================================================
# Run ONCE on your LAPTOP, from the repo root:   bash cluster/push_to_baldo.sh
# Copies the demo inputs + helper scripts to baldo. After this:
#     ssh baldo
#     cd ~/OpenInterBias && bash cluster/run_demo.sh smoke
# =============================================================================
set -euo pipefail
R="baldo:OpenInterBias"   # ~/OpenInterBias on baldo

echo ">> copying demo proposals (smoke + 6k) ..."
scp proposed_biases/coco/3/coco_train_demo_smoke.json \
    proposed_biases/coco/3/coco_train_demo6k.json \
    "$R/proposed_biases/coco/3/"

echo ">> copying helper scripts ..."
scp intersectional/apply_demo_config.py intersectional/dryrun_count.py "$R/intersectional/"
scp cluster/ob_demo.sbatch cluster/run_demo.sh "$R/cluster/"

echo ">> done."
echo "   Next:  ssh baldo"
echo "          cd ~/OpenInterBias && bash cluster/run_demo.sh smoke"
