"""Safely patch utils/config.py for the Stage-5 demo scale-up (run ON baldo, from the repo root).

Why: hand-editing the 5 values (one is an f-string with nested quotes) is error-prone. This makes
each replacement exactly once and refuses if anything looks off. Fully reversible: undo with
`git checkout utils/config.py`.

Usage:
    python intersectional/apply_demo_config.py coco_train_demo_smoke.json   # smoke test (50 imgs)
    git checkout utils/config.py                                            # then revert and...
    python intersectional/apply_demo_config.py coco_train_demo6k.json       # ...the real 6k run
"""
import os
import sys

CONFIG = "utils/config.py"

EDITS = [
    ("'max_prompts_per_bias': 2,", "'max_prompts_per_bias': 1000,"),
    ("'filter_threshold': 0.50,", "'filter_threshold': 0,"),
    ("'proposed_biases_path': f'proposed_biases/coco/{BIAS_PROPOSAL_SETTING[\"coco\"]"
     "[\"n_prompts_per_image\"]}/coco_train.json',",
     "'proposed_biases_path': 'proposed_biases/coco/3/{FILE}',"),
    ("'subfolder': 'coco/train',", "'subfolder': 'coco/train_demo',"),
    ("'save_path': 'results/VQA'", "'save_path': 'results/VQA_demo'"),
]


def main(proposed_file):
    if not os.path.exists(CONFIG):
        sys.exit(f"ERROR: {CONFIG} not found. Run this from the OpenInterBias repo root on baldo.")
    src = open(CONFIG).read()

    # refuse to double-apply
    if "'max_prompts_per_bias': 1000," in src or "results/VQA_demo" in src:
        sys.exit("ERROR: config already patched. `git checkout utils/config.py` first, then re-run.")

    for old, new in EDITS:
        new = new.replace("{FILE}", proposed_file)
        count = src.count(old)
        if count != 1:
            sys.exit(f"ERROR: expected exactly 1 occurrence, found {count}:\n   {old[:70]}...\n"
                     f"Aborting; config.py untouched (manual check needed).")
        src = src.replace(old, new)

    open(CONFIG, "w").write(src)
    print(f"OK — patched {CONFIG} to use proposed_biases/coco/3/{proposed_file}")
    print("Verify (should show ONLY utils/config.py, ~5 lines):")
    os.system("git diff --stat utils/config.py")
    print("\nNext: run the dry-run gate, then sbatch the job. Revert with: git checkout utils/config.py")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python intersectional/apply_demo_config.py <proposed_biases_filename.json>")
    main(sys.argv[1])
