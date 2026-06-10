"""Point utils/config.py at a demographic subset and separate output dirs (run from the repo root).

Hand-editing the values (one is an f-string with nested quotes) is error-prone; this applies each
replacement and refuses if anything looks off. Undo with `git checkout utils/config.py`.

    python intersectional/apply_demo_config.py coco_train_demo6k.json
    python intersectional/apply_demo_config.py coco_train_ctxaware.json --n-images 10 --tag ctxaware
    python intersectional/apply_demo_config.py coco_train_openset.json --tag openset --max-prompts 1000000
    git checkout utils/config.py

``--max-prompts`` matters for the open-set run: max_prompts_per_bias caps which captions each
bias is generated AND VQA-questioned on (utils.get_first_caption), so it must exceed the number
of captions in the file or the all-pairs support is silently starved.
"""
import os
import sys
import argparse

CONFIG = "utils/config.py"
PROP_OLD = ("'proposed_biases_path': f'proposed_biases/coco/{BIAS_PROPOSAL_SETTING[\"coco\"]"
            "[\"n_prompts_per_image\"]}/coco_train.json',")


def main(proposed_file, n_images, tag, max_prompts):
    if not os.path.exists(CONFIG):
        sys.exit(f"ERROR: {CONFIG} not found. Run this from the repo root.")
    src = open(CONFIG).read()
    if f"results/VQA_{tag}" in src or "'max_prompts_per_bias': 2," not in src:
        sys.exit("ERROR: config already patched. `git checkout utils/config.py` first, then re-run.")

    # (old, new, expected_count); None = replace all occurrences (>=1)
    edits = [
        ("'max_prompts_per_bias': 2,", f"'max_prompts_per_bias': {max_prompts},", 1),
        ("'filter_threshold': 0.50,", "'filter_threshold': 0,", 1),
        (PROP_OLD, f"'proposed_biases_path': 'proposed_biases/coco/3/{proposed_file}',", 1),
        ("'subfolder': 'coco/train',", f"'subfolder': 'coco/train_{tag}',", 1),
        ("'save_path': 'results/VQA'", f"'save_path': 'results/VQA_{tag}'", 1),
    ]
    if n_images != 1:
        # this literal is shared by several datasets; only coco is generated, so replacing all is safe
        edits.append(("'n-images': 1,", f"'n-images': {n_images},", None))

    for old, new, expected in edits:
        count = src.count(old)
        if count == 0 or (expected is not None and count != expected):
            sys.exit(f"ERROR: expected {expected or '>=1'} occurrence(s), found {count}: {old[:60]}...")
        src = src.replace(old, new)

    open(CONFIG, "w").write(src)
    print(f"OK -> proposed_biases/coco/3/{proposed_file} | output tag '{tag}' | n-images={n_images}")
    os.system("git diff --stat utils/config.py")
    print("Revert when done with: git checkout utils/config.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("proposed_file")
    p.add_argument("--n-images", type=int, default=1, dest="n_images")
    p.add_argument("--tag", default="demo")
    p.add_argument("--max-prompts", type=int, default=1000, dest="max_prompts")
    o = p.parse_args()
    main(o.proposed_file, o.n_images, o.tag, o.max_prompts)
