"""Exact pre-GPU verification of the open-set plan. CPU-only; run on the cluster AFTER
``apply_demo_config.py ... --tag openset --max-prompts 1000000`` and BEFORE any GPU job.

``openset_select.py`` plans with an approximate model (potential pairs x usable-answer rates).
The pipeline, however, decides which questions are actually asked through
``utils.post_processing`` (valid-bias filtering, class-cluster merging, per-bias caption
capping) — this script replicates the VQA_dataset question assignment on the CURRENT config
and reports, per pair, how many captions will really be asked BOTH questions. Numbers here are
upper bounds on realized support (the VQA can still answer 'unknown'); multiply by the
usable-answer rates in openset_selection_report.json for the expected support.

    PYTHONHASHSEED=0 python intersectional/dryrun_pairs.py
    PYTHONHASHSEED=0 python intersectional/dryrun_pairs.py \
        --proposals proposed_biases/coco/3/coco_train.json --dump_asked   # full-pool survey
"""
import os
import sys
import json
import argparse
import itertools
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "intersectional"))

import pairing  # noqa: E402

cli = argparse.ArgumentParser()
cli.add_argument("--proposals", default=None,
                 help="override the config's proposals file (e.g. the FULL pool, to know the "
                      "exact post-filter survival of every candidate caption)")
cli.add_argument("--dump_asked", action="store_true",
                 help="also dump {caption_id: [[cluster, attr], ...]} for the selector")
cli_args = cli.parse_args()

sys.argv = ["x", "--dataset", "coco", "--generator", "sd-xl"]
import utils.arg_parse as ap  # noqa: E402
import utils.utils as uu  # noqa: E402
from utils.datasets import Proposed_biases  # noqa: E402

opt = ap.argparse_generate_images()
if cli_args.proposals:
    opt["dataset_setting"]["proposed_biases_path"] = cli_args.proposals
ds = Proposed_biases(
    opt["dataset_setting"]["proposed_biases_path"],
    opt["gen_setting"]["max_prompts_per_bias"],
    opt["gen_setting"]["filter_threshold"],
    opt["gen_setting"]["hard_threshold"],
    opt["gen_setting"]["merge_threshold"],
    opt["dataset_setting"]["valid_bias_fn"],
    opt["dataset_setting"]["filter_caption_fn"],
    opt["dataset_setting"]["all_images"],
)
print(f"proposals: {opt['dataset_setting']['proposed_biases_path']}")
print(f"IMAGES (captions to generate or reuse) = {len(ds.get_data())}")

# replicate the VQA_dataset question assignment: which (caption, bias) get asked
bias_captions_final, _bias_classes_final, captions = ds.get_biases()
attr_map = pairing.load_attr_map(os.path.join(ROOT, "intersectional", "attr_synonyms.json"))
asked = collections.defaultdict(set)  # caption_id -> {(cluster, attr)}
n_questions = 0
for cluster in bias_captions_final:
    for bias_name in bias_captions_final[cluster]:
        for class_cluster in bias_captions_final[cluster][bias_name]:
            cpts = uu.get_first_caption(
                captions_id=bias_captions_final[cluster][bias_name][class_cluster],
                captions=captions,
                max_prompts=opt["gen_setting"]["max_prompts_per_bias"],
            )
            for cpt_id, _q in cpts:
                asked[cpt_id].add((cluster, pairing.canonical_attr(bias_name, "last_token",
                                                                   attr_map)))
                n_questions += 1
print(f"VQA questions = {n_questions} over {len(asked)} captions "
      f"(~{n_questions / max(len(asked), 1):.1f} per caption)")

pair_captions = collections.Counter()
for _cid, attrs in asked.items():
    by = collections.defaultdict(set)
    for cluster, attr in attrs:
        by[cluster].add(attr)
    for cluster, s in by.items():
        for a, b in itertools.combinations(sorted(s), 2):
            pair_captions[(cluster, a, b)] += 1

print(f"\npairs with both questions asked on the same caption: {len(pair_captions)}")
for th in (30, 100, 200, 400):
    print(f"  pairs with >= {th:3d} asked captions: "
          f"{sum(1 for v in pair_captions.values() if v >= th)}")
print("\ntop 25 pairs by asked captions:")
for (c, a, b), v in pair_captions.most_common(25):
    print(f"  {pairing.pair_key(c, a, b):45s} {v}")

out = os.path.join(ROOT, "results", "intersectional", "openset_dryrun_pairs.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"proposals": opt["dataset_setting"]["proposed_biases_path"],
               "n_captions": len(ds.get_data()), "n_questions": n_questions,
               "pairs_asked": {pairing.pair_key(*k): v for k, v in pair_captions.items()}},
              f, indent=2)
print(f"\nwrote {out}")

if cli_args.dump_asked:
    dump = os.path.join(ROOT, "results", "intersectional", "openset_asked_attrs.json")
    with open(dump, "w") as f:
        json.dump({str(cid): sorted(map(list, attrs)) for cid, attrs in asked.items()}, f)
    print(f"wrote {dump}")
