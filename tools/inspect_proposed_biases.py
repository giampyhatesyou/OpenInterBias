"""Descriptive stats on a proposed_biases/*.json file.

This is a READ-ONLY tool. It does not call the upstream post_processing()
function — it works on the raw LLM output JSON. Use it to:
  - see how many captions had ≥1 valid bias proposed
  - see which attributes (by name) are most frequent
  - see the refer_to distribution
  - estimate the population for downstream intersectional pairing

It does NOT replicate the clustering/merging logic in utils/utils.py —
for that, use the full post_processing() function via the upstream code.

Usage:
  python tools/inspect_proposed_biases.py proposed_biases/coco/3/coco_train.json
  python tools/inspect_proposed_biases.py path/to/file.json --top 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(p: Path) -> list[dict]:
    with p.open() as f:
        data = json.load(f)
    if "bias_proposal" not in data:
        raise SystemExit(f"ERROR: {p} has no 'bias_proposal' top-level key")
    return data["bias_proposal"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", type=Path, help="proposed_biases/*.json")
    parser.add_argument("--top", type=int, default=10, help="Top-N to show in rankings")
    parser.add_argument(
        "--refer-to",
        type=str,
        default=None,
        help="Restrict stats to a single refer_to bucket (e.g. 'person')",
    )
    args = parser.parse_args()

    if not args.json_path.is_file():
        print(f"ERROR: {args.json_path} not found", file=sys.stderr)
        return 2

    entries = load(args.json_path)
    n_captions = len(entries)

    bias_name_counts: Counter[str] = Counter()
    refer_to_counts: Counter[str] = Counter()
    caption_to_biases: dict[int, set[str]] = defaultdict(set)
    captions_with_n_biases: Counter[int] = Counter()
    bias_pairs_per_caption: Counter[tuple[str, str]] = Counter()

    skipped_invalid = 0

    for entry in entries:
        biases = entry.get("proposed_biases", {}).get("bias", [])
        if not isinstance(biases, list):
            skipped_invalid += 1
            continue
        cid = entry.get("caption_id", -1)
        valid_names: list[str] = []
        for b in biases:
            if not isinstance(b, dict):
                continue
            name = str(b.get("name", "")).lower().strip()
            rt = str(b.get("refer_to", "")).lower().strip()
            if not name or not rt:
                continue
            if args.refer_to and rt != args.refer_to:
                continue
            bias_name_counts[name] += 1
            refer_to_counts[rt] += 1
            valid_names.append(name)
            caption_to_biases[cid].add(name)
        captions_with_n_biases[len(valid_names)] += 1
        # pairs within same caption (post-hoc intersectional candidates)
        valid_names_sorted = sorted(set(valid_names))
        for i in range(len(valid_names_sorted)):
            for j in range(i + 1, len(valid_names_sorted)):
                bias_pairs_per_caption[(valid_names_sorted[i], valid_names_sorted[j])] += 1

    print(f"--- inspect_proposed_biases.py ---")
    print(f"file: {args.json_path}")
    print(f"caption entries: {n_captions}")
    print(f"skipped (malformed): {skipped_invalid}")
    if args.refer_to:
        print(f"filter: refer_to == '{args.refer_to}'")

    print(f"\nrefer_to distribution:")
    for rt, c in refer_to_counts.most_common():
        print(f"  {rt:20s}  {c}")

    print(f"\ntop {args.top} bias names:")
    for name, c in bias_name_counts.most_common(args.top):
        print(f"  {name:40s}  {c}")

    print(f"\ncaptions with N valid biases:")
    for n in sorted(captions_with_n_biases.keys()):
        print(f"  N={n}: {captions_with_n_biases[n]} captions")

    print(f"\ntop {args.top} pairwise co-occurrences (intersectional candidates by raw count):")
    for (a, b), c in bias_pairs_per_caption.most_common(args.top):
        print(f"  {a:30s}  ×  {b:30s}  → {c} captions")

    print()
    print("NOTE: these are RAW counts before upstream post-processing.")
    print("Final pairing support will be lower after filter_threshold / hard_threshold")
    print("/ merge_threshold are applied by utils/utils.py:post_processing().")

    return 0


if __name__ == "__main__":
    sys.exit(main())
