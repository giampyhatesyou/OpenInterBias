"""Stage 5 — open-set generation planner: pick the captions that buy pair support cheapest.

The all-pairs scan needs joint support (~hundreds of observations) on MANY pairs, not just the
three demographic ones. Generating every caption of the full COCO proposals (73k) is wasteful;
generating a random subset starves the tail pairs. This selector solves the coverage problem
directly: greedily pick captions that contribute to the most still-unsatisfied target pairs
(weighted max-coverage), until every target pair is expected to reach ``--target_obs`` joint
observations or the budget is hit.

"Expected" support: a caption proposing a pair yields a usable joint observation when the VQA
answers BOTH attributes with non-excluded classes, so rate(pair) ~= P_use(a) * P_use(b), with
the per-attribute usable-answer rates measured on the existing runs (age 0.997, race 0.766,
occupation 0.093, global 0.89). The naive per-pair empirical rate from coco_demo/coco is NOT
used: those runs capped questions per bias (``max_prompts_per_bias`` feeds
``utils.get_first_caption`` in VQA_dataset too), so their realized pair support reflects the
cap, not answerability — the new run must set ``max_prompts_per_bias`` >> #captions, which
uncaps both generation and VQA. The exact asked-pair counts (post_processing merging etc.) are
verified on the cluster by ``dryrun_pairs.py`` before any GPU time is spent.

Already-generated captions (train/ + train_demo/) are selected FOR FREE: their image folders
are symlinked into the new subfolder, generation auto-skips populated folders, and the
uncapped VQA re-asks the full question set on the existing images. The openset vqa_answers
then supersedes the old capped ones (single source for the analysis). New captions are deduped
per source image (the upstream per-bias selection keeps one caption per image_id).

Outputs (never edits upstream files):
- ``proposed_biases/coco/3/coco_train_openset.json``  (same schema; selected captions only)
- ``results/intersectional/openset_selection_report.json`` (per-pair plan + config)

    python intersectional/openset_select.py --target_obs 400 --min_potential 100
"""
import os
import sys
import json
import math
import heapq
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairing  # noqa: E402

DEFAULT_FULL = "proposed_biases/coco/3/coco_train.full.json"
DEFAULT_OUT = "proposed_biases/coco/3/coco_train_openset.json"
DEFAULT_REPORT = "results/intersectional/openset_selection_report.json"
DEFAULT_MEASURED = [
    ("results/VQA/coco_demo/generated/sd-xl/llava-1.5-13b/vqa_answers.json"),
    ("results/VQA/coco/generated/sd-xl/llava-1.5-13b/vqa_answers.json"),
]


def caption_pairs(entry, attr_map):
    """Same-cluster attribute pairs proposed by one caption entry (canonicalized attrs)."""
    by = collections.defaultdict(set)
    for b in entry.get("proposed_biases", {}).get("bias", []):
        name, ref = b.get("name"), b.get("refer_to")
        if not isinstance(name, str) or not isinstance(ref, str) \
                or not isinstance(b.get("classes"), list):
            continue  # a handful of malformed LLM proposals in the full file
        by[ref].add(pairing.canonical_attr(name, "last_token", attr_map))
    out = set()
    for ref, attrs in by.items():
        attrs = sorted(attrs)
        for i, a in enumerate(attrs):
            for bb in attrs[i + 1:]:
                out.add((ref, a, bb))
    return out


def load_full(path, attr_map):
    """-> ``{caption_id(str): (image_id, pairs)}`` for the full proposals file."""
    with open(path) as f:
        data = json.load(f)["bias_proposal"]
    caps = {}
    for e in data:
        caps[str(e["caption_id"])] = (e.get("image_id"), caption_pairs(e, attr_map))
    return caps, data


def apply_asked(caps, asked_path):
    """Replace PROPOSED pairs with the pairs the pipeline will actually ASK.

    ``asked_path`` is the openset_asked_attrs.json dump of ``dryrun_pairs.py --dump_asked``
    run on the FULL pool: the upstream post-processing (valid-bias filter, class-cluster
    merging, and the caption-vs-class-synonyms filter) drops many (caption, bias) combos, so
    planning on proposals alone overestimates support ~2x. Captions absent from the dump are
    filtered out entirely and leave the pool.
    """
    with open(asked_path) as f:
        asked = json.load(f)
    out = {}
    for cid, attrs in asked.items():
        if cid not in caps:
            continue
        by = collections.defaultdict(set)
        for cluster, attr in attrs:
            by[cluster].add(attr)
        pairs = set()
        for cluster, s in by.items():
            ss = sorted(s)
            for i, a in enumerate(ss):
                for b in ss[i + 1:]:
                    pairs.add((cluster, a, b))
        if pairs:
            out[cid] = (caps[cid][0], pairs)
    return out


def usable_rates(measured_paths, class_map, attr_map, min_n=20):
    """Per-attribute usable-answer rate from the existing VQA runs.

    Usable = the predicted class survives class normalization + exclusions. Question CAPPING
    in those runs biases which questions were asked, not how usable an answer is, so the
    per-answer rate transfers to the uncapped run. Returns ``(rates, global_rate, gen_ids)``.
    """
    merged = {}
    for p in measured_paths:
        if os.path.exists(p):
            merged.update(pairing.load_vqa_answers(p))
    tot, ok = collections.Counter(), collections.Counter()
    for _img, biases in merged.items():
        for bias_name, val in biases.items():
            attr = pairing.canonical_attr(bias_name, "last_token", attr_map)
            tot[attr] += 1
            if pairing.normalize_class(attr, val[-1], class_map) is not None:
                ok[attr] += 1
    glob = sum(ok.values()) / max(sum(tot.values()), 1)
    rates = {a: ok[a] / tot[a] for a in tot if tot[a] >= min_n}
    gen_ids = {key.split("/")[-2] for key in merged}
    return rates, glob, gen_ids


def pair_rate(pair, attr_rates, glob, floor=0.02):
    """rate(r, a, b) ~= P_use(a) * P_use(b) (independent-answerability model)."""
    _r, a, b = pair
    return max(attr_rates.get(a, glob) * attr_rates.get(b, glob), floor)


def greedy_select(caps, target_pairs, target_obs, rate, gen_ids, budget):
    """Free phase (reuse generated images) + lazy-greedy max-coverage over new captions.

    Returns ``(reused, selected_new, expected)`` with expected joint obs per target pair.
    """
    expected = collections.defaultdict(float)

    # phase 1: already-generated captions cost no GPU generation, only VQA time, so reuse
    # every one that proposes ANY same-cluster pair — sub-target pairs still feed the scan
    # (min_support in the analysis is far below target_obs).
    reused = []
    for cid in sorted(gen_ids & set(caps)):
        if caps[cid][1]:
            reused.append(cid)
            for p in caps[cid][1] & target_pairs:
                expected[p] += rate[p]

    # phase 2: new captions, one per source image (the upstream selection dedups by image_id)
    by_image = {}
    for cid, (img, pairs) in caps.items():
        if cid in gen_ids:
            continue
        relevant = pairs & target_pairs
        if relevant and (img not in by_image or len(relevant) > len(by_image[img][1])):
            by_image[img] = (cid, relevant)
    pool = {cid: relevant for cid, relevant in by_image.values()}

    def gain(cid):
        return sum(1 for p in pool[cid] if expected[p] < target_obs)

    heap = [(-len(relevant), cid) for cid, relevant in pool.items()]
    heapq.heapify(heap)
    selected = []
    while heap and len(selected) < budget:
        neg, cid = heapq.heappop(heap)
        g = gain(cid)
        if g == 0:
            continue
        if -neg != g:                       # stale priority -> re-queue with current gain
            heapq.heappush(heap, (-g, cid))
            continue
        selected.append(cid)
        for p in pool.pop(cid):
            expected[p] += rate[p]
    return reused, selected, expected


def main(o):
    attr_map = pairing.load_attr_map(o.attr_map)
    class_map = pairing.load_class_map(o.class_map)
    caps, data = load_full(o.proposals, attr_map)
    if o.asked and os.path.exists(o.asked):
        caps = apply_asked(caps, o.asked)
        print(f"planning on EXACT asked pairs ({o.asked}): {len(caps)} captions survive "
              f"the upstream post-filter")
    attr_rates, glob, gen_ids = usable_rates(o.measured, class_map, attr_map)
    print(f"pool: {len(caps)} captions; reusable generated captions: "
          f"{len(gen_ids & set(caps))}; global usable-answer rate: {glob:.2f} "
          f"({len(attr_rates)} attrs estimable)")

    potential = collections.Counter()
    for _cid, (_img, pairs) in caps.items():
        for p in pairs:
            potential[p] += 1
    target_pairs = {p for p, n in potential.items() if n >= o.min_potential}
    rate = {p: pair_rate(p, attr_rates, glob) for p in target_pairs}
    print(f"target pairs (potential >= {o.min_potential}): {len(target_pairs)}")

    reused, selected, expected = greedy_select(caps, target_pairs, o.target_obs, rate,
                                               gen_ids, o.budget)

    rows = []
    for p in sorted(target_pairs, key=lambda x: -potential[x]):
        rows.append({"pair": pairing.pair_key(*p), "potential_captions": potential[p],
                     "rate": round(rate[p], 3),
                     "expected_obs": round(expected[p], 1),
                     "satisfied": expected[p] >= o.target_obs})
    sat = sum(r["satisfied"] for r in rows)
    for th in (30, 100, 200, 400):
        n = sum(1 for r in rows if r["expected_obs"] >= th)
        print(f"target pairs expected to reach >= {th:3d} obs: {n}/{len(rows)}")
    n_total = len(reused) + len(selected)
    avg_q = (sum(len(e["proposed_biases"].get("bias", [])) for e in data) / max(len(data), 1))
    print(f"\nreused {len(reused)} generated captions + {len(selected)} NEW captions "
          f"(budget {o.budget}) = {n_total} total; "
          f"{sat}/{len(rows)} target pairs expected to reach {o.target_obs} obs")
    print(f"GPU estimate: SDXL ~{len(selected) * 7 / 3600:.1f} h (new captions only, 1xL40S; "
          f"/4 with --gres=gpu:4); VQA ~{n_total * avg_q * 0.5 / 3600:.1f} h "
          f"(~{avg_q:.1f} questions/caption)")

    keep = set(selected) | set(reused)
    out_data = [e for e in data if str(e["caption_id"]) in keep]
    os.makedirs(os.path.dirname(o.out), exist_ok=True)
    with open(o.out, "w") as f:
        json.dump({"bias_proposal": out_data}, f)
    os.makedirs(os.path.dirname(o.report), exist_ok=True)
    with open(o.report, "w") as f:
        json.dump({"_config": vars(o), "n_reused": len(reused), "n_new": len(selected),
                   "n_satisfied": sat, "attr_usable_rates":
                       {a: round(r, 3) for a, r in sorted(attr_rates.items())},
                   "reused_caption_ids": reused, "pairs": rows}, f, indent=2)
    print(f"wrote {o.out} ({len(out_data)} captions) and {o.report}")
    return rows, reused, selected


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--proposals", default=DEFAULT_FULL)
    ap.add_argument("--measured", nargs="*", default=DEFAULT_MEASURED,
                    help="vqa_answers.json of already-generated runs (support + realization)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--asked", default="results/intersectional/openset_asked_attrs.json",
                    help="exact asked-attrs dump from `dryrun_pairs.py --dump_asked` on the "
                         "full pool; if present, plan on what the pipeline will really ask "
                         "instead of raw proposals")
    ap.add_argument("--target_obs", type=int, default=400,
                    help="joint observations to aim for per target pair")
    ap.add_argument("--min_potential", type=int, default=100,
                    help="a pair is a target if proposed by at least this many captions")
    ap.add_argument("--budget", type=int, default=15000,
                    help="hard cap on newly selected captions (= images at n-images=1)")
    ap.add_argument("--attr_map", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "attr_synonyms.json"))
    ap.add_argument("--class_map", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "class_map.json"))
    main(ap.parse_args())
