"""Stage 5 — CLI entry point.

Reads ``results/VQA/<dataset>/generated/<gen>/<vqa>/vqa_answers.json``, forms same-cluster
attribute pairs, scores them, and writes (NEW, never edits upstream):
- ``joint_answers.json``            (per-pair joint observations)
- ``intersectional_results.json``   (metrics + support + bootstrap CI per pair)
- ``intersectional_results.md``     (human-readable ranked summary)

CPU-only, deterministic. Mirrors the upstream CLI surface so it chains in ``cluster/``.

    python intersectional/run_analysis.py --dataset coco --generator sd-xl \
        --vqa_model llava-1.5-13b --mode generated
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairing  # noqa: E402
import scoring  # noqa: E402
import baseline_marginals as bm  # noqa: E402

DEFAULT_CLASS_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_map.json")


def results_dir(dataset, generator, vqa_model, mode):
    if mode == "original":
        vqa = f"results/VQA/{dataset}/{mode}/{vqa_model}"
    else:
        vqa = f"results/VQA/{dataset}/{mode}/{generator}/{vqa_model}"
    out = vqa.replace("results/VQA/", "results/intersectional/")
    return vqa, out


def run(dataset, generator, vqa_model, mode, attr_mode, cluster_filter,
        normalize, min_support, n_boot, n_perm, seed, class_map_path=DEFAULT_CLASS_MAP):
    vqa_path, out_path = results_dir(dataset, generator, vqa_model, mode)
    vqa_answers = pairing.load_vqa_answers(os.path.join(vqa_path, "vqa_answers.json"))
    class_map = pairing.load_class_map(class_map_path)

    joint = pairing.build_pairs(vqa_answers, attr_mode=attr_mode,
                                cluster_filter=cluster_filter, class_map=class_map)
    os.makedirs(out_path, exist_ok=True)
    pairing.write_joint_answers(joint, os.path.join(out_path, "joint_answers.json"))

    # full-data single-attribute marginals (context for the pair numbers)
    marginals = {}
    dc_path = os.path.join(vqa_path, "data_counts.json")
    if os.path.exists(dc_path):
        with open(dc_path) as f:
            data_counts = json.load(f)
        marginals = bm.baseline_marginals(data_counts, cluster_filter or "person", class_map)

    results = {}
    for (cluster, a, b), caps in joint.items():
        pooled = [obs for lst in caps.values() for obs in lst]
        cf = scoring.score_pair(pooled, normalize=normalize, n_boot=n_boot,
                                n_perm=n_perm, seed=seed)
        ca = scoring.context_aware_joint_intensity(caps)
        results[pairing.pair_key(cluster, a, b)] = {
            "refer_to": cluster,
            "attr_a": a,
            "attr_b": b,
            "support_captions": len(caps),
            "below_min_support": cf["support_images"] < min_support,
            "context_free": cf,
            "context_aware": ca,  # NOTE: degenerate at n-images=1 (see STAGE5_LOG §2)
        }

    config = {
        "dataset": dataset, "generator": generator, "vqa_model": vqa_model, "mode": mode,
        "attr_mode": attr_mode, "cluster_filter": cluster_filter, "mi_normalize": normalize,
        "min_support": min_support, "n_boot": n_boot, "n_perm": n_perm, "seed": seed,
        "class_map": class_map_path if class_map else None,
        "n_pairs": len(results), "n_images_total": len(vqa_answers),
    }
    payload = {"_config": config, "baseline_marginals": marginals, "pairs": results}
    with open(os.path.join(out_path, "intersectional_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    _write_markdown(payload, os.path.join(out_path, "intersectional_results.md"))
    _print_summary(payload, min_support)
    return payload, out_path


def _ranked(payload):
    rows = []
    for key, r in payload["pairs"].items():
        cf = r["context_free"]
        rows.append({
            "key": key, "refer_to": r["refer_to"], "support": cf["support_images"],
            "nmi": cf["mutual_information"], "nmi_mm": cf.get("mutual_information_mm"),
            "ci": cf["nmi_ci95"], "pval": cf.get("nmi_pvalue"),
            "ji": cf["joint_intensity"], "low": r["below_min_support"],
        })
    # rank by support so the trustworthy pairs are on top
    return sorted(rows, key=lambda x: x["support"], reverse=True)


def _write_markdown(payload, path):
    cfg = payload["_config"]
    lines = ["# Intersectional results (Stage 5)\n",
             f"- config: `{cfg['dataset']}/{cfg['generator']}/{cfg['vqa_model']}` "
             f"| attr_mode=`{cfg['attr_mode']}` | MI norm=`{cfg['mi_normalize']}` "
             f"| min_support={cfg['min_support']} | class_map=`{bool(cfg.get('class_map'))}`",
             f"- {cfg['n_pairs']} pairs over {cfg['n_images_total']} images.\n"]
    bmg = payload.get("baseline_marginals") or {}
    if bmg:
        lines.append("**Full-data single-attribute marginals** (all person obs, for context — the "
                     "pairs below are measured on much smaller intersections):")
        lines.append("| attribute | bias intensity | total obs |")
        lines.append("|---|---:|---:|")
        for attr in sorted(bmg):
            m = bmg[attr]
            lines.append(f"| {attr} | {m['intensity'] if m['intensity'] is not None else '—'} "
                         f"| {m['total']} |")
        lines.append("")
    lines += [
             "**Read support first.** Pairs below `min_support` are flagged ⚠ — their NMI is",
             "dominated by small-sample bias. Trust `NMI_MM` (Miller-Madow, bias-corrected) and",
             "`p` (permutation test, H0=independence) over the raw plug-in `NMI`.\n",
             "| pair | refer_to | support | NMI | NMI_MM | p | NMI 95% CI | Joint Int | |",
             "|---|---|---:|---:|---:|---:|---|---:|---|"]
    for d in _ranked(payload):
        flag = "⚠ low" if d["low"] else "ok"
        ci = d["ci"]
        ci_s = f"[{ci[0]}, {ci[1]}]" if ci else "—"
        g = lambda v: v if v is not None else "—"  # noqa: E731
        lines.append(f"| `{d['key']}` | {d['refer_to']} | {d['support']} | {g(d['nmi'])} "
                     f"| {g(d['nmi_mm'])} | {g(d['pval'])} | {ci_s} | {g(d['ji'])} | {flag} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_summary(payload, min_support):
    print(f"\n=== Stage 5 summary ({payload['_config']['n_pairs']} pairs) ===")
    print(f"{'pair':38s} {'sup':>5s} {'NMI':>7s} {'NMI_MM':>7s} {'p':>6s}  {'NMI 95% CI':>16s}")
    shown = 0
    for d in _ranked(payload):
        if d["support"] < min_support and shown >= 12:
            continue
        ci = d["ci"]
        ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "—"
        f = lambda v, w=7, p=4: (f"{v:.{p}f}" if v is not None else "—").rjust(w)  # noqa: E731
        flag = " ⚠" if d["low"] else ""
        print(f"{d['key'][:38]:38s} {d['support']:5d} {f(d['nmi'])} {f(d['nmi_mm'])} "
              f"{f(d['pval'],6,3)}  {ci_s:>16s}{flag}")
        shown += 1
    print(f"\n(⚠ = support < {min_support}; NMI not reportable. NMI_MM=Miller-Madow, p=permutation test.)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", default="coco")
    p.add_argument("--generator", default="sd-xl")
    p.add_argument("--vqa_model", default="llava-1.5-13b")
    p.add_argument("--mode", default="generated", choices=["generated", "original"])
    p.add_argument("--attr_mode", default="last_token", choices=["last_token", "raw"],
                   help="canonicalize bias_name by last token (default) or keep raw (plan D3)")
    p.add_argument("--cluster", default=None,
                   help="restrict to one refer_to cluster, e.g. 'person' (default: all)")
    p.add_argument("--mi_normalize", default="min", choices=["min", "max", "geom"])
    p.add_argument("--min_support", type=int, default=30,
                   help="flag (not hide) pairs with fewer joint images")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--n_perm", type=int, default=1000, help="permutation-test iterations (0 to skip)")
    p.add_argument("--class_map", default=DEFAULT_CLASS_MAP,
                   help="class-normalization json (default: intersectional/class_map.json)")
    p.add_argument("--no_class_map", action="store_true", help="disable class normalization")
    p.add_argument("--seed", type=int, default=0)
    o = p.parse_args()
    run(o.dataset, o.generator, o.vqa_model, o.mode, o.attr_mode, o.cluster,
        o.mi_normalize, o.min_support, o.n_boot, o.n_perm, o.seed,
        class_map_path=(None if o.no_class_map else o.class_map))
