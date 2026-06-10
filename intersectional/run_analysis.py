"""Stage 5 — CLI entry point.

Reads ``results/VQA/<dataset>/generated/<gen>/<vqa>/vqa_answers.json``, forms same-cluster
attribute pairs, scores them, and writes (NEW, never edits upstream):
- ``joint_answers.json``            (per-pair joint observations)
- ``intersectional_results.json``   (metrics + support + bootstrap CI per pair)
- ``intersectional_results.md``     (human-readable ranked summary)

CPU-only, deterministic. Same CLI surface as the upstream stages.

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
import prompt_quality as pq  # noqa: E402
import fdr  # noqa: E402

DEFAULT_CLASS_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_map.json")
DEFAULT_ATTR_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attr_synonyms.json")


def results_dir(dataset, generator, vqa_model, mode):
    if mode == "original":
        vqa = f"results/VQA/{dataset}/{mode}/{vqa_model}"
    else:
        vqa = f"results/VQA/{dataset}/{mode}/{generator}/{vqa_model}"
    out = vqa.replace("results/VQA/", "results/intersectional/")
    return vqa, out


def run(dataset, generator, vqa_model, mode, attr_mode, cluster_filter,
        normalize, min_support, n_boot, n_perm, seed, class_map_path=DEFAULT_CLASS_MAP,
        proposed_biases_path=None, exclude_leaky=False, fdr_q=0.0, n_perm_refine=20000,
        openset_leak=True, attr_map_path=DEFAULT_ATTR_MAP):
    vqa_path, out_path = results_dir(dataset, generator, vqa_model, mode)
    vqa_answers = pairing.load_vqa_answers(os.path.join(vqa_path, "vqa_answers.json"))
    class_map = pairing.load_class_map(class_map_path)
    attr_map = pairing.load_attr_map(attr_map_path)

    # prompt quality: the caption is the generation prompt. Map caption_id -> caption, and
    # build the open-set leak index (class-label based; covers every proposed attribute).
    caption_map = pq.load_caption_map(proposed_biases_path) if proposed_biases_path else None
    leak_index = (pq.build_leak_index(proposed_biases_path, attr_map)
                  if proposed_biases_path and openset_leak else None)

    joint = pairing.build_pairs(vqa_answers, attr_mode=attr_mode, cluster_filter=cluster_filter,
                                class_map=class_map, caption_map=caption_map,
                                exclude_leaky=exclude_leaky, leak_index=leak_index,
                                attr_map=attr_map)
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
    _unkey = {}  # pair_key string -> joint tuple key (for the FDR pass below)
    for (cluster, a, b), caps in joint.items():
        _unkey[pairing.pair_key(cluster, a, b)] = (cluster, a, b)
        pooled = [obs for lst in caps.values() for obs in lst]
        cf = scoring.score_pair(pooled, normalize=normalize, n_boot=n_boot,
                                n_perm=n_perm, seed=seed)
        ca = scoring.context_aware_metrics(caps, normalize)
        pqual = (pq.pair_leakage(list(caps.keys()), a, b, caption_map, leak_index=leak_index)
                 if caption_map else None)
        results[pairing.pair_key(cluster, a, b)] = {
            "refer_to": cluster,
            "attr_a": a,
            "attr_b": b,
            "support_captions": len(caps),
            "below_min_support": cf["support_images"] < min_support,
            "context_free": cf,
            "context_aware": ca,  # degenerate at n-images=1 (single image per caption)
            "prompt_quality": pqual,  # caption leakage of attr_a / attr_b
        }

    # FDR over the all-pairs scan: family = pairs at/above min_support with a defined p
    # (support is independent of the association under H0, so the filter is pre-registered).
    if fdr_q:
        family = {k: r for k, r in results.items()
                  if not r["below_min_support"]
                  and r["context_free"].get("nmi_pvalue") is not None}
        fam_obs = {k: [obs for lst in joint[_unkey[k]].values() for obs in lst]
                   for k in family}
        screened = {k: family[k]["context_free"]["nmi_pvalue"] for k in family}
        qvals, refined = fdr.two_stage_fdr(fam_obs, screened, q=fdr_q,
                                           n_perm_refine=n_perm_refine,
                                           normalize=normalize, seed=seed)
        for k in family:
            results[k]["context_free"]["nmi_pvalue_refined"] = refined[k]
            results[k]["q_value"] = qvals[k]
            results[k]["fdr_discovery"] = qvals[k] <= fdr_q

    config = {
        "dataset": dataset, "generator": generator, "vqa_model": vqa_model, "mode": mode,
        "attr_mode": attr_mode, "cluster_filter": cluster_filter, "mi_normalize": normalize,
        "min_support": min_support, "n_boot": n_boot, "n_perm": n_perm, "seed": seed,
        "class_map": class_map_path if class_map else None,
        "attr_map": attr_map_path if attr_map else None,
        "proposed_biases": proposed_biases_path, "exclude_leaky": exclude_leaky,
        "leak_check": ("openset_classes" if leak_index is not None
                       else ("demographic_words" if caption_map else None)),
        "fdr_q": fdr_q or None, "n_perm_refine": n_perm_refine if fdr_q else None,
        "n_tests_family": sum(1 for r in results.values() if "q_value" in r) if fdr_q else None,
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
        pqual = r.get("prompt_quality")
        rows.append({
            "key": key, "refer_to": r["refer_to"], "support": cf["support_images"],
            "nmi": cf["mutual_information"], "nmi_mm": cf.get("mutual_information_mm"),
            "ci": cf["nmi_ci95"], "pval": cf.get("nmi_pvalue_refined", cf.get("nmi_pvalue")),
            "ji": cf["joint_intensity"], "low": r["below_min_support"],
            "leak": pqual["leak_any_rate"] if pqual else None,
            "q": r.get("q_value"), "disc": r.get("fdr_discovery"),
        })
    # rank by support so the trustworthy pairs are on top
    return sorted(rows, key=lambda x: x["support"], reverse=True)


def _md_key(key):
    """Pair keys contain ``|`` (pair_key separator), which breaks GFM table cells even
    inside code spans; escape for display only."""
    return key.replace("|", "\\|")


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
    if cfg.get("fdr_q"):
        disc = [d for d in _ranked(payload) if d.get("disc")]
        lines.append(f"### FDR discoveries (Benjamini-Hochberg, q ≤ {cfg['fdr_q']}, "
                     f"family of {cfg['n_tests_family']} pairs with support ≥ {cfg['min_support']})\n")
        if disc:
            lines += ["| pair | support | NMI_MM | p (refined) | q | leak |",
                      "|---|---:|---:|---:|---:|---:|"]
            for d in sorted(disc, key=lambda x: -(x["nmi_mm"] or 0)):
                leak = f"{d['leak']:.0%}" if d["leak"] is not None else "—"
                lines.append(f"| `{_md_key(d['key'])}` | {d['support']} | {d['nmi_mm']} "
                             f"| {d['pval']} | {d['q']} | {leak} |")
            lines.append("")
        else:
            lines.append("No pair survives FDR correction: no evidence of intersectional "
                         "coupling anywhere in the scanned pair space.\n")
    lines += [
             "**Read support first.** Pairs below `min_support` are flagged ⚠ — their NMI is",
             "dominated by small-sample bias. Trust `NMI_MM` (Miller-Madow, bias-corrected) and",
             "`p` (permutation test, H0=independence) over the raw plug-in `NMI`.\n",
             "| pair | refer_to | support | NMI | NMI_MM | p | NMI 95% CI | Joint Int | leak | |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for d in _ranked(payload):
        flag = "⚠ low" if d["low"] else "ok"
        ci = d["ci"]
        ci_s = f"[{ci[0]}, {ci[1]}]" if ci else "—"
        g = lambda v: v if v is not None else "—"  # noqa: E731
        leak = f"{d['leak']:.0%}" if d["leak"] is not None else "—"
        lines.append(f"| `{_md_key(d['key'])}` | {d['refer_to']} | {d['support']} | {g(d['nmi'])} "
                     f"| {g(d['nmi_mm'])} | {g(d['pval'])} | {ci_s} | {g(d['ji'])} | {leak} | {flag} |")

    # context-aware (per-caption average) — only informative with n-images > 1
    g = lambda v: v if v is not None else "—"  # noqa: E731
    ca_rows = [d for d in _ranked(payload) if not d["low"]]
    if ca_rows:
        lines += ["", "### Context-aware (per-caption average; meaningful only with n-images>1)",
                  "| pair | captions | CA Joint Intensity | CA NMI | degenerate frac |",
                  "|---|---:|---:|---:|---:|"]
        for d in ca_rows:
            ca = payload["pairs"][d["key"]]["context_aware"]
            lines.append(f"| `{_md_key(d['key'])}` | {ca['support_captions']} "
                         f"| {g(ca['mean_joint_intensity'])} "
                         f"| {g(ca['mean_mutual_information'])} | {g(ca['degenerate_fraction'])} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_summary(payload, min_support):
    cfg = payload["_config"]
    print(f"\n=== Stage 5 summary ({cfg['n_pairs']} pairs) ===")
    if cfg.get("fdr_q"):
        disc = [d for d in _ranked(payload) if d.get("disc")]
        print(f"FDR (BH q<={cfg['fdr_q']}, family={cfg['n_tests_family']}): "
              f"{len(disc)} discoveries")
        for d in sorted(disc, key=lambda x: -(x["nmi_mm"] or 0)):
            print(f"  * {d['key']:38s} sup={d['support']:<5d} NMI_MM={d['nmi_mm']} "
                  f"p={d['pval']} q={d['q']}")
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
    p.add_argument("--attr_map", default=DEFAULT_ATTR_MAP,
                   help="attribute-synonym json (default: intersectional/attr_synonyms.json)")
    p.add_argument("--no_attr_map", action="store_true", help="disable attribute-synonym merging")
    p.add_argument("--proposed_biases", default=None,
                   help="path to the stage-1 proposals JSON (enables prompt-quality / leakage report)")
    p.add_argument("--exclude_leaky", action="store_true",
                   help="drop joint observations whose prompt lexically states one of the attributes")
    p.add_argument("--demographic_leak_only", action="store_true",
                   help="leakage check via demographic ATTR_WORDS only (legacy), instead of "
                        "the open-set class-label index")
    p.add_argument("--fdr_q", type=float, default=0.0,
                   help="Benjamini-Hochberg FDR level for the all-pairs scan (0 = off). "
                        "Family = pairs with support >= min_support.")
    p.add_argument("--n_perm_refine", type=int, default=20000,
                   help="permutations for the second-stage refinement of candidate p-values")
    p.add_argument("--seed", type=int, default=0)
    o = p.parse_args()
    run(o.dataset, o.generator, o.vqa_model, o.mode, o.attr_mode, o.cluster,
        o.mi_normalize, o.min_support, o.n_boot, o.n_perm, o.seed,
        class_map_path=(None if o.no_class_map else o.class_map),
        proposed_biases_path=o.proposed_biases, exclude_leaky=o.exclude_leaky,
        fdr_q=o.fdr_q, n_perm_refine=o.n_perm_refine,
        openset_leak=not o.demographic_leak_only,
        attr_map_path=(None if o.no_attr_map else o.attr_map))
