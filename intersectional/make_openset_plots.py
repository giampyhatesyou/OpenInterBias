"""Stage 5 — open-set figures. Reads the openset/ctxopen result JSONs and draws the six
figures of the open-set story: how the pairs were discovered (funnel, volcano), what was
discovered (effect-size bar, joint heatmaps), and how the discoveries decompose
(leakage raw->clean, context-aware validation). Self-contained, CPU, matplotlib only.

    python intersectional/make_openset_plots.py     # writes results/intersectional/figures/
"""
import os
import sys
import json
import random
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring  # noqa: E402

BASE = "results/intersectional/{}/generated/sd-xl/llava-1.5-13b/{}"
OUT = "results/intersectional/figures"
DISCOVERED = [  # the 12 clean FDR discoveries, strongest first (RESULTS.md)
    "vehicle:color|size", "dog:breed|size", "person:age|style", "person:age|attire",
    "person:gender|occupation", "dog:age|size", "person:ability|age",
    "person:age|occupation", "person:activity|age", "person:activity|gender",
    "person:age|race", "person:age|gender"]
DEMOGRAPHIC = {"person:age|gender", "person:age|race", "person:gender|race"}
WITHIN_PROMPT = {"dog:breed|size", "dog:age|size"}  # ctx-validated entanglement

C_OPEN, C_DEMO, C_NULL, C_KILL = "#d95f02", "#7570b3", "#bbbbbb", "#e7298a"


def load(ds, name):
    with open(BASE.format(ds, name)) as f:
        return json.load(f)


def fam(results):
    return {k: r for k, r in results["pairs"].items() if "q_value" in r}


def volcano(clean):
    plt.figure(figsize=(9, 6.5))
    crowded = 0  # stagger the labels of the small-effect/high-significance cluster
    for k, r in sorted(fam(clean).items(),
                       key=lambda kv: kv[1]["context_free"]["mutual_information_mm"]):
        cf = r["context_free"]
        p = max(cf.get("nmi_pvalue_refined") or cf["nmi_pvalue"], 1e-5)
        disc = r.get("fdr_discovery")
        y = -np.log10(p)
        plt.scatter(cf["mutual_information_mm"], y,
                    s=18 + cf["support_images"] / 12,
                    c=C_OPEN if disc and k not in DEMOGRAPHIC
                    else C_DEMO if disc else C_NULL,
                    edgecolors="k", linewidths=0.4, zorder=3)
        if disc:
            dy = 3
            if cf["mutual_information_mm"] < 0.07 and y > 4:
                dy = 3 - crowded * 12
                crowded += 1
            plt.annotate(k.replace("person:", ""), (cf["mutual_information_mm"], y),
                         fontsize=7.5, xytext=(7, dy), textcoords="offset points")
    plt.axhline(-np.log10(0.05), color="gray", ls=":", lw=1)
    plt.xlabel("effect size (Miller-Madow NMI)")
    plt.ylabel("-log10 permutation p (refined)")
    plt.title("Open-set all-pairs scan (prompt-leakage filtered):\n"
              "34 testable pairs, 12 BH-FDR discoveries (q≤0.05)")
    for c, l in [(C_OPEN, "open-set discovery"), (C_DEMO, "demographic (closed-set) pair"),
                 (C_NULL, "not significant")]:
        plt.scatter([], [], c=c, edgecolors="k", linewidths=0.4, label=l)
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig(f"{OUT}/volcano_discoveries.png", dpi=200, bbox_inches="tight")
    plt.close()


def effect_bar(clean):
    rows = [(k, fam(clean)[k]["context_free"]["mutual_information_mm"],
             fam(clean)[k]["context_free"]["support_images"]) for k in DISCOVERED
            if k in fam(clean)]
    rows.sort(key=lambda r: r[1])
    ks, vs, sup = zip(*rows)
    cols = [C_DEMO if k in DEMOGRAPHIC else C_OPEN for k in ks]
    plt.figure(figsize=(8, 5.5))
    plt.barh(range(len(ks)), vs, color=cols, edgecolor="k", linewidth=0.4)
    for i, (v, s) in enumerate(zip(vs, sup)):
        plt.text(v + 0.006, i, f"n={s}", va="center", fontsize=7.5)
    plt.yticks(range(len(ks)), ks, fontsize=8.5)
    plt.xlabel("Miller-Madow NMI (context-free, leakage-filtered)")
    plt.title("The 12 FDR discoveries: the hand-picked demographic pairs\n"
              "(purple) are the weakest couplings found")
    plt.savefig(f"{OUT}/discoveries_effect.png", dpi=200, bbox_inches="tight")
    plt.close()


def leakage_dumbbell(raw, clean):
    fr, fc = fam(raw), fam(clean)
    keys = sorted(set(fr) & set(fc),
                  key=lambda k: fr[k]["context_free"]["mutual_information_mm"])
    keys = [k for k in keys if fr[k].get("fdr_discovery") or fc[k].get("fdr_discovery")]
    plt.figure(figsize=(8, 5.5))
    for i, k in enumerate(keys):
        a = fr[k]["context_free"]["mutual_information_mm"]
        b = fc[k]["context_free"]["mutual_information_mm"]
        killed = fr[k].get("fdr_discovery") and not fc[k].get("fdr_discovery")
        plt.plot([a, b], [i, i], color=C_KILL if killed else "#888888", lw=1.4, zorder=2)
        plt.scatter([a], [i], c="#cccccc", edgecolors="k", linewidths=0.4, zorder=3, s=38)
        plt.scatter([b], [i], c=C_KILL if killed else C_OPEN, edgecolors="k",
                    linewidths=0.4, zorder=3, s=38)
    plt.yticks(range(len(keys)), keys, fontsize=8.5)
    plt.xlabel("Miller-Madow NMI  (gray = raw prompts, colored = leaky prompts removed)")
    plt.title("Prompt-quality control changes conclusions:\n"
              "3 raw discoveries (pink) do not survive the leakage filter")
    plt.savefig(f"{OUT}/leakage_effect.png", dpi=200, bbox_inches="tight")
    plt.close()


def ctx_validation():
    res = load("coco_ctxopen", "intersectional_results.json")
    ja = load("coco_ctxopen", "joint_answers.json")
    random.seed(0)
    rows = []
    for k in DISCOVERED:
        ca = res["pairs"].get(k, {}).get("context_aware")
        if not ca:
            continue
        floors = []
        for _ in range(30):
            vals = []
            for lst in ja.get(k, {}).values():
                if len(lst) < 2:
                    continue
                bs = [b for _, b in lst]
                random.shuffle(bs)
                v = scoring.mutual_information([(a, b) for (a, _), b in zip(lst, bs)])
                if v is not None:
                    vals.append(v)
            if vals:
                floors.append(np.mean(vals))
        rows.append((k, ca["mean_mutual_information"] - float(np.mean(floors)),
                     ca["support_captions"]))
    rows.sort(key=lambda r: r[1])
    ks, ex, sup = zip(*rows)
    cols = [C_OPEN if k in WITHIN_PROMPT else C_NULL for k in ks]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(ks)), ex, color=cols, edgecolor="k", linewidth=0.4)
    for i, (e, s) in enumerate(zip(ex, sup)):
        plt.text(max(e, 0) + 0.004, i, f"caps={s}", va="center", fontsize=7.5)
    plt.axvline(0, color="k", lw=0.8)
    plt.yticks(range(len(ks)), ks, fontsize=8.5)
    plt.xlabel("context-aware NMI excess over within-caption permutation floor")
    plt.title("Context-aware validation: only the dog pairs (orange) are\n"
              "within-prompt entanglement; the rest is contextual (prompt-mix) bias")
    plt.savefig(f"{OUT}/ctx_validation.png", dpi=200, bbox_inches="tight")
    plt.close()


def heatmaps(clean_ja):
    picks = ["person:gender|occupation", "dog:breed|size", "person:age|attire"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, k in zip(axes, picks):
        pooled = [tuple(o) for lst in clean_ja[k].values() for o in lst]
        ca = sorted({a for a, _ in pooled})
        cb = sorted({b for _, b in pooled})
        # keep the most frequent classes so labels stay readable
        fa = collections.Counter(a for a, _ in pooled)
        fb = collections.Counter(b for _, b in pooled)
        ca = [c for c, _ in fa.most_common(6)]
        cb = [c for c, _ in fb.most_common(6)]
        M = np.zeros((len(ca), len(cb)))
        for a, b in pooled:
            if a in ca and b in cb:
                M[ca.index(a), cb.index(b)] += 1
        M = M / np.maximum(M.sum(axis=1, keepdims=True), 1)  # row-normalized
        im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cb)), cb, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(ca)), ca, fontsize=8)
        a_name, b_name = k.split(":")[1].split("|")
        ax.set_title(f"{k}  (n={len(pooled)})", fontsize=10)
        ax.set_ylabel(a_name, fontsize=9)
        ax.set_xlabel(f"P({b_name} | {a_name})", fontsize=9)
        for i in range(len(ca)):
            for j in range(len(cb)):
                if M[i, j] >= 0.005:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.55 else "black")
    fig.suptitle("What the discovered couplings look like (row-normalized joints, "
                 "leakage-filtered)", y=1.04)
    plt.savefig(f"{OUT}/joint_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()


def funnel():
    # stage counts: full pool / selection (openset_selection_report) / realized scan /
    # FDR family / discoveries / ctx-validated (see RESULTS.md)
    stages = [
        ("caption proposals (full COCO pool)", 73399),
        ("captions selected (coverage-greedy:\n4213 reused + 2748 generated)", 6961),
        ("attribute pairs realized in VQA", 1957),
        ("pairs testable (support ≥ 30)", 34),
        ("FDR discoveries (raw prompts)", 15),
        ("discoveries after leakage filter", 12),
        ("within-prompt entanglement (ctx-aware)", 2),
    ]
    labels, vals = zip(*stages)
    plt.figure(figsize=(8.5, 5))
    y = range(len(stages))[::-1]
    w = np.log10(np.array(vals) + 1)
    plt.barh(y, w, color=["#999999"] * 3 + [C_DEMO] + [C_OPEN] * 3,
             edgecolor="k", linewidth=0.4)
    for yi, (l, v), wi in zip(y, stages, w):
        plt.text(5.6, yi, f"{v:,}", va="center", fontsize=10, fontweight="bold")
        if wi > 3:  # label fits inside the long bars, after them otherwise
            plt.text(0.07, yi, l, va="center", fontsize=8.5, color="white")
        else:
            plt.text(wi + 0.07, yi, l, va="center", fontsize=8.5)
    plt.xlim(0, 6.4)
    plt.xlabel("log10 scale")
    plt.yticks([])
    plt.title("How the discoveries were made: open-set discovery funnel")
    plt.savefig(f"{OUT}/discovery_funnel.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    raw = load("coco_openset", "intersectional_results.json")
    clean = load("coco_opensetclean", "intersectional_results.json")
    volcano(clean)
    effect_bar(clean)
    leakage_dumbbell(raw, clean)
    ctx_validation()
    heatmaps(load("coco_opensetclean", "joint_answers.json"))
    funnel()
    print("wrote 6 figures to", OUT)
