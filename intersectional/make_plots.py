"""Stage 5 — figures. Reads ``intersectional_results.json`` and draws two bar charts
(NMI and Joint Intensity), styled to match the upstream ``make_plots.py`` for visual
comparability. Only pairs with ``support >= --min_support`` are plotted; the support count
is annotated on each bar so low-power pairs are obvious.

Decoupled from scoring (reads the JSON), so it is the only module that needs matplotlib.

    python intersectional/make_plots.py --dataset coco --generator sd-xl \
        --vqa_model llava-1.5-13b --mode generated --min_support 30
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _out_path(dataset, generator, vqa_model, mode):
    if mode == "original":
        return f"results/intersectional/{dataset}/{mode}/{vqa_model}"
    return f"results/intersectional/{dataset}/{mode}/{generator}/{vqa_model}"


def _bar(labels, values, supports, title, ylabel, path, yerr=None):
    plt.figure(figsize=(max(10, len(labels) * 1.2), 9))
    x = np.arange(len(labels))
    bars = plt.bar(x, values, color="#C5E898", alpha=0.95, edgecolor="#7f8c8d", width=0.5)
    if yerr is not None:
        plt.errorbar(x, values, yerr=yerr, fmt="none", ecolor="#c0392b",
                     elinewidth=1.5, capsize=5, label="95% CI")
        plt.legend(fontsize=13, loc="upper left")
    for rect, s in zip(bars, supports):
        plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                 f"n={s}", ha="center", va="bottom", fontsize=11)
    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=13)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, format="png", bbox_inches="tight")
    plt.close()
    print("wrote", path)


def _heatmap(joint_distribution, title, path):
    """Joint count matrix as a heatmap, for a single pair (the concrete picture of the bias)."""
    rows = sorted({k.split("|")[0] for k in joint_distribution})
    cols = sorted({k.split("|")[1] for k in joint_distribution})
    M = np.zeros((len(rows), len(cols)))
    for k, v in joint_distribution.items():
        a, b = k.split("|")
        M[rows.index(a), cols.index(b)] = v
    plt.figure(figsize=(max(6, len(cols) * 1.3), max(5, len(rows) * 1.0)))
    plt.imshow(M, cmap="YlGn", aspect="auto")
    plt.colorbar(label="images")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=12)
    plt.yticks(range(len(rows)), rows, fontsize=12)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if M[i, j] > 0:
                plt.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=12)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(path, format="png", bbox_inches="tight")
    plt.close()
    print("wrote", path)


def _scatter_mi_vs_marginals(payload, cluster, min_support, path):
    """Key diagnostic: NMI (dependence) vs the two attributes' summed marginal intensity.
    Points high on y = genuine intersectional coupling; points near y=0 = the joint is just the
    product of the marginals. Color = permutation significance, size = support."""
    xs, ys, ss, sig, labels = [], [], [], [], []
    for key, r in payload["pairs"].items():
        if cluster is not None and r["refer_to"] != cluster:
            continue
        cf = r["context_free"]
        ma, mb, nmi = cf.get("marginal_intensity_a"), cf.get("marginal_intensity_b"), cf["mutual_information"]
        if None in (ma, mb, nmi) or cf["support_images"] < min_support:
            continue
        xs.append(ma + mb); ys.append(nmi); ss.append(cf["support_images"])
        p = cf.get("nmi_pvalue")
        sig.append(p is not None and p < 0.05)
        labels.append(key.split(":", 1)[-1])
    if not xs:
        return
    plt.figure(figsize=(10, 8))
    for x, y, s, sg, lab in zip(xs, ys, ss, sig, labels):
        plt.scatter(x, y, s=30 + s * 4, c=("#c0392b" if sg else "#95a5a6"),
                    alpha=0.8, edgecolors="k", linewidths=0.5)
        plt.annotate(lab, (x, y), fontsize=9, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("sum of single-attribute intensities  (marginal bias)", fontsize=13)
    plt.ylabel("NMI  (intersectional coupling)", fontsize=13)
    plt.title(f"{cluster or 'all'}: does the joint add beyond the marginals?  "
              f"(red = perm. p<0.05; size ∝ support)", fontsize=13)
    plt.grid(linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, format="png", bbox_inches="tight")
    plt.close()
    print("wrote", path)


def main(dataset, generator, vqa_model, mode, min_support, cluster=None):
    out = _out_path(dataset, generator, vqa_model, mode)
    with open(os.path.join(out, "intersectional_results.json")) as f:
        payload = json.load(f)

    rows = []
    for key, r in payload["pairs"].items():
        if cluster is not None and r["refer_to"] != cluster:
            continue
        cf = r["context_free"]
        if cf["support_images"] >= min_support and cf["mutual_information"] is not None:
            ci = cf["nmi_ci95"] or [cf["mutual_information"], cf["mutual_information"]]
            rows.append((key, cf["support_images"], cf["mutual_information"],
                         cf["joint_intensity"], ci))
    if not rows:
        print(f"No pair reaches min_support={min_support}; nothing to plot. "
              f"This itself is the finding (see STAGE5_LOG §2) — scale up generation.")
        return

    rows.sort(key=lambda x: x[2])  # by NMI ascending (matches upstream sort direction)
    labels = [r[0].split(":", 1)[-1] for r in rows]
    supports = [r[1] for r in rows]
    nmi = np.array([r[2] for r in rows])
    # asymmetric error bars from the bootstrap CI, clipped to be non-negative
    lo = np.clip(nmi - np.array([r[4][0] for r in rows]), 0, None)
    hi = np.clip(np.array([r[4][1] for r in rows]) - nmi, 0, None)
    suffix = f"_{cluster}" if cluster else ""
    _bar(labels, list(nmi), supports,
         f"{dataset}{(' — ' + cluster) if cluster else ''} intersectional NMI (support ≥ {min_support})",
         "Normalized Mutual Information",
         os.path.join(out, f"intersectional_nmi{suffix}.png"), yerr=[lo, hi])
    _bar(labels, [r[3] if r[3] is not None else 0 for r in rows], supports,
         f"{dataset}{(' — ' + cluster) if cluster else ''} intersectional Joint Intensity (support ≥ {min_support})",
         "Joint Intensity", os.path.join(out, f"intersectional_joint_intensity{suffix}.png"))

    # heatmap of the joint distribution for the best-supported pair (concrete picture)
    best = max(rows, key=lambda r: r[1])
    jd = payload["pairs"][best[0]]["context_free"]["joint_distribution"]
    _heatmap(jd, f"{best[0]}  (n={best[1]}, NMI={best[2]})",
             os.path.join(out, f"intersectional_heatmap{suffix}.png"))

    # key diagnostic: intersectional coupling vs marginal bias
    _scatter_mi_vs_marginals(payload, cluster, min_support,
                             os.path.join(out, f"intersectional_mi_vs_marginals{suffix}.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", default="coco")
    p.add_argument("--generator", default="sd-xl")
    p.add_argument("--vqa_model", default="llava-1.5-13b")
    p.add_argument("--mode", default="generated", choices=["generated", "original"])
    p.add_argument("--min_support", type=int, default=30)
    p.add_argument("--cluster", default=None, help="restrict figure to one refer_to, e.g. 'person'")
    o = p.parse_args()
    main(o.dataset, o.generator, o.vqa_model, o.mode, o.min_support, o.cluster)
