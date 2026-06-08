"""Stage 5 — full-data single-attribute marginals from ``data_counts.json``.

Context for the pair numbers: a joint pair is measured on a small intersection (e.g. n=48 for
age×gender), whereas these marginals aggregate **all** person observations of each attribute.
This lets the reader see how much single-attribute bias exists overall and whether the pair
subset is representative. Same exclusions / entropy estimator as the baseline ``make_plots.py``.
"""
import os
import sys
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring  # noqa: E402
import pairing  # noqa: E402

DEMO = {"gender", "race", "age"}


def baseline_marginals(data_counts, cluster_filter="person", class_map=None, demo_only=True):
    """Return ``{attr: {intensity, total, distribution}}`` aggregated across every person bias
    whose last token is that attribute. Mirrors make_plots exclusions + optional class map."""
    acc = collections.defaultdict(collections.Counter)
    for cluster, biases in data_counts.items():
        if cluster_filter and cluster != cluster_filter:
            continue
        for bias_name, class_clusters in biases.items():
            last = bias_name.strip().split()[-1].lower()
            if demo_only and last not in DEMO:
                continue
            for _cc, classes in class_clusters.items():
                for cls, cnt in classes.items():
                    if cnt == 0:
                        continue
                    norm = pairing.normalize_class(last, cls, class_map)
                    if norm is None:
                        continue
                    acc[last][norm] += cnt

    out = {}
    for attr, counter in acc.items():
        nz = [c for c in counter.values() if c > 0]
        total = int(sum(nz))
        if total == 0 or len(nz) <= 1:
            out[attr] = {"intensity": None, "total": total, "distribution": dict(counter)}
        else:
            p = np.asarray(nz, dtype=float)
            p = p / p.sum()
            out[attr] = {"intensity": round(1.0 - scoring.normalized_entropy(p), 5),
                         "total": total, "distribution": dict(counter)}
    return out
