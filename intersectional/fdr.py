"""Stage 5 — multiple-testing control for the open-set all-pairs scan.

Scanning every same-cluster attribute pair means hundreds of simultaneous NMI permutation
tests, so per-pair p-values overstate the evidence. We control the False Discovery Rate with
Benjamini-Hochberg over the pre-registered family = pairs with support >= min_support and a
defined p-value (the support filter does not look at the labels' association, so it is
independent of the test statistic under H0 and BH stays valid).

Two-stage permutation: the screening pass (n_perm ~ 1000) has p-resolution 1/(n_perm+1), too
coarse for BH whose strictest cutoff is q*1/m. Pairs with screened p <= q are re-estimated
with a much larger n_perm (pairs with p > q can never be rejected by BH at level q, since BH
rejects only p <= q*k/m <= q, so skipping them is exact, not an approximation). Refinement is
a better Monte Carlo estimate of the same p-value, not data reuse.

``perm_pvalue_fast`` is a vectorized re-implementation of ``scoring.permutation_test_nmi``:
same rng stream, same NMI, same add-one smoothing -> identical output for equal seed/n_perm
(parity-tested), but bincount-based so 20k permutations on ~1k observations take seconds.
"""
import numpy as np

import scoring


def perm_pvalue_fast(obs, n_perm=1000, normalize="min", seed=0):
    """Drop-in fast equivalent of ``scoring.permutation_test_nmi`` (see module docstring)."""
    obs = list(obs)
    if len(obs) < 2:
        return None
    classes_a = sorted({a for a, _ in obs})
    classes_b = sorted({b for _, b in obs})
    ia = {c: i for i, c in enumerate(classes_a)}
    ib = {c: i for i, c in enumerate(classes_b)}
    ka, kb = len(classes_a), len(classes_b)
    a = np.array([ia[x] for x, _ in obs], dtype=np.int64)
    b = np.array([ib[y] for _, y in obs], dtype=np.int64)
    N = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(float)
    observed = scoring._nmi_from_counts(N, normalize)
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        bp = rng.permutation(b)
        Np = np.bincount(a * kb + bp, minlength=ka * kb).reshape(ka, kb).astype(float)
        v = scoring._nmi_from_counts(Np, normalize)
        if v is not None and v >= observed:
            ge += 1
    return round((ge + 1) / (n_perm + 1), 6)


def benjamini_hochberg(pvalues):
    """``{key: p}`` -> ``{key: q}`` (BH adjusted p-values, monotone, clipped to [p, 1]).

    q_i = min over j>=i of (m * p_(j) / j); a pair is an FDR-q discovery iff q_i <= q.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    qvals = {}
    running_min = 1.0
    for rank in range(m, 0, -1):
        key, p = items[rank - 1]
        running_min = min(running_min, p * m / rank)
        qvals[key] = round(min(max(running_min, p), 1.0), 6)
    return qvals


def two_stage_fdr(joint_obs, screened_pvalues, q=0.05, n_perm_refine=20000,
                  normalize="min", seed=0):
    """Refine the screened p-values that could survive BH, then apply BH.

    ``joint_obs``: {key: pooled list of (pred_a, pred_b)} for the test family only.
    ``screened_pvalues``: {key: p from the n_perm~1000 screening pass}.
    Returns ``(qvalues, refined_pvalues)``; refined_pvalues holds the post-refinement p for
    every family member (refined where p <= q, screened value elsewhere).
    """
    refined = dict(screened_pvalues)
    for key, p in screened_pvalues.items():
        if p <= q:
            refined[key] = perm_pvalue_fast(joint_obs[key], n_perm=n_perm_refine,
                                            normalize=normalize, seed=seed)
    return benjamini_hochberg(refined), refined
