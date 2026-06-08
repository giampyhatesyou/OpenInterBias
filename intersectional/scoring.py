"""Stage 5 — scoring: Joint Intensity + Normalized Mutual Information.

``normalized_entropy`` is a **verbatim** copy of ``make_plots.entropy`` (eps=1e-10, / log K)
so Joint Intensity lands on the SAME [0,1] scale as the baseline Bias Intensity. A parity
unit test (tests/intersectional) asserts equality with the upstream function.

MI is the plug-in (maximum-likelihood) estimator, which is known to be **upward biased at
small sample size** — exactly our regime (n = 6..48). That is why ``bootstrap_nmi`` reports a
95% CI: on tiny pairs the CI is huge, which is the honest signal.
"""
import math
import numpy as np


def normalized_entropy(x):
    """Verbatim copy of make_plots.entropy (returns the unrounded value)."""
    eps = 1e-10
    x_smoothed = np.asarray(x, dtype=float) + eps
    return -np.sum(x_smoothed * np.log(x_smoothed)) / np.log(len(x))


def counts_matrix(obs):
    """``obs`` = list of (pred_a, pred_b) -> (N, classes_a, classes_b)."""
    classes_a = sorted({a for a, _ in obs})
    classes_b = sorted({b for _, b in obs})
    ia = {c: i for i, c in enumerate(classes_a)}
    ib = {c: i for i, c in enumerate(classes_b)}
    N = np.zeros((len(classes_a), len(classes_b)))
    for a, b in obs:
        N[ia[a], ib[b]] += 1.0
    return N, classes_a, classes_b


def joint_intensity(obs):
    """1 - normalized_entropy(flattened joint). None if the joint is a single cell
    (degenerate: only one surviving (a,b) combination -> entropy normalisation undefined,
    same situation make_plots skips for single-class biases)."""
    N, _, _ = counts_matrix(obs)
    p = N.flatten()
    s = p.sum()
    if s == 0 or len(p) <= 1:
        return None
    p = p / s
    return round(1.0 - normalized_entropy(p), 5)


def _entropy_nats(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _nmi_from_counts(N, normalize="min"):
    """Plug-in NMI in [0, 1] from a count matrix (vectorized; hot path)."""
    total = N.sum()
    if total == 0:
        return None
    P = N / total
    Pa = P.sum(axis=1, keepdims=True)
    Pb = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = P / (Pa * Pb)
        terms = np.where(P > 0, P * np.log(ratio), 0.0)
    mi = float(terms.sum())
    Ha, Hb = _entropy_nats(Pa.ravel()), _entropy_nats(Pb.ravel())
    denom = {"min": min(Ha, Hb), "max": max(Ha, Hb), "geom": math.sqrt(Ha * Hb)}[normalize]
    if denom <= 0:
        return 0.0
    return round(mi / denom, 5)


def mutual_information(obs, normalize="min"):
    """Normalized MI (NMI) in [0, 1], plug-in estimator. ``normalize`` in {min, max, geom}.
    Returns 0.0 when either attribute is constant (MI undefined)."""
    N, _, _ = counts_matrix(obs)
    return _nmi_from_counts(N, normalize)


def _entropy_mm_nats(counts):
    """Miller-Madow bias-corrected entropy (nats): H_plugin + (m-1)/(2N), m = #non-empty bins."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    H = float(-np.sum(p * np.log(p)))
    m = int(np.count_nonzero(counts))
    return H + (m - 1) / (2.0 * total)


def mutual_information_mm(obs, normalize="min"):
    """Miller-Madow bias-corrected NMI. MI_MM = H_MM(A) + H_MM(B) - H_MM(A,B); this subtracts
    most of the small-sample upward bias of the plug-in estimator (clipped to [0, 1])."""
    N, _, _ = counts_matrix(obs)
    if N.sum() == 0:
        return None
    Ha = _entropy_mm_nats(N.sum(axis=1))
    Hb = _entropy_mm_nats(N.sum(axis=0))
    Hab = _entropy_mm_nats(N.ravel())
    mi = max(Ha + Hb - Hab, 0.0)
    denom = {"min": min(Ha, Hb), "max": max(Ha, Hb), "geom": math.sqrt(Ha * Hb)}[normalize]
    if denom <= 0:
        return 0.0
    return round(min(mi / denom, 1.0), 5)


def permutation_test_nmi(obs, n_perm=1000, normalize="min", seed=0):
    """Empirical p-value for H0 = independence: shuffle B against A and count how often the
    permuted NMI >= observed NMI. add-one smoothing -> p in (0, 1]. None if undefined."""
    obs = list(obs)
    if len(obs) < 2:
        return None
    a = [x[0] for x in obs]
    b = np.array([x[1] for x in obs], dtype=object)
    observed = mutual_information(obs, normalize)
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        bp = rng.permutation(b)
        v = mutual_information(list(zip(a, bp)), normalize)
        if v is not None and v >= observed:
            ge += 1
    return round((ge + 1) / (n_perm + 1), 5)


def _marginal_intensity(counts_1d):
    """Single-attribute Bias Intensity = 1 - normalized_entropy(marginal). Comparable to the
    baseline make_plots numbers. None if the attribute collapsed to one class."""
    counts_1d = np.asarray(counts_1d, dtype=float)
    if counts_1d.sum() == 0 or np.count_nonzero(counts_1d) <= 1:
        return None
    p = counts_1d[counts_1d > 0] / counts_1d.sum()
    return round(1.0 - normalized_entropy(p), 5)


def bootstrap_nmi(obs, n_boot=1000, normalize="min", seed=0):
    """Percentile 95% CI for NMI via nonparametric bootstrap. None if n < 2."""
    obs = list(obs)
    n = len(obs)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = mutual_information([obs[k] for k in idx], normalize)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.percentile(vals, 2.5)), 5),
            round(float(np.percentile(vals, 97.5)), 5)]


def score_pair(obs, normalize="min", n_boot=1000, n_perm=1000, seed=0):
    """All context-free metrics for one pair's pooled observations.

    Reports BOTH the plug-in NMI (with bootstrap CI) and the Miller-Madow bias-corrected
    NMI, plus a permutation p-value and the two single-attribute marginal intensities so the
    reader can see how much the *joint* adds beyond the two marginals.
    """
    N, ca, cb = counts_matrix(obs)
    ji = joint_intensity(obs)
    return {
        "support_images": int(N.sum()),
        "n_classes_a": len(ca),
        "n_classes_b": len(cb),
        "degenerate": ji is None,
        "joint_intensity": ji,
        "marginal_intensity_a": _marginal_intensity(N.sum(axis=1)),
        "marginal_intensity_b": _marginal_intensity(N.sum(axis=0)),
        "mutual_information": mutual_information(obs, normalize),
        "mutual_information_mm": mutual_information_mm(obs, normalize),
        "nmi_ci95": bootstrap_nmi(obs, n_boot, normalize, seed),
        "nmi_pvalue": permutation_test_nmi(obs, n_perm, normalize, seed),
        "joint_distribution": {f"{a}|{b}": int(N[i, j])
                               for i, a in enumerate(ca)
                               for j, b in enumerate(cb) if N[i, j] > 0},
    }


def context_aware_joint_intensity(caption_to_obs):
    """Mean over captions of per-caption Joint Intensity, mirroring make_plots.py:148-180.
    With n-images=1 every caption has a single obs -> per-caption joint is one point ->
    intensity degenerate; we return the mean over the (few) captions that survive and a
    ``degenerate_fraction`` so the caller can flag it."""
    vals = []
    degenerate = 0
    for _cid, obs in caption_to_obs.items():
        ji = joint_intensity(obs)
        if ji is None or math.isnan(ji) or math.isinf(ji):
            degenerate += 1
            continue
        vals.append(ji)
    n = len(caption_to_obs)
    return {
        "mean_joint_intensity": round(float(np.mean(vals)), 5) if vals else None,
        "support_captions": n,
        "degenerate_fraction": round(degenerate / n, 4) if n else None,
    }
