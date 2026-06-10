"""Pairing: read ``vqa_answers.json`` -> per-pair joint observations.

Read-only, CPU-only. Mirrors ``make_plots.py`` class exclusions so the joint analysis is
comparable to the single-attribute baseline.

The VQA file keys biases by free-text ``bias_name`` (4554 distinct on COCO, e.g. ``"driver age"``,
``"soccer player race"``); pairing on the raw name yields ~2-caption support per pair. We therefore
canonicalize the attribute by its last token (age/gender/race/...) within a cluster, which recovers
usable support. Pass ``attr_mode="raw"`` to keep the raw name.
"""
import json
import os
import itertools
import collections

# Identical to make_plots.py UNK_CLASS / OTHER_CLASS / NON_BINRAY_CLASS.
EXCLUDED_CLASSES = {"unknown", "other", "non-binary"}


def canonical_attr(bias_name, mode="last_token", attr_map=None):
    """Map a free-text ``bias_name`` to a canonical attribute name.

    ``last_token``: ``"driver age" -> "age"``, ``"person race" -> "race"``,
    ``"pitcher's gender" -> "gender"``. ``raw``: identity (no canonicalization).
    ``attr_map`` (attr_synonyms.json) optionally merges reviewed synonyms AFTER the
    last token (``"clothing" -> "attire"``), so the open-set tail is less fragmented.
    """
    if mode == "raw":
        return bias_name
    last = bias_name.strip().split()[-1].lower()
    return attr_map.get(last, last) if attr_map else last


# class normalization sentinel: a class mapped to this is treated like an excluded class
DROP = "__drop__"


def load_attr_map(path):
    """Load the optional attribute-synonym map (flat ``{last_token: canonical}``).
    ``None``/missing path -> ``None`` (no merging)."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        m = json.load(f)
    m.pop("_comment", None)
    return m


def load_class_map(path):
    """Load an optional per-attribute class-normalization map (``{attr: {raw: canonical}}``).
    ``None``/missing path -> ``{}`` (no normalization). A class mapped to ``"__drop__"`` is
    excluded like ``unknown``. The map is keyed by the canonical (last-token) attribute, so it
    applies regardless of the raw bias_name ("driver gender" and "person gender" share it)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def normalize_class(attr_last_token, pred, class_map):
    """Apply the class map; return the (possibly remapped) class, or ``None`` if it drops out."""
    if class_map:
        pred = class_map.get(attr_last_token, {}).get(pred, pred)
    if pred in EXCLUDED_CLASSES or pred == DROP:
        return None
    return pred


def load_vqa_answers(path):
    with open(path, "r") as f:
        return json.load(f)


def per_image_attributes(image_biases, attr_mode="last_token", cluster_filter=None, class_map=None,
                         attr_map=None):
    """Return ``{cluster: {attr: pred}}`` for a single image, after class exclusion + optional
    class normalization.

    ``image_biases`` is the per-image dict from vqa_answers.json:
    ``{bias_name: [cluster(refer_to), class_cluster, predicted_class]}``.
    If several raw biases canonicalize to the same attr in one image, the first
    non-excluded prediction wins (rare; logged as a v1 simplification).
    """
    out = collections.defaultdict(dict)
    for bias_name, val in image_biases.items():
        cluster, _class_cluster, pred = val[0], val[1], val[-1]
        if cluster_filter is not None and cluster != cluster_filter:
            continue
        attr = canonical_attr(bias_name, "last_token", attr_map)
        pred = normalize_class(attr, pred, class_map)
        if pred is None:
            continue
        if attr_mode == "raw":
            attr = bias_name
        out[cluster].setdefault(attr, pred)
    return out


try:
    import prompt_quality as _pq
except ImportError:  # pragma: no cover
    _pq = None


def build_pairs(vqa_answers, attr_mode="last_token", cluster_filter=None, class_map=None,
                caption_map=None, exclude_leaky=False, leak_index=None, attr_map=None):
    """Return ``joint[(cluster, a, b)] = {caption_id: [(pred_a, pred_b), ...]}``.

    ``a, b`` are the two attribute names sorted; pairs are within the same cluster
    (same ``refer_to``), so they are semantically intersectional (person gender x
    person race), not mere co-occurrence (person x kitchen). ``caption_id`` is the
    second-to-last path component of the image key (identical to make_plots.py:132).

    If ``exclude_leaky``, a joint observation is dropped when the caption (the prompt)
    lexically states one of the two attributes (prompt-quality control). With a
    ``leak_index`` (prompt_quality.build_leak_index) the check covers every open-set
    attribute via the proposed class labels; with only a ``caption_map`` it falls back to
    the demographic ATTR_WORDS lists. The leakage rate is reported separately by
    run_analysis regardless.
    """
    joint = collections.defaultdict(lambda: collections.defaultdict(list))
    for image_key, image_biases in vqa_answers.items():
        caption_id = image_key.split("/")[-2]
        cap = caption_map.get(caption_id, "") if caption_map else ""
        by_cluster = per_image_attributes(image_biases, attr_mode, cluster_filter, class_map,
                                          attr_map)
        for cluster, attrs in by_cluster.items():
            for a, b in itertools.combinations(sorted(attrs), 2):
                if exclude_leaky and _pq:
                    if leak_index is not None:
                        if (_pq.leaks(a, caption_id, leak_index, attr_map)
                                or _pq.leaks(b, caption_id, leak_index, attr_map)):
                            continue
                    elif caption_map and (_pq.mentions(a, cap) or _pq.mentions(b, cap)):
                        continue
                joint[(cluster, a, b)][caption_id].append((attrs[a], attrs[b]))
    return joint


def pair_key(cluster, a, b):
    """Stable, unambiguous human-readable id. ``|`` separator avoids clashing with
    class names containing ``x``; the cluster prefix disambiguates the same
    attribute pair across clusters (e.g. person vs child)."""
    return f"{cluster}:{a}|{b}"


def write_joint_answers(joint, path):
    """Persist the joint observations (NEW artefact; never edits upstream files)."""
    serial = {pair_key(c, a, b): dict(caps) for (c, a, b), caps in joint.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serial, f)
    return serial
