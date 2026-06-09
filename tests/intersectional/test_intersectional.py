"""Stage 5 unit tests. CPU-only, no model inference, no network.

Run: ``PYTHONHASHSEED=0 python -m pytest tests/intersectional -q``
(make_plots parity test self-skips if matplotlib/utils are unavailable).
"""
import os
import sys
import json
import math
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "intersectional"))
import pairing   # noqa: E402
import scoring   # noqa: E402


# ----- entropy parity with upstream make_plots (D6) -----
def test_entropy_parity_with_make_plots():
    sys.path.insert(0, ROOT)
    try:
        import make_plots  # noqa: E402  (imports matplotlib + utils.config)
    except Exception as e:  # pragma: no cover - env dependent
        pytest.skip(f"upstream make_plots not importable here: {e}")
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = int(rng.integers(2, 8))
        x = rng.random(k)
        x = x / x.sum()
        assert round(scoring.normalized_entropy(x), 5) == make_plots.entropy(x)


# ----- known-joint sanity (D6/Step 5) -----
def test_independent_uniform_has_zero_nmi():
    obs = [("m", "w"), ("m", "b"), ("f", "w"), ("f", "b")]  # product distribution
    assert scoring.mutual_information(obs) == 0.0


def test_perfect_dependence_has_unit_nmi():
    obs = [("m", "w"), ("m", "w"), ("f", "b"), ("f", "b")]  # A determines B
    assert scoring.mutual_information(obs) == pytest.approx(1.0, abs=1e-6)


def test_constant_attribute_nmi_zero():
    obs = [("m", "w"), ("m", "b"), ("m", "w")]  # A constant -> MI undefined -> 0
    assert scoring.mutual_information(obs) == 0.0


def test_single_cell_joint_is_degenerate():
    obs = [("m", "w"), ("m", "w")]  # one surviving combination
    assert scoring.joint_intensity(obs) is None
    assert scoring.score_pair(obs, n_boot=10)["degenerate"] is True


def test_joint_intensity_in_unit_interval():
    obs = [("m", "w")] * 9 + [("f", "b")]  # skewed but 2 cells
    ji = scoring.joint_intensity(obs)
    assert ji is not None and 0.0 < ji < 1.0


# ----- exclusion mirrors make_plots (D5) -----
def test_excluded_classes_dropped():
    image = {
        "person gender": ["person", "c0", "male"],
        "person race": ["person", "c0", "unknown"],   # dropped
        "person age": ["person", "c0", "other"],       # dropped
    }
    attrs = pairing.per_image_attributes(image)["person"]
    assert attrs == {"gender": "male"}  # only the surviving attribute remains


def test_canonicalization_recovers_attribute():
    assert pairing.canonical_attr("driver age") == "age"
    assert pairing.canonical_attr("pitcher's gender") == "gender"
    assert pairing.canonical_attr("soccer player race") == "race"
    assert pairing.canonical_attr("person gender", mode="raw") == "person gender"


def test_pairing_same_cluster_only():
    vqa = {
        "p/12/0.jpg": {
            "person age": ["person", "c", "young"],
            "person gender": ["person", "c", "male"],
            "scene lighting": ["location", "c", "dim"],  # different cluster -> no cross pair
        }
    }
    joint = pairing.build_pairs(vqa)
    assert ("person", "age", "gender") in joint
    assert all(c == "person" for (c, _a, _b) in joint)  # no person×location pair


def test_bootstrap_ci_wider_on_small_n():
    rng = np.random.default_rng(1)
    big = [("m", "w") if rng.random() < 0.7 else ("f", "b") for _ in range(2000)]
    small = big[:8]
    ci_big = scoring.bootstrap_nmi(big, n_boot=300)
    ci_small = scoring.bootstrap_nmi(small, n_boot=300)
    assert (ci_big[1] - ci_big[0]) < (ci_small[1] - ci_small[0])


# ----- Miller-Madow bias correction -----
def test_miller_madow_below_plugin_on_small_independent_sample():
    # 3x3 independent attributes, tiny n -> plug-in NMI is upward biased; MM corrects it down
    rng = np.random.default_rng(3)
    cls = ["x", "y", "z"]
    obs = [(rng.choice(cls), rng.choice(cls)) for _ in range(18)]
    plugin = scoring.mutual_information(obs)
    mm = scoring.mutual_information_mm(obs)
    assert mm < plugin              # correction strictly reduces the upward bias
    assert (plugin - mm) > 0.02     # by a non-trivial amount
    assert mm < 0.20                # pulled toward the true value (0)


def test_miller_madow_recovers_strong_dependence():
    obs = [("m", "w")] * 500 + [("f", "b")] * 500  # perfectly dependent, large n
    assert scoring.mutual_information_mm(obs) == pytest.approx(1.0, abs=0.05)


# ----- permutation test (H0 = independence) -----
def test_permutation_pvalue_high_when_independent():
    obs = [("m", "w"), ("m", "b"), ("f", "w"), ("f", "b")] * 20  # product distribution
    assert scoring.permutation_test_nmi(obs, n_perm=500) > 0.05


def test_permutation_pvalue_low_when_dependent():
    obs = [("m", "w")] * 50 + [("f", "b")] * 50  # strong dependence
    assert scoring.permutation_test_nmi(obs, n_perm=500) < 0.05


# ----- marginal single-attribute intensity (comparable to baseline make_plots) -----
def test_marginal_intensity_zero_on_uniform_and_high_on_skew():
    import numpy as _np
    assert scoring._marginal_intensity(_np.array([10, 10])) == pytest.approx(0.0, abs=1e-4)
    assert scoring._marginal_intensity(_np.array([99, 1])) > 0.5
    assert scoring._marginal_intensity(_np.array([10, 0])) is None  # single surviving class


# ----- class normalization map -----
def test_class_map_normalizes_and_drops():
    cm = {"gender": {"boy": "male", "girl": "__drop__"}}
    a1 = pairing.per_image_attributes(
        {"person gender": ["person", "c", "boy"]}, class_map=cm)["person"]
    assert a1["gender"] == "male"                      # boy -> male
    a2 = pairing.per_image_attributes(
        {"driver gender": ["person", "c", "girl"]}, class_map=cm)
    assert a2.get("person", {}).get("gender") is None  # girl -> __drop__ -> excluded


def test_default_class_map_merges_race_synonym():
    default = json.load(open(os.path.join(ROOT, "intersectional", "class_map.json")))
    assert default["race"]["white"] == "caucasian"
    assert default["gender"]["boy"] == "male"


def test_baseline_marginals_excludes_unknown():
    sys.path.insert(0, os.path.join(ROOT, "intersectional"))
    import baseline_marginals as bm
    dc = {"person": {"person gender": {"c0": {"male": 30, "female": 10, "unknown": 5}}}}
    out = bm.baseline_marginals(dc, "person", None)
    assert out["gender"]["total"] == 40                # unknown excluded
    assert out["gender"]["intensity"] > 0              # 30:10 is skewed


# ----- prompt quality / caption leakage -----
def test_prompt_quality_detects_attribute_in_caption():
    sys.path.insert(0, os.path.join(ROOT, "intersectional"))
    import prompt_quality as pq
    assert pq.mentions("gender", "A man riding a bike") is True     # gender stated -> not neutral
    assert pq.mentions("gender", "A person near a plane") is False  # neutral prompt
    assert pq.mentions("race", "An asian tourist taking a photo") is True


def test_pair_leakage_rate():
    sys.path.insert(0, os.path.join(ROOT, "intersectional"))
    import prompt_quality as pq
    cmap = {"1": "A man in a kitchen", "2": "A person on a bench", "3": "A woman cooking"}
    out = pq.pair_leakage(["1", "2", "3"], "gender", "race", cmap)
    assert out["captions"] == 3
    assert out["leak_a_rate"] == pytest.approx(2 / 3, abs=1e-3)  # man + woman = 2/3 leak gender
    assert out["clean_captions"] == 1                            # only "A person on a bench"


def test_context_aware_needs_multiple_images_per_caption():
    one = {"1": [("m", "w")], "2": [("f", "b")]}            # n-images=1 -> degenerate
    out1 = scoring.context_aware_metrics(one)
    assert out1["degenerate_fraction"] == 1.0
    assert out1["mean_joint_intensity"] is None
    many = {"1": [("m", "w"), ("m", "b"), ("f", "w"), ("f", "b")],   # several images per caption
            "2": [("m", "w"), ("m", "w"), ("f", "b"), ("f", "b")]}
    out2 = scoring.context_aware_metrics(many)
    assert out2["degenerate_fraction"] == 0.0
    assert out2["mean_joint_intensity"] is not None
    assert out2["mean_mutual_information"] is not None


def test_exclude_leaky_drops_observations():
    cmap = {"7": "A man riding"}     # gender stated in the prompt
    vqa = {"p/7/0.jpg": {"person gender": ["person", "c", "male"],
                         "person race": ["person", "c", "asian"]}}
    kept = pairing.build_pairs(vqa, caption_map=cmap, exclude_leaky=False)
    dropped = pairing.build_pairs(vqa, caption_map=cmap, exclude_leaky=True)
    assert ("person", "gender", "race") in kept
    assert ("person", "gender", "race") not in dropped   # leaky -> excluded
