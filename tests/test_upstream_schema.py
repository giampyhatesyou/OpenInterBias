"""Invariants on the upstream proposed_biases JSON shape.

Future intersectional code consumes this shape. If these invariants break,
the increment will silently misbehave — better to fail loudly here.
"""

from __future__ import annotations


def test_root_has_bias_proposal_key(proposed_biases_coco_data: dict) -> None:
    assert "bias_proposal" in proposed_biases_coco_data
    assert isinstance(proposed_biases_coco_data["bias_proposal"], list)


def test_entries_have_required_top_fields(proposed_biases_coco_data: dict) -> None:
    required = {"caption_id", "image_id", "caption", "proposed_biases"}
    for entry in proposed_biases_coco_data["bias_proposal"]:
        missing = required - set(entry.keys())
        assert not missing, f"entry missing keys {missing}: {entry}"


def test_proposed_biases_payload_shape(proposed_biases_coco_data: dict) -> None:
    """proposed_biases is a dict with a 'bias' list of single-attribute records."""
    for entry in proposed_biases_coco_data["bias_proposal"]:
        pb = entry["proposed_biases"]
        assert isinstance(pb, dict), f"proposed_biases must be dict, got {type(pb)}"
        assert "bias" in pb, "missing 'bias' key inside proposed_biases"
        assert isinstance(pb["bias"], list)


def test_each_bias_has_minimum_fields(proposed_biases_coco_data: dict) -> None:
    required_bias_fields = {"name", "refer_to", "classes", "question"}
    for entry in proposed_biases_coco_data["bias_proposal"]:
        for b in entry["proposed_biases"]["bias"]:
            missing = required_bias_fields - set(b.keys())
            assert not missing, f"bias entry missing {missing}: {b}"
            assert isinstance(b["classes"], list)
            assert len(b["classes"]) >= 2, "need at least 2 classes for a bias"


def test_intersectional_pairing_assumption(proposed_biases_coco_data: dict) -> None:
    """At least one caption proposes ≥2 distinct biases with same refer_to.

    This is the *minimum* precondition for pairwise intersectional analysis.
    If this fails on real data, no pairing can be derived post-hoc.
    """
    found_pairable_caption = False
    for entry in proposed_biases_coco_data["bias_proposal"]:
        biases = entry["proposed_biases"]["bias"]
        by_refer_to: dict[str, set[str]] = {}
        for b in biases:
            by_refer_to.setdefault(b["refer_to"], set()).add(b["name"])
        if any(len(names) >= 2 for names in by_refer_to.values()):
            found_pairable_caption = True
            break
    assert found_pairable_caption, (
        "No caption in the toy file has ≥2 biases with the same refer_to. "
        "Pairwise intersectional analysis would be empty on this data."
    )
