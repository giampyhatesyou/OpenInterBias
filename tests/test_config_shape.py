"""Invariants on utils/config.py shape.

We do NOT import the heavy upstream modules (which would pull torch / diffusers
/ modelscope and slow tests down massively). We import only what's needed.

If these keys disappear, every entry-point breaks — so this test is a
canary, not a guarantee of correctness.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def config_module():
    """Import utils.config with the repo root on sys.path.

    NOTE: this import currently triggers imports of utils.datasets and
    utils.bias_proposals_manager, which in turn pull heavy deps (pycocotools,
    fiftyone, torch). If the test machine lacks them, we skip rather than fail.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        mod = importlib.import_module("utils.config")
    except Exception as e:  # pragma: no cover — environment-dependent
        pytest.skip(f"utils.config could not be imported: {e!r}")
    return mod


def test_bias_proposal_setting_keys(config_module) -> None:
    s = config_module.BIAS_PROPOSAL_SETTING
    for k in ("seed", "batch_size", "max_seq_len", "llama2", "coco", "flickr_30k"):
        assert k in s, f"BIAS_PROPOSAL_SETTING missing key: {k}"
    for ds in ("coco", "flickr_30k"):
        for k in ("path", "n_prompts_per_image", "dataset", "bias_proposal_module", "system_prompt"):
            assert k in s[ds], f"BIAS_PROPOSAL_SETTING[{ds!r}] missing key: {k}"


def test_gen_setting_keys(config_module) -> None:
    g = config_module.GEN_SETTING
    for k in ("generators", "save_path", "inference_steps", "max_prompts_per_bias", "filter_threshold", "hard_threshold", "merge_threshold"):
        assert k in g, f"GEN_SETTING missing key: {k}"
    for gen in ("sd-xl", "sd-1.5", "sd-2"):
        assert gen in g["generators"], f"generator '{gen}' missing from GEN_SETTING['generators']"


def test_vqa_setting_keys(config_module) -> None:
    v = config_module.VQA_SETTING
    for k in ("vqa_models", "filter_threshold", "hard_threshold", "merge_threshold", "UNK_CLASS", "save_path"):
        assert k in v, f"VQA_SETTING missing key: {k}"
    assert "llava-1.5-13b" in v["vqa_models"], "default VQA model 'llava-1.5-13b' missing"


def test_bias_proposal_system_prompt_shape(config_module) -> None:
    sp = config_module.BIAS_PROPOSAL_SYSTEM_PROMPT
    assert "std_domain" in sp and "facial_domain" in sp
    for domain in ("std_domain", "facial_domain"):
        assert len(sp[domain]) >= 3, "system prompt should be system + ≥1 user/assistant pair"
        assert sp[domain][0]["role"] == "system"
