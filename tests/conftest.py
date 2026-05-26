"""Shared pytest fixtures.

Kept tiny on purpose: at this stage the tests only need filesystem paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def proposed_biases_coco_path(repo_root: Path) -> Path:
    """Path to the small COCO proposed_biases JSON shipped with the fork."""
    p = repo_root / "proposed_biases" / "coco" / "3" / "coco_train.json"
    if not p.is_file():
        pytest.skip(f"Toy proposed_biases JSON not present: {p}")
    return p


@pytest.fixture(scope="session")
def proposed_biases_coco_data(proposed_biases_coco_path: Path) -> dict:
    with proposed_biases_coco_path.open() as f:
        return json.load(f)
