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
    """Path to the OpenBias authors' COCO bias proposals (73,399 captions, ~153 MB).

    Not tracked in git: download it with the link in docs/OPENBIAS_README.md. The schema
    tests that need it skip when it is absent.
    """
    p = repo_root / "proposed_biases" / "coco" / "3" / "coco_train.json"
    if not p.is_file():
        pytest.skip(f"COCO bias proposals not present (see docs/OPENBIAS_README.md): {p}")
    return p


@pytest.fixture(scope="session")
def proposed_biases_coco_data(proposed_biases_coco_path: Path) -> dict:
    with proposed_biases_coco_path.open() as f:
        return json.load(f)
