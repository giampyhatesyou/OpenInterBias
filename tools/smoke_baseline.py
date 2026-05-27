"""Smoke test for OpenBias baseline — run BEFORE launching any pipeline stage.

Checks (in order, fail-fast):
  1. Python version + key environment vars
  2. torch + CUDA visibility
  3. diffusers (Stable Diffusion stack)
  4. utils/llava model code imports
  5. llama wrapper imports
  6. utils/config.py loads cleanly and the dataset paths resolve
  7. proposed_biases/<dataset>/3/<dataset>_train.json is readable

Each step prints PASS or FAIL with a short hint. Exit code = number of failures.

Usage on baldo (from repo root):
    source .env                # load OPENBIAS_* env vars
    source ~/openbias/bin/activate
    python tools/smoke_baseline.py --dataset coco
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def step(name: str):
    """Decorator that captures pass/fail + traceback for a single check."""

    def deco(fn):
        def wrapper(*args, **kwargs):
            print(f"\n--- {name} ---")
            try:
                ok, hint = fn(*args, **kwargs)
                print(f"  {'PASS' if ok else 'FAIL'}  {hint}")
                return 0 if ok else 1
            except Exception as e:  # pragma: no cover
                print(f"  FAIL  {type(e).__name__}: {e}")
                if os.environ.get("SMOKE_VERBOSE"):
                    traceback.print_exc()
                return 1

        return wrapper

    return deco


@step("Step 1 — Python & env vars")
def check_python():
    print(f"  python: {sys.version.split()[0]}")
    needed = [
        "OPENBIAS_LLAMA_PATH",
        "OPENBIAS_LLAMA_TOKENIZER_PATH",
        "OPENBIAS_COCO_PATH",
        "OPENBIAS_FLICKR30K_PATH",
    ]
    missing = [v for v in needed if v not in os.environ]
    for v in needed:
        val = os.environ.get(v, "<unset>")
        print(f"  {v} = {val}")
    if missing:
        return False, f"missing env vars: {missing}. Did you `source .env`?"
    return True, "all OPENBIAS_* vars set"


@step("Step 2 — torch + CUDA")
def check_torch():
    import torch

    print(f"  torch: {torch.__version__}")
    avail = torch.cuda.is_available()
    print(f"  cuda available: {avail}")
    if avail:
        print(f"  cuda version: {torch.version.cuda}")
        print(f"  device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  device {i}: {torch.cuda.get_device_name(i)}")
    return avail, "CUDA visible" if avail else "torch.cuda.is_available()==False (allocate a GPU?)"


@step("Step 3 — diffusers stack")
def check_diffusers():
    from diffusers import DiffusionPipeline, StableDiffusionPipeline, EulerDiscreteScheduler  # noqa: F401

    import diffusers

    print(f"  diffusers: {diffusers.__version__}")
    return True, "diffusers importable"


@step("Step 4 — utils/llava code")
def check_llava():
    sys.path.insert(0, str(REPO_ROOT))
    from utils.llava.model.builder import load_pretrained_model  # noqa: F401
    from utils.llava.constants import IMAGE_TOKEN_INDEX  # noqa: F401
    from utils.llava.conversation import conv_templates  # noqa: F401

    return True, "utils.llava code imports OK"


@step("Step 5 — llama wrapper")
def check_llama():
    sys.path.insert(0, str(REPO_ROOT))
    from llama import Llama  # noqa: F401

    return True, "llama package importable (weights not loaded yet)"


@step("Step 6 — utils/config.py + dataset path resolution")
def check_config(dataset: str, skip_stage1: bool = False):
    sys.path.insert(0, str(REPO_ROOT))
    from utils import config as cfg

    ds_setting = cfg.BIAS_PROPOSAL_SETTING[dataset]
    ds_path = ds_setting["path"]
    print(f"  resolved {dataset} path: {ds_path}")
    if "<insert>" in ds_path:
        return False, f"path still a placeholder — set OPENBIAS_{dataset.upper().replace('_','')}_PATH in .env"
    if ds_path.startswith("/UNUSED"):
        if skip_stage1:
            return True, f"path intentionally /UNUSED — OK because Stage 1 will be skipped"
        return False, f"path is /UNUSED but you did not pass --skip-stage1"
    if not Path(ds_path).exists():
        return False, f"path does not exist on disk: {ds_path}"
    return True, f"dataset path exists"


@step("Step 7 — proposed_biases JSON")
def check_proposed(dataset: str):
    n = 3  # n_prompts_per_image
    p = REPO_ROOT / "proposed_biases" / dataset / str(n) / f"{dataset}_train.json"
    # fallback name for flickr_30k etc.
    if not p.is_file():
        alt = REPO_ROOT / "proposed_biases" / dataset / str(n) / f"{dataset}.json"
        if alt.is_file():
            p = alt
    if not p.is_file():
        return False, f"missing {p} — run bias_proposals.py or download from paper Drive"
    size_mb = p.stat().st_size / 1e6
    with p.open() as f:
        data = json.load(f)
    n_captions = len(data.get("bias_proposal", []))
    print(f"  file: {p}")
    print(f"  size: {size_mb:.1f} MB, captions: {n_captions}")
    if n_captions < 10:
        return False, f"only {n_captions} captions — toy file, not the real one"
    return True, f"{n_captions} captions ready"


@step("Step 8 — LLaVA weights (needed for Stage 3)")
def check_llava_weights():
    weights_dir = REPO_ROOT / "utils" / "llava" / "weights" / "llava-v1.5-13b"
    if not weights_dir.is_dir():
        return False, f"missing {weights_dir} — download from huggingface.co/liuhaotian/llava-v1.5-13b"
    files = list(weights_dir.iterdir())
    big = [f for f in files if f.is_file() and f.stat().st_size > 100 * 1024 * 1024]
    print(f"  dir: {weights_dir}")
    print(f"  total files: {len(files)}, big files (>100MB): {len(big)}")
    if not big:
        return False, "weights dir is empty or only metadata — model not downloaded"
    return True, f"{len(big)} large weight files present"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="coco", choices=["coco", "flickr_30k"])
    parser.add_argument("--skip-llava-weights", action="store_true", help="If you only plan to run Stage 1/2, skip the LLaVA weights check")
    parser.add_argument("--skip-stage1", action="store_true", help="Accept /UNUSED/* dataset paths because Stage 1 (bias proposal) will be skipped")
    args = parser.parse_args()

    failures = 0
    failures += check_python()
    failures += check_torch()
    failures += check_diffusers()
    failures += check_llava()
    failures += check_llama()
    failures += check_config(args.dataset, skip_stage1=args.skip_stage1)
    failures += check_proposed(args.dataset)
    if not args.skip_llava_weights:
        failures += check_llava_weights()

    print()
    print("=" * 60)
    if failures:
        print(f"  {failures} CHECKS FAILED — fix above before running the baseline.")
    else:
        print("  ALL CHECKS PASSED — baseline is launchable.")
    print("=" * 60)
    return failures


if __name__ == "__main__":
    sys.exit(main())
