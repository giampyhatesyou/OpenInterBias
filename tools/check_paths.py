"""Pre-flight: check that paths referenced in utils/config.py actually exist.

This script does NOT import the heavy upstream modules (no torch, no diffusers).
It reads utils/config.py as text and extracts any string literal that looks like
a filesystem path, then checks existence.

Use BEFORE launching expensive jobs on baldo — catches stale paths in seconds.

Exit codes:
  0  — all referenced paths exist
  1  — at least one referenced path is missing
  2  — could not parse utils/config.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "utils" / "config.py"

# Lines like:
#   'weights_path': '/some/path',
#   'path': '/.../coco',
# We also match relative paths starting with 'proposed_biases/' etc.
PATH_KEY_PATTERN = re.compile(
    r"""['"](?:path|weights_path|tokenizer_path|images_path|proposed_biases_path|checkpoint_path|ckpt_dir)['"]\s*:\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)

# Also catch top-level TODO sentinels like '/<insert>/<path>/<here>/...'
PLACEHOLDER_PATTERN = re.compile(r"/<insert>/<path>/<here>/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat placeholder paths (/<insert>/...) as failures (default: warn only).",
    )
    args = parser.parse_args()

    if not CONFIG_PATH.is_file():
        print(f"ERROR: {CONFIG_PATH} not found", file=sys.stderr)
        return 2

    text = CONFIG_PATH.read_text()
    matches = PATH_KEY_PATTERN.findall(text)

    if not matches:
        print(f"ERROR: no path keys found in {CONFIG_PATH} — regex likely broken", file=sys.stderr)
        return 2

    missing: list[str] = []
    placeholders: list[str] = []
    ok: list[str] = []

    seen: set[str] = set()
    for raw_path in matches:
        if raw_path in seen:
            continue
        seen.add(raw_path)
        if PLACEHOLDER_PATTERN.search(raw_path):
            placeholders.append(raw_path)
            continue
        # Resolve relative to repo root
        p = Path(raw_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            ok.append(str(p))
        else:
            missing.append(str(p))

    print(f"--- check_paths.py ---")
    print(f"config: {CONFIG_PATH}")
    print(f"distinct path strings: {len(seen)}")
    print(f"OK ({len(ok)}):")
    for p in ok:
        print(f"  ✓ {p}")
    if placeholders:
        print(f"PLACEHOLDERS ({len(placeholders)}) — must be edited before launching:")
        for p in placeholders:
            print(f"  ⚠ {p}")
    if missing:
        print(f"MISSING ({len(missing)}) — paths set in config but not present:")
        for p in missing:
            print(f"  ✗ {p}")

    if missing:
        return 1
    if placeholders and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
