# tools/

Small read-only utilities for inspecting the repo state and capturing run metadata. **None of these tools mutate upstream data** — they only read or write into the per-run folder.

| Script | Purpose |
|---|---|
| `check_paths.py` | Validate that every path referenced by `utils/config.py` actually exists on disk. Useful as a pre-flight check on baldo before launching expensive jobs. |
| `inspect_proposed_biases.py` | Read a `proposed_biases/*.json` and print descriptive stats (number of captions, attributes by `refer_to`, top biases by count). Useful for sizing intersectional pairs. |
| `snapshot_run.sh` | Capture git commit, status, env (`pip freeze`), and Python version into `runs/<run-name>/`. Called automatically by the `.sbatch` templates. |

## Conventions

- All Python tools should run with `python tools/<script>.py --help` and exit 0 on success, ≠0 on failure.
- All shell tools use `set -euo pipefail`.
- No tool should write outside `runs/<run-name>/` or `stdout`.
