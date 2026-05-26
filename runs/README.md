# runs/

This directory holds **experiment run artefacts**. Each run is a sibling subfolder; the rest of the tree is gitignored, so cloned copies of the repo do not carry past runs.

## Naming convention (from AGENT.md)

```
runs/<YYYY-MM-DD>_<short-purpose>_<scope>_v<n>/
```

Examples:

```
runs/2026-05-26_baseline_coco-sdxl_v1/
runs/2026-06-02_min-e2e_gender-age_v1/
runs/2026-06-10_joint-entropy_ablation_v2/
```

Keep the slug short, kebab-case, no spaces. Use `vN` only when the same purpose is rerun with a meaningful change (new seed sweep ≠ new version; new metric = new version).

## Required contents per run

Every run folder MUST contain:

| File | Why |
|---|---|
| `config.snapshot.yaml` (or `.json`) | The exact config used. Copy from `configs/` and freeze. |
| `git.txt` | Output of `git rev-parse HEAD` and `git status --porcelain` at launch time. Captured by `tools/snapshot_run.sh`. |
| `env.txt` | Output of `pip freeze` (or conda list). |
| `purpose.md` | One paragraph: what this run is for, what we expect to learn. Written *before* launching. |
| `command.sh` | The exact shell command that was launched (the literal `python ...` line). |
| `logs/` | stdout/stderr per stage. |
| `outputs/` | Cached intermediate JSONs, plots, anything the stage produces locally to this run. |

Optional but recommended:

- `notes.md` — running log of observations during the run (e.g. "stage 3 OOM at batch 12, retried with batch 8").
- `RESULT.md` — written *after* the run completes: what worked, what failed, numbers.

## Why this structure

AGENT.md § "Experimental Discipline" requires that we can always answer:

- which code produced this result?  → `git.txt`
- with which config?                  → `config.snapshot.yaml`
- on which data?                       → `purpose.md` + `config.snapshot.yaml`
- at what scale?                       → `command.sh` + `config.snapshot.yaml`

If any of these is missing the run is **not** reproducible and should be re-launched.

## What does NOT go here

- Model weights (use `weights/`, gitignored)
- Generated images (use `sd_generated_dataset/`, gitignored, shared across runs)
- VQA answer dumps shared across runs (use `results/`, gitignored)

Runs reference shared assets by path; they don't copy them.
