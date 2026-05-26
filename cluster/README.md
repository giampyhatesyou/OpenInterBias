# cluster/

SLURM submission templates for running the OpenBias pipeline on a GPU cluster (`baldo` at the moment).

⚠️ These are **templates**. You must replace every `TODO_*` placeholder before submitting. Do not commit a version with your personal partition/account hard-coded.

## What's here

| File | Stage | Notes |
|---|---|---|
| `01_bias_proposal.sbatch` | Upstream § 1 — LLM bias proposal (Llama-2) | 1 GPU, ~10–30 min on COCO subset |
| `02_generate_images.sbatch` | Upstream § 2 — T2I generation (SD-XL by default) | Multi-GPU recommended, several hours on full COCO |
| `03_run_vqa.sbatch` | Upstream § 3 — VQA (LLaVA-1.5-13B) | 1 GPU minimum, scales with image count |
| `04_make_plots.sbatch` | Upstream § 4 — Quantification & plots | CPU-only, fast |
| `05_intersectional_analysis.sbatch` | Fork extension — Intersectional post-hoc analysis | **PLACEHOLDER**: python entry point not implemented yet. See `intersectional/SCHEMA.md` and `docs/SCHEMA_DECISION.md`. CPU-only, no GPU. |
| `pilot_all_stages.sbatch` | All 4 upstream stages chained | For the smoke-test pilot. Stage 5 is intentionally NOT chained here — run it as a separate submission once the baseline outputs exist. |
| `_common.env` | Shared env vars (sourced by each script) | Edit ONCE per environment |

### Stage 5 — why a separate submission?

The intersectional analysis reads cached `vqa_answers.json` from the baseline (stages 1-4) and runs on CPU. Keeping it as a standalone submission means:

- it can be re-run cheaply (seconds-minutes) when the scoring code changes, without re-running the GPU stages;
- it makes the dependency explicit (`stage 5 needs stage 3 output`), which mirrors the post-hoc design in `intersectional/ARCHITECTURE_NOTE.md`.

## Usage

```bash
# 1. Customise the env once
cp cluster/_common.env cluster/_common.local.env   # .local.env is gitignored
$EDITOR cluster/_common.local.env

# 2. Submit a stage
sbatch cluster/01_bias_proposal.sbatch

# 3. Monitor
squeue -u $USER
tail -f slurm-<jobid>.out
```

## Conventions

- Each script writes its log to `runs/<run-name>/logs/<stage>.out`.
- Each script verifies `runs/<run-name>/config.snapshot.yaml` exists before running (fail-fast).
- Each script calls `tools/snapshot_run.sh` at start to capture git state.
- The `--export=NONE` flag is set so cluster-side env vars don't leak into the job.

## If the cluster scheduler is NOT SLURM

These scripts assume SLURM. If `baldo` uses PBS / LSF / something else, port the directives at the top of each `.sbatch` file (the `python ...` lines stay the same). The `#SBATCH` lines are the only scheduler-specific parts.
