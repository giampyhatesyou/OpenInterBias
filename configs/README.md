# configs/

Run-level YAML configuration templates. **These do NOT replace `utils/config.py`** — that file remains the canonical source of paths, prompts, and model registry.

The files here are meant to be **copied into a run folder** and frozen as `config.snapshot.yaml`. They document the run's *intent* (which dataset, which generator, what budget, why), in a format that's easy to diff across runs and easy to read by humans.

Workflow:

1. Copy a template:
   ```
   cp configs/baseline.template.yaml runs/2026-05-26_baseline_coco-sdxl_v1/config.snapshot.yaml
   ```
2. Fill in the `TODO` fields.
3. Launch the pipeline using the shell command in `command.sh` (with the same arg values as in the YAML).
4. The YAML stays in `runs/<run-name>/` forever — it's the immutable record of what was run.

We are intentionally **not** wiring an auto-loader from these YAMLs into the upstream entry points. The upstream code reads `utils/config.py` directly; the YAML here is documentation-grade, not runtime.

When the intersectional code is added, **that** module may consume these YAMLs directly (since it's a new code path we control).
