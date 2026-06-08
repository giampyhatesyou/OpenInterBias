# Stage 5 — Execution log (live)

> Transparency anchor. Every action taken by the autonomous session is logged here
> with **how to reverse it**. Nothing is committed; nothing touches upstream code or
> the cluster's state. Read this top-to-bottom to know exactly what was done.

Session date: 2026-06-08.

## 0. Guarantees held throughout
- **No git commit/push.** All changes left in the working tree for you to review & commit yourself.
- **Additive only.** New files under `intersectional/`, `tests/intersectional/`, and gitignored
  `results/`. No edit to `bias_proposals.py`, `generate_images.py`, `run_VQA.py`, `make_plots.py`, `utils/`.
- **Cluster: reads only.** Pulled 3 JSON via `scp` from baldo. No writes, no job submission by me.
- **All reversible.** Undo instructions next to each step.

## 1. Data pulled from baldo (read-only) — DONE
`scp baldo:~/OpenInterBias/...` →
- `results/VQA/coco/generated/sd-xl/llava-1.5-13b/vqa_answers.json` (1.29 MB, 6384 images)
- `results/VQA/coco/generated/sd-xl/llava-1.5-13b/data_counts.json` (1.48 MB)
- `proposed_biases/coco/3/coco_train.baldo.json` (153 MB — the real stage-1 proposals; only
  for labels, **not** used by the metrics; kept out of the way, gitignored).

Undo: `rm` those files. The git-tracked stub `proposed_biases/coco/3/coco_train.json` (1.4 KB) is untouched.

## 2. CRITICAL FINDING — the realized baseline is too sparse for intersectional analysis
The VQA file keys biases by **free-text `bias_name`** (4554 distinct: "driver age",
"soccer player race", "pilot experience", …), not canonical "person gender/race/age".
- Pairing on **raw** `bias_name` → top pair support = **2 captions** → useless.
- Canonicalizing the attribute by its **last token** (age/gender/race/…) within cluster `person`
  recovers some support, but it is still tiny on the 6384 realized images:

| person pair | joint images |
|---|---|
| age × gender | 48 |
| age × race | 10 |
| gender × race | 6 |

The plan's "age×gender 50k / gender×race 11.5k" came from `tools/inspect_proposed_biases.py`
reading the **proposal** universe (all COCO), **not** the generated+VQA'd images
(`max_prompts_per_bias=2`, `n-images=1`). At n=6–48 the Mutual Information estimate is
dominated by small-sample upward bias → not reportable as-is.

**Implication:** the CPU deliverable below is correct infrastructure + an honest *preliminary*
read; a publishable intersectional result needs the **GPU scale-up** in §5.

## 3. Method decisions (for team review — all flagged, all overridable)
- **Attribute canonicalization** `attr_mode="last_token"` (default). Deviates from STAGE5_PLAN D3
  (raw bias_name) because D3 is empty on real data. `attr_mode="raw"` recovers the literal plan.
- **Exclusions** mirror `make_plots.py` exactly: drop `unknown`/`other`/`non-binary` before any joint obs.
- **Joint Intensity** = `1 - normalized_entropy(joint)`, using a **verbatim copy** of
  `make_plots.entropy` (parity unit test). Single-cell joints are flagged `degenerate`, not forced to 1.0.
- **NMI** = MI / `min(H_A, H_B)` (nats); guarded to 0 when an attribute is constant.
- **Honesty add-ons (beyond the plan):** per-pair **bootstrap 95% CI** on NMI and a
  `--min-support` flag (default 30) that *flags* (does not hide) low-support pairs.
- **Class taxonomies** kept raw (messy: boy/girl in gender, caucasian vs white). Optional
  normalization left as a future `--class-map` for the team to curate.

## 4. Files created (CPU, reversible) — see §6 for status
```
intersectional/pairing.py        # load vqa_answers -> per-pair joint observations (+ joint_answers.json)
intersectional/scoring.py        # normalized entropy, joint intensity, MI/NMI, bootstrap CI
intersectional/run_analysis.py   # CLI: orchestrates -> intersectional_results.json + .md summary
intersectional/make_plots.py     # figures (needs matplotlib)
tests/intersectional/test_intersectional.py
```
Outputs land in (gitignored) `results/intersectional/coco/generated/sd-xl/llava-1.5-13b/`.
Undo everything: `rm -r intersectional/{pairing,scoring,run_analysis,make_plots}.py tests/intersectional results/intersectional`.

Local env change (only when plotting): `pip install matplotlib` into base conda — reverse with
`pip uninstall matplotlib`. numpy already present.

## 5. GPU scale-up (prepared for you to paste; I do NOT submit) — see separate section once ready
Goal: raise realized support for the demographic pairs by generating many more prompts per
person-demographic bias (and optionally `n-images>1`). Reversible by construction: separate
output dir, gitignored, `generate_images.py` auto-resumes. Commands in §7 below.

## 6. Status checklist
- [x] Pull data from baldo
- [x] Sparsity finding documented
- [x] Modules written (pairing, scoring, run_analysis, make_plots)
- [x] Tests pass — `9 passed, 1 skipped` (parity self-skips w/o matplotlib; verified separately:
      entropy parity exact on 200 random dists)
- [x] Run on real data -> `results/intersectional/coco/generated/sd-xl/llava-1.5-13b/`
      (`intersectional_results.json`, `.md`, `joint_answers.json`)
- [x] Figures: `intersectional_nmi_person.png`, `intersectional_joint_intensity_person.png`,
      `intersectional_heatmap_person.png`
- [x] GPU scale-up recipe prepared -> see `intersectional/SCALEUP.md`
- [x] **Robustness (point 2):** Miller-Madow bias-corrected NMI + permutation-test p-value +
      single-attribute marginal intensities added to `scoring.py` (MI vectorized for speed);
      surfaced in results JSON/MD/summary; 14 unit tests pass.
- [x] Concise report written -> `intersectional/REPORT.md`
- [x] `cluster/05_intersectional_analysis.sbatch` wired to the real CLI (was a placeholder; the only
      git-**tracked** file changed — reversible via `git checkout`). All other new files are untracked.
- [x] Extra figure: `intersectional_mi_vs_marginals_person.png` (NMI vs marginal bias; the key
      diagnostic — no person pair is significant, all points grey/p≥0.05).
- [x] **Class normalization** (`class_map.json` + `pairing.normalize_class`, `--class_map`/`--no_class_map`):
      boy→male, white→caucasian, young-child→young, … (no-op on the pilot's rare classes, matters post-scale-up).
- [x] **Full-data marginals** (`baseline_marginals.py`): aggregated single-attribute intensities from
      `data_counts.json` → race **0.49** (caucasian-dominated, 18 obs), gender 0.12, age 0.15. In results JSON/MD.
- [x] **Sensitivity** (`sensitivity.py` → `sensitivity.md`): MI norm (min/geom/max) × class_map grid;
      raw-bias_name pairing max support = 2 (why we canonicalize). Verdict invariant; magnitude shifts only.
- 17 unit tests pass, 1 skipped (parity self-skip).

### Headline result (CPU, current baseline)
59 person pairs over 6384 images. **Only `person: age×gender` (n=48) clears min_support=30:**
NMI = 0.069, **Miller-Madow = 0.034, permutation p = 0.23** (95% CI [0.006, 0.293]), Joint
Intensity = 0.147 → age and gender are **statistically independent** in SDXL on COCO (strong
marginal male-skew 34:14, but uncoupled across age bands). Every other pair is low-support
(p≈0.7–1.0, CI spans [0,1]) → not reportable. The honest conclusion: **the realized baseline is
underpowered for intersectional analysis** (see §2; fix in `SCALEUP.md`).

## 7. GPU scale-up — see `intersectional/SCALEUP.md`
Feasibility quantified from the stage-1 proposals (the captions already exist, just ungenerated):

| pair | now | reachable (6k tier, ~3h/4×L40S) |
|---|---:|---:|
| age × gender | 48 | 3 295 |
| age × race | 10 | 5 821 |
| gender × race | 6 | 3 474 |

Recipe is reversible by construction: separate output roots (`*/train_demo/`, `results/VQA_demo/`)
+ 5 git-revertable lines in `utils/config.py` + a CPU dry-run go/no-go gate. Pre-built filtered
proposals: `proposed_biases/coco/3/coco_train_demo6k.json` (and `_demo.json` for the full set).
**I did NOT submit anything to the cluster** — the commands are staged for you to paste.
