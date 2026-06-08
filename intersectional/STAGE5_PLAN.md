# Stage 5 — Intersectional Analysis: Implementation Plan (for team review)

Status: **proposal, pre-implementation**. This document specifies *what* to add for the
intersectional (joint-attribute) bias analysis and *how* it integrates with the existing
OpenBias pipeline, with a justification for every decision. It operationalises
[`SCHEMA.md`](SCHEMA.md) and resolves the open questions in
[`../docs/SCHEMA_DECISION.md`](../docs/SCHEMA_DECISION.md). Nothing here changes upstream code.

Companion docs: [`SCHEMA.md`](SCHEMA.md) (metrics + JSON schema), [`ARCHITECTURE_NOTE.md`](ARCHITECTURE_NOTE.md)
(post-hoc integration), [`../docs/SCHEMA_DECISION.md`](../docs/SCHEMA_DECISION.md) (scope Q1–Q5).

---

## 0. TL;DR of decisions (all justified below)

| # | Decision | Rationale (grounded in code) |
|---|---|---|
| D1 | Stage 5 is **post-hoc, read-only, CPU-only**; no upstream file is modified. | Repo invariant (ARCHITECTURE_NOTE §2). All inputs already exist as cached JSON. |
| D2 | Consume **`results/VQA/<dataset>/generated/<gen>/<vqa>/vqa_answers.json`** + `data_counts.json` + the `proposed_biases` JSON. | Exact artefacts written by `run_VQA.py:176-203`. |
| D3 | **Pairs only**, **same `refer_to`**, both attributes from the per-image answer set. | SCHEMA_DECISION Q1/Q2; pairing is read directly from `vqa_answers.json` keys. |
| D4 | `present_in_prompt=false` is **already guaranteed** for `--mode generated`. | `valid_bias_generated_images` (`utils/utils.py:44-51`) is the `valid_bias_fn` for generated mode, so every bias in `vqa_answers.json` already satisfies it. No extra filter needed. |
| D5 | Drop `unknown` / `other` / `non-binary` **before** forming a joint observation. | Exact mirror of `make_plots.py:99-103,140`. Required for comparability with the single-attribute baseline. |
| D6 | Reuse `make_plots.py`'s **normalized `entropy()`** verbatim for the joint distribution. | Guarantees the intersectional "Joint Intensity" is on the same scale as the baseline "Bias Intensity". |
| D7 | Report **context-free** and **context-aware** variants. | Mirrors the two baseline metrics (`make_plots.py`), so single vs joint are directly comparable. |
| D8 | Write a **new** `joint_answers.json` + `intersectional_results.json`; never edit upstream artefacts. | SCHEMA_DECISION Q4; preserves reproducibility of the baseline. |
| D9 | Pair identity key `bias_pair_name = "A|B"` with A,B **sorted**; `|` separator. | SCHEMA_DECISION (avoids clash with class names containing "x"; makes A×B == B×A). |

---

## 1. Objective & scope

**Goal.** Quantify *intersectional* bias — the joint distribution of two attributes (e.g.
`person gender × person race`) in the images a target T2I model produces — and compare it to the
single-attribute baseline already produced by `make_plots.py`.

**In scope (v1):** pairwise, same-`refer_to`, on **generated** images, on the COCO baseline we are
running now. Two metrics: **Joint Intensity** (skew of the joint distribution) and **Normalized
Mutual Information** (dependence between the two attributes).

**Out of scope (v1):** triples/N-way (Q1), cross-`refer_to` pairs (Q2), real-image `--mode original`
(different filtering path), any change to bias proposal / generation / VQA.

---

## 2. Invariants (what we must NOT break) — justified

1. **No edits to `bias_proposals.py`, `generate_images.py`, `run_VQA.py`, `make_plots.py`, or `utils/`.**
   *Why:* the fork's stated principle (ARCHITECTURE_NOTE §2) and the user's constraint ("no logic
   changes"). Stage 5 only *reads* their outputs.
2. **Deterministic & CPU-only.** *Why:* it consumes cached VQA predictions; there is no model
   inference. This makes it cheap to re-run when scoring changes (cluster/README §"Stage 5").
3. **Baseline artefacts stay byte-identical.** *Why:* the comparison is only valid if the
   single-attribute numbers are exactly those `make_plots.py` produced. Hence D8 (separate output).

---

## 3. Data contracts (exact, grounded in code)

### 3.1 Input A — `vqa_answers.json` (from `run_VQA.py:72-80`)
```jsonc
{
  // key = full image path; caption_id is the second-to-last path component
  "sd_generated_dataset/coco/train/sd-xl/26/0.jpg": {
     // bias_name : [ bias_cluster(refer_to), class_cluster, predicted_class ]
     "person gender": ["person", "cluster_0", "male"],
     "person race":   ["person", "cluster_0", "white"],
     "person age":    ["person", "cluster_0", "young"]
  }
}
```
- `caption_id = key.split('/')[-2]` (identical to `make_plots.py:132`).
- Each image carries **all** biases proposed for its caption → a per-image joint observation
  `(pred_A, pred_B)` requires no extra lookup.
- `predicted_class` ∈ classes(bias) ∪ {`unknown`}; may also be `other`/`non-binary` per class set.

### 3.2 Input B — `data_counts.json` (from `run_VQA.py:98-109`)
Aggregated per-class counts: `{bias_cluster: {bias_name: {class_cluster: {class: count, "unknown": n}}}}`.
Used only for **cross-checking** marginal counts against what we recompute from `vqa_answers.json`.

### 3.3 Input C — `proposed_biases/<dataset>/3/<dataset>_train.json`
The LLM output. Used to recover, per `caption_id`, the **question text** and the **declared class
set** of each attribute, for the output schema (SCHEMA.md §2A). Not needed to compute the metrics
(the predictions in 3.1 suffice), but kept for traceability/plots labels.

### 3.4 Output 1 — `joint_answers.json` (NEW, D8)
```jsonc
{
  "person gender|person race": {
     "26": [["male","white"]],          // caption_id -> list of per-image joint obs (post-exclusion)
     "412": [["female","black"]]
  }
}
```

### 3.5 Output 2 — `intersectional_results.json` (NEW; shape from SCHEMA.md §2B + SCHEMA_DECISION metadata)
```jsonc
{
  "person gender|person race": {
    "refer_to": "person",
    "context_free":  { "joint_intensity": 0.421, "mutual_information": 0.156,
                       "support_images": 1180, "support_captions": 1130,
                       "joint_distribution": {"male|white": 600, "male|black": 200,
                                              "female|white": 290, "female|black": 90} },
    "context_aware": { "mean_joint_intensity": 0.354, "mean_mutual_information": 0.112,
                       "support_captions": 1130 },
    "marginals_crosscheck": { "person gender": {"male":800,"female":380},
                              "person race": {"white":890,"black":290} }
  }
}
```
Plus plots `results/intersectional/<dataset>/generated/<gen>/<vqa>/intersectional_context_{free,aware}.png`.

---

## 4. Algorithm (step by step, with justification)

Let an **attribute** be a `(bias_cluster, bias_name)` pair as it appears in `vqa_answers.json`.

**Step 1 — Load & index.** Read `vqa_answers.json`. For each image key, derive `caption_id`. Keep the
map `caption_id -> {bias_name: (bias_cluster, class_cluster, pred)}` per image. *Justification:* mirrors
`make_plots.py:130-146` exactly, so grouping semantics match the baseline.

**Step 2 — Candidate pairs.** For each image's answer set, enumerate unordered attribute pairs
`(A,B)` with `cluster(A)==cluster(B)` (D3/Q2). Accumulate the set of caption_ids per pair.
*Justification:* same-`refer_to` keeps pairs semantically intersectional (`person gender × person race`),
not mere co-occurrence (`person × kitchen`) — SCHEMA_DECISION Q2. Person×person dominates the data
(56,665 captions with ≥2 person-attrs; top pairs age×gender 50k, gender×race 11.5k — see
`tools/inspect_proposed_biases.py`).

**Step 3 — Class exclusion (D5).** When reading `pred_A`, `pred_B`, **discard the image for that pair**
if either prediction ∈ {`unknown`,`other`,`non-binary`}. *Justification:* byte-for-byte the same
exclusion as `make_plots.py:99-103` (context-free) and `:140` (context-aware); without it the joint
distribution is not comparable to the baseline marginals.

**Step 4 — Joint observations.** For each surviving image: emit `(pred_A, pred_B)`. Group by
`caption_id`. Persist to `joint_answers.json` (D8).

**Step 5 — Context-free metrics** (aggregate over ALL images of the pair):
- Build the joint count matrix `N[a,b]`; classes = surviving classes of A × surviving classes of B.
- `P = N / N.sum()`.
- **Joint Intensity** `= 1 - entropy(P.flatten())` using `make_plots.entropy` (D6). *Justification:*
  identical normalized-entropy estimator (eps=1e-10, ÷log K) as the baseline → same [0,1] scale; a
  single-attribute baseline value is the special case where one axis has one class.
- **Mutual Information**: `I(A;B)=Σ P(a,b) log(P(a,b)/(P(a)P(b)))`; normalize by
  `min(H(A),H(B))` (SCHEMA.md §B), same log base as `entropy()`. Guard `min(H(A),H(B))==0` → MI:=0.
- Record `support_images`, `support_captions`, and `joint_distribution` (raw counts).

**Step 6 — Context-aware metrics** (per-caption then average, mirroring `make_plots.py:148-180`):
- For each caption with ≥1 surviving joint obs, build its joint distribution, compute
  `1 - entropy(per-caption joint)`, skip NaN/inf (as `make_plots.py:161-163`).
- `mean_joint_intensity = mean` over captions; same for a per-caption MI.
  *Justification:* "context-aware" in OpenBias = average over prompt contexts; we replicate the
  exact aggregation so single vs joint context-aware numbers are comparable.

**Step 7 — Cross-check.** Recompute marginal class counts from `vqa_answers.json` and assert they
match `data_counts.json` (after the same exclusions). *Justification:* catches any drift between our
reader and the upstream writer before we trust the joint numbers.

---

## 5. Module & file layout (additive only)

```
intersectional/
  pairing.py        # Steps 1-4: load vqa_answers.json -> candidate pairs + joint_answers.json
  scoring.py        # Steps 5-6: joint entropy intensity + normalized MI (imports make_plots.entropy)
  run_analysis.py   # CLI entry: orchestrates pairing+scoring, writes intersectional_results.json
  make_plots.py     # plotting, mirroring ../make_plots.py styling for visual comparability
  SCHEMA.md, ARCHITECTURE_NOTE.md, STAGE5_PLAN.md (this file)
cluster/05_intersectional_analysis.sbatch   # already a placeholder -> wire to run_analysis.py
tests/intersectional/                        # new unit tests (currently empty per repo)
```
- `run_analysis.py` CLI mirrors upstream: `--dataset coco --generator sd-xl --vqa_model llava-1.5-13b
  --mode generated`. *Justification:* same argument surface as `run_VQA.py`/`make_plots.py` → trivial to
  chain in `cluster/`.
- `scoring.py` **imports** `entropy` from the upstream `make_plots` (or a copy with a test asserting
  equality) rather than reimplementing it (D6).

---

## 6. Integration with the cluster pipeline

`cluster/05_intersectional_analysis.sbatch` today exits with code 2 (placeholder). Wire it to:
```bash
python intersectional/run_analysis.py --dataset coco --generator sd-xl \
       --vqa_model llava-1.5-13b --mode generated
python intersectional/make_plots.py   --dataset coco --generator sd-xl \
       --vqa_model llava-1.5-13b --mode generated
```
- **CPU partition, no `--gres=gpu`** (cluster/README §"Stage 5"). Runtime: seconds–minutes.
- Depends on Stage 3 having produced `vqa_answers.json` (`--dependency=afterok` on the VQA job, or run
  standalone once the file exists).

---

## 7. Validation & tests (before any real run)

1. **Unit — entropy parity:** assert `intersectional` uses the identical `entropy()` as `make_plots`
   on random vectors (D6 guarantee).
2. **Unit — known joint:** hand-built 2×2 joint (e.g. all "male|white") → Joint Intensity = 1.0,
   MI = 0.0 (independent uniform) and MI = 1.0 (perfectly dependent) on crafted inputs.
3. **Unit — exclusion:** images with `unknown`/`other`/`non-binary` are dropped (D5) — count matches.
4. **Unit — marginal cross-check:** recomputed marginals == `data_counts.json` (Step 7).
5. **Integration — tiny slice:** run on a 20-caption synthetic `vqa_answers.json`; verify
   `joint_answers.json` and `intersectional_results.json` shapes against SCHEMA.md.
6. **Invariant guard:** a test that `git diff --stat` touches nothing under the upstream entry points.

*Justification:* the repo already gates upstream invariants with `tests/test_upstream_schema.py`;
`tests/intersectional/` is the documented home for these (ARCHITECTURE_NOTE §5).

---

## 8. Edge cases (enumerated, with handling)

| Case | Handling | Why |
|---|---|---|
| A bias predicted only `unknown` over all images | pair skipped (0 surviving obs) | matches `make_plots.py:111-114` "all zeros, skip". |
| After exclusion a class set has <2 classes | intensity defined (entropy of 1 class via eps), but flag `degenerate=true` | avoid div-by-zero in MI; keep transparent. |
| `min(H(A),H(B)) == 0` | `mutual_information := 0.0` | MI undefined when an attribute is constant. |
| Low support (e.g. <30 joint images) | compute but tag `support_images`; recommend a `--min-support` flag (default 0, configurable) | statistical honesty without hiding data. |
| Same caption, multiple images (n-images>1) | all images contribute to context-free; per-caption pooled for context-aware | mirrors `make_plots.py:142-146`. |
| Class name contains "x" | use `|` separator in `bias_pair_name`/joint keys | SCHEMA_DECISION D9. |

---

## 9. Open questions for the team (please review)

- **Q1/Q2/Q3/Q5:** defaults adopted (pairs-only, same-refer_to, present_in_prompt already enforced,
  exclude unknown/other/non-binary). Confirm or override.
- **MI normalization:** SCHEMA.md uses `min(H(A),H(B))`. Alternatives: normalize by `√(H(A)H(B))`
  or `max`. Recommend `min` (upper-bounds NMI at 1). **Confirm.**
- **Context-aware MI:** is a per-caption MI meaningful when most captions have 1 image (n-images=1)?
  With n-images=1 the per-caption joint is a single point → context-aware MI ≈ 0 by construction.
  **Proposal:** report context-aware **Joint Intensity** only, and compute MI **context-free** only,
  unless we raise `n-images`. (This is the most important methodological call — flagging explicitly.)
- **`--min-support` threshold:** what minimum joint support to *report* vs *flag*?
- **Pair universe:** all same-refer_to pairs, or a curated demographic shortlist
  (gender×race, gender×age, age×race, gender×occupation) for the paper's headline figure?

---

## 10. Sequencing / effort

1. Land `pairing.py` + unit tests (Steps 1-4). *(~½ day)*
2. Land `scoring.py` + entropy-parity & known-joint tests (Steps 5-6). *(~½ day)*
3. `run_analysis.py` + `make_plots.py` + tiny-slice integration test. *(~½ day)*
4. Wire `cluster/05_*.sbatch`; dry-run on the COCO baseline's `vqa_answers.json`. *(~¼ day)*
5. Team review of `intersectional_results.json` on real data; settle §9; iterate on metric/plot.

Prerequisite for steps 4-5: the COCO baseline must have produced `vqa_answers.json` (Stage 2→3→4),
which is what the current generation run is building.
