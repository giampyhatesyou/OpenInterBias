# Intersectional Schema — Scope Decisions

Status: **scope to refine with the group**. The decision to extend OpenBias toward intersectional bias detection is **already committed** (this is the OpenInterBias fork). What is still open is the *shape* of the schema, not whether to have one.

The canonical schema in the repo is [`intersectional/SCHEMA.md`](../intersectional/SCHEMA.md). This document records the open scope questions and trade-offs that should be settled before the increment code is written.

---

## Current schema in the repo

[`intersectional/SCHEMA.md`](../intersectional/SCHEMA.md) defines:

- A **pairwise** intersectional candidate (two attributes `attribute_a`, `attribute_b`)
- Two metrics:
  - **Normalized Joint Entropy** → joint intensity in $[0, 1]$
  - **Normalized Mutual Information** → dependence between marginals in $[0, 1]$
- Both **context-free** and **context-aware** variants (mirroring upstream `make_plots.py`)

Reference shape:

```json
{
  "caption_id": 12345,
  "caption": "A photo of a doctor",
  "bias_pair_name": "person gender x person race",
  "attribute_a": {"name": "person gender", "classes": [...], "question": "..."},
  "attribute_b": {"name": "person race",   "classes": [...], "question": "..."}
}
```

There is also an **earlier exploratory draft** outside the repo at `../../understanding/INTERSECTIONAL_SCHEMA.md` (not tracked, kept locally). It uses a more verbose shape (`component_attributes[]`, denormalised `joint_classes[]`). It is historical context, not a competing in-repo schema.

---

## Open scope questions

### Q1. Pairs only, or also triples / N-way?

| Option | Implication |
|---|---|
| **Pairs only** (current) | Simpler schema, smaller joint space, statistically tractable with the support we'll have. Aligned with AGENT.md ("pairwise intersectional biases only" — default scope). |
| **N-way from day one** | Forces `component_attributes[]` shape (list, not `a`/`b`). Most pairs will already be data-starved → triples are very likely empty. |

Recommendation: **pairs only for v1**. Revisit only if the data supports it.

### Q2. Same `refer_to` requirement?

OpenBias clusters single biases by `refer_to` (e.g. `person`, `kitchen`). Should intersectional pairs be restricted to same `refer_to`?

| Option | Implication |
|---|---|
| **Same `refer_to`** | Semantically clean (`person gender × person race`). Excludes `person gender × kitchen style`, which is more "co-occurrence" than "intersectional". |
| **Cross `refer_to`** | Larger candidate space, includes some scientifically interesting cases (e.g. `person attire × room style`) but mostly noise. |

Recommendation: **same `refer_to` for v1**, leave room in the schema to relax later.

### Q3. Filter on `present_in_prompt`?

OpenBias single-bias filters out biases whose answer is already in the caption (e.g. "a woman cooking" → gender is in the prompt, so don't measure gender bias on that caption). For pairs, should we require **both** attributes to be `present_in_prompt=false`?

Recommendation: **yes, both must be `false`**. Otherwise the "intersectional bias" leaks information from the prompt.

### Q4. Joint prediction record — separate file?

Per-image joint predictions can either:

- Be stored alongside the upstream `vqa_answers.json` (extending the file), **or**
- Live in a new file `joint_answers.json` produced by the post-hoc analysis stage.

Recommendation: **separate file**, written by the new stage 5. Keeps the upstream artefact untouched (one of AGENT.md's invariants).

### Q5. Class-cluster handling

Upstream `post_processing` ends with **one class_cluster per bias** (asserted in `make_plots.py:89-91`). The pair's joint class set is `classes(A) × classes(B)`. Open: do we drop `unknown` / `other` / `non-binary` predictions before pairing, like `make_plots.py:99-103` does?

Recommendation: **yes, replicate the upstream exclusion** for coherent comparison with the single-bias baseline.

---

## Provisional canonical extension

Building on [`intersectional/SCHEMA.md`](../intersectional/SCHEMA.md), the minimal additions needed to operationalise the answers above:

```json
{
  "caption_id": 12345,
  "caption": "A photo of a doctor",
  "bias_pair_name": "person|gender|person|race",
  "refer_to": "person",
  "attribute_a": {"name": "person gender", "classes": [...], "question": "..."},
  "attribute_b": {"name": "person race",   "classes": [...], "question": "..."},
  "metadata": {
    "support_intersect": 87,
    "filter": {"present_in_prompt_a": false, "present_in_prompt_b": false, "same_refer_to": true},
    "git_commit": "abcdef0"
  }
}
```

Notes:
- `bias_pair_name` uses `|` to avoid clashes with class names containing "x"; names pre-sorted alphabetically so `A|B == B|A`.
- `metadata.support_intersect` = number of captions where **both** attributes survived upstream filtering and got VQA answers — the actual population size of the joint distribution.
- `metadata.filter` makes the filtering decisions in Q3/Q5 explicit per record.

---

## Action items

- [ ] Group reviews this doc, picks answers for Q1–Q5.
- [ ] Update [`intersectional/SCHEMA.md`](../intersectional/SCHEMA.md) to include the `metadata` block and the `bias_pair_name` separator decision.
- [ ] Write `intersectional/SCHEMA.schema.json` (machine-validatable) once the shape is locked.
- [ ] Only then start `intersectional/pairing.py`, `intersectional/scoring.py`.

This document is a record of trade-offs. Update it when answers are agreed.
