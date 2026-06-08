# Intersectional bias in SDXL (COCO baseline) — Stage 5 report

**TL;DR.** We built the post-hoc intersectional analysis (Stage 5) on top of the existing
OpenBias baseline and ran it on the 6384-image SDXL/COCO/LLaVA-13B run. The machinery works and is
validated, but the **realized baseline is too sparse** to measure intersectional bias reliably: of
59 person attribute-pairs, only **age×gender (n=48)** has enough support to report, and there the
two attributes are **statistically independent** (NMI 0.069, Miller-Madow-corrected 0.034,
permutation p=0.23). The captions needed to fix this already exist in the stage-1 proposals and just
need to be generated — see `SCALEUP.md` (≈3 h of GPU lifts the key pairs from 6–48 to ~3–6k samples).

---

## 1. What was built
A read-only, CPU-only, deterministic add-on (no upstream file changed):
- `pairing.py` — reads `vqa_answers.json`; canonicalizes the 4554 free-text `bias_name`s to a
  semantic attribute (last token: age/gender/race/…) within each `refer_to` cluster; forms
  same-cluster attribute pairs and per-caption joint observations. An optional **class-normalization
  map** (`class_map.json`: boy→male, white→caucasian, young-child→young, …) keeps messy LLaVA
  classes from inflating the entropy.
- `scoring.py` — **Joint Intensity** = 1 − normalized entropy of the joint (same estimator as the
  baseline `make_plots.entropy`, parity-tested); **NMI** (plug-in) with a bootstrap 95% CI; the
  **Miller-Madow** bias-corrected NMI; and a **permutation test** (H0 = independence) p-value. Also
  reports each attribute's single-attribute marginal intensity for comparison.
- `run_analysis.py` / `make_plots.py` — CLI + figures (ranked NMI with CIs, Joint Intensity, and a
  joint-distribution heatmap). 14 unit tests (`tests/intersectional/`).

Exclusions (`unknown`/`other`/`non-binary`) mirror `make_plots.py` exactly, so intersectional
numbers are directly comparable to the single-attribute baseline.

## 2. What was found
| person pair | support (imgs) | NMI | NMI (Miller-Madow) | perm. p | reportable? |
|---|---:|---:|---:|---:|:--:|
| age × gender | 48 | 0.069 | **0.034** | 0.23 | ✅ (weak/independent) |
| age × attire | 26 | 0.629 | 0.567 | 0.70 | ❌ low support |
| attire × gender | 17 | 0.518 | 0.428 | 0.97 | ❌ low support |
| age × race | 10 | 0.225 | 0.199 | 0.67 | ❌ low support |
| gender × race | 6 | 0.000 | 0.000 | 1.00 | ❌ low support |
| (… 54 more, all support ≤ 26 …) | | | | | |

The high-NMI pairs are **artifacts of small samples**: their permutation p-values (0.7–1.0) and
bootstrap CIs (spanning [0, 1]) say the apparent dependence is indistinguishable from chance. This
is the central, honest result — and the reason we added Miller-Madow + permutation testing.

**Full-data single-attribute marginals** (all person obs, for context): **race is the most skewed
single attribute — bias intensity 0.49** (caucasian-dominated, but only 18 non-`unknown` obs),
vs gender 0.12 and age 0.15. The cruel irony: race carries the strongest single-attribute bias yet
its *intersections* (gender×race, n=6) are exactly the ones the pilot cannot measure.

**Robustness** (`sensitivity.md`): the verdict is invariant to MI normalization (min/geom/max) and
to the class map; under raw `bias_name` pairing the max support over all pairs is **2**, confirming
canonicalization is necessary, not cosmetic.

## 3. The one solid result, interpreted
For **age × gender** (n=48), the joint distribution (`intersectional_heatmap_person.png`):

| | female | male |
|---|---:|---:|
| middle-aged | 5 | 20 |
| old | 1 | 5 |
| young | 8 | 9 |

SDXL skews strongly **male** (34 vs 14) — a marginal gender bias — but the male-lean is *similar
across age bands*, so age and gender are **not coupled**: NMI_MM=0.034, p=0.23. In OpenBias terms,
the marginal intensities (age 0.120, gender 0.129 on this subset) are not amplified at their
intersection (joint 0.147 ≈ what independence predicts). **Strong-ish single-attribute biases, weak
intersectional bias** — for this one measurable pair.

## 4. Why the data is sparse (mechanism)
Three compounding causes, none of them bugs:
1. **Pilot-scale generation.** `utils/config.py` ran with `max_prompts_per_bias=2`, `n-images=1`
   (the team's own `baseline.template.yaml` intends 100 / 10). So each bias produced ≤2 images.
2. **`present_in_prompt` filter.** Generated-mode only measures biases *not* stated in the prompt;
   for a caption like "A *man* …", gender is excluded → fewer gender observations.
3. **VQA `unknown`s.** LLaVA frequently answers `unknown`/`other` for race; those are dropped
   (for comparability), thinning race pairs the most (gender×race → 6).

The plan's "age×gender ≈ 50k support" figure came from the **proposal** universe (all 73k COCO
captions), not the **realized** 6384 images.

## 5. Limitations & what to add next
Full list in `STAGE5_LOG.md §2-3`. Headline limits: labels are LLaVA's *predictions* (not ground
truth); discrete/loaded taxonomies with non-binary dropped; pairs-only, same-cluster, correlational.
**The single highest-value next step is the GPU scale-up** in `SCALEUP.md`, which is reversible by
construction and turns this from "infrastructure + a null result on one pair" into a
properly-powered measurement of gender×race, age×race and age×gender.

## 6. Reproduce
```bash
PYTHONHASHSEED=0 python3 intersectional/run_analysis.py \
    --dataset coco --generator sd-xl --vqa_model llava-1.5-13b --mode generated --cluster person
python3 intersectional/make_plots.py \
    --dataset coco --generator sd-xl --vqa_model llava-1.5-13b --mode generated --cluster person --min_support 5
PYTHONHASHSEED=0 python3 -m pytest tests/intersectional -q
```
Outputs: `results/intersectional/coco/generated/sd-xl/llava-1.5-13b/` (`intersectional_results.json`,
`.md`, `joint_answers.json`, `intersectional_{nmi,joint_intensity,heatmap}_person.png`).
