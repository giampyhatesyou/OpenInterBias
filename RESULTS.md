# Results

Fork of [OpenBias (CVPR 2024)](https://github.com/Picsart-AI-Research/OpenBias) extending
open-set bias detection from single attributes to **intersectional attribute pairs**.
How to reproduce everything: [REPRODUCE.md](REPRODUCE.md). Figures:
`results/intersectional/figures/`.

## Open-set all-pairs scan (SDXL on COCO captions)

6,961 captions (4,213 reused from previous runs + 2,748 newly generated), uncapped VQA
questioning, scan of all 1,957 same-`refer_to` attribute pairs realized in the VQA answers.
Significance: permutation test per pair + Benjamini–Hochberg FDR (q ≤ 0.05) over the 34 pairs
with support ≥ 30. Two variants: raw prompts, and prompt-leakage filtered (a caption is dropped
for a pair when it lexically states one of the proposed class labels of either attribute).

**15 discoveries raw → 12 survive the leakage filter** (attire×race, age×emotion, age×level do
not). Leakage-filtered discoveries, by effect size (Miller–Madow NMI, all q ≤ 0.05):

| pair | NMI_MM | support |
|---|---:|---:|
| vehicle: color × size | 0.50 | 31 |
| dog: breed × size | 0.48 | 34 |
| person: age × style | 0.30 | 64 |
| person: age × attire | 0.24 | 197 |
| person: gender × occupation | 0.23 | 136 |
| dog: age × size | 0.19 | 64 |
| person: ability × age | 0.11 | 125 |
| person: age × occupation | 0.08 | 283 |
| person: activity × age | 0.05 | 770 |
| person: activity × gender | 0.05 | 376 |
| person: age × race | 0.025 | 1834 |
| person: age × gender | 0.008 | 2146 |

The hand-picked demographic pairs (age×race, age×gender; gender×race is not even significant)
are the **weakest** couplings found: the interesting intersectional structure lives outside the
closed demographic set. Example joints (row-normalized, see `joint_heatmaps.png`):
P(working professional | female) = 0.71 vs P(businessman | male) = 0.41;
P(large | golden retriever) = 0.95, P(small | beagle) = 0.80; P(suit | old) = 0.75.

## Context-aware validation of the discoveries

For each discovered pair: 10 images × ~30 non-leaky captions, mean per-caption NMI compared to a
per-pair within-caption permutation floor. Only **dog:breed×size (+0.14)** and
**dog:age×size (+0.12)** exceed the floor — within-prompt entanglement of the generator. All
person pairs show ≈ 0 excess: their context-free coupling is **contextual** (per-context marginal
bias aggregated across prompts), not joint sampling at fixed prompt.

## What was changed / added

- Changes to upstream files are delimited in-source by `# CHANGED (fork) - START/END` blocks:
  `utils/utils.py` (KeyError bugfix in `merge_class_clusters`), `utils/config.py` (paths via
  environment variables), `make_plots.py` (`--vqa_model` choices from config),
  `requirements.txt` (pinned dependency set). `utils/synonyms.json` is also changed
  (107 → 59,315 keys, generated offline by `intersectional/complete_synonyms.py` to remove a
  runtime ConceptNet-API stall) — JSON cannot carry markers, recorded here.
- Added, no upstream logic touched: the post-hoc stage-5 module `intersectional/` (pairing,
  scoring with Miller–Madow/bootstrap/permutation, FDR, prompt-leakage control, caption
  selector, dry-runs, drivers, figures) and `tests/` (39 passing + env-dependent skips).
