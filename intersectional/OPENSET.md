# Open-set intersectional discovery — plan and runbook

The demographic study (REPORT.md) reduced ~4500 free-text bias names to gender/race/age: a
closed-set analysis that gives up OpenBias's defining strength. This stage recovers the open
set: scan EVERY same-`refer_to` attribute pair the proposal LLM ever produced, control the
false discovery rate, and report the unexpected couplings.

## What changes vs the demographic study

1. **Canonicalization stays mild.** Free-text bias names are still reduced by last token
   within a cluster (`"driver age"` -> `age`), plus a small reviewed synonym map
   (`attr_synonyms.json`: clothing->attire, ethnicity->race, ...). 2020 distinct attributes
   survive on the full COCO proposals — the pair space stays open, nothing is forced into a
   demographic triple.
2. **All pairs, FDR-controlled.** `run_analysis.py --fdr_q 0.05` scans every pair, refines the
   permutation p-values of the candidates (two-stage, `fdr.py`), and applies
   Benjamini-Hochberg over the family of pairs with support >= `min_support`. On the existing
   3044-image demographic run, 0/10 pairs survive — including the previously reported
   age x race (p=0.011): a multiple-testing lesson and the motivation for this scale-up.
3. **Open-set prompt-quality control.** The ATTR_WORDS lexicon only covered demographics;
   `prompt_quality.build_leak_index` flags a caption as leaky for ANY attribute when it
   lexically names one of the proposed class labels ("a man WALKING ..." leaks activity).
   Word-boundary lexical matching misses inflections ("jumps" vs class "jumping"), so reported
   leakage is a lower bound. Both variants are produced (raw and `--exclude_leaky`).
4. **Support is planned on what the pipeline will really ask, not hoped for.** Two pipeline
   behaviors silently destroy pair support: (a) `max_prompts_per_bias` caps generation AND VQA
   questioning per bias (`utils.get_first_caption`) — the baseline's max_prompts=2 left a max
   realized pair support of 51 over 6384 images; (b) `utils.post_processing` (valid-bias
   filter, class-cluster merging, caption-vs-class-synonyms filter) drops most (caption, bias)
   combos — planning on raw proposals overestimates support ~2x, and over the WHOLE 73k pool
   only 38 pairs can ever reach 30 asked captions. So: `dryrun_pairs.py --dump_asked` (CPU, on
   the cluster) records the exact post-filter asked attributes of every candidate caption, and
   `openset_select.py --asked` plans on those, weighting by per-attribute usable-answer rates
   measured on the existing runs (age 1.00, race 0.77, occupation 0.09 — LLaVA answers
   "unknown" to most occupation questions, so occupation pairs die regardless of captions: a
   finding in itself). Already-generated images are reused for free (symlinks; generation
   auto-skips). A confirm dry-run on the selected file verifies the final numbers, because the
   post-processing is file-dependent.

## The run (confirm dry-run numbers on the selected file)

- 6961 captions total = 4213 reused (train/ + train_demo/, re-VQA'd uncapped) + 2748 new;
  20076 VQA questions (~2.9/caption); this small file already achieves the full-pool ceiling.
- Asked-support: 38 pairs >= 30 captions, 16 >= 100, 9 >= 200, 6 >= 400; top non-demographic:
  activity x age 882, age x emotion 409, age x occupation 337, emotion x gender 155,
  ability x age 129, level x race 119, skiing ability x location 83, dog age x size 64.
- Realized support ~= asked x usable rates: age x gender ~2400, age x race ~1800,
  gender x race ~930, activity x age ~850 — enough that an age x race-sized effect
  (NMI_MM ~0.014) clears the BH cutoff in a family of ~25-38 tests.
- GPU: ~5.3 h SDXL on one L40S (/4 with `--gres=gpu:4`) for the new captions; ~2-8 h VQA.
- One upstream pitfall was fixed on the way: `filter_caption_generated` queries ConceptNet
  (HTTP, mostly 502) for every class missing from `utils/synonyms.json`; with
  filter_threshold=0 ~6k tail classes were uncovered. `complete_synonyms.py` pre-populates
  offline entries (class + plural/singular — what the pipeline would cache on API failure).
  Also `merge_class_clusters`'s PYTHONHASHSEED-dependent KeyError now has the real fix
  (`del` -> `pop(key, None)` in utils/utils.py) instead of relying on a lucky seed.

## Runbook (baldo)

```bash
# CPU, already done by the prep: synonyms, full-pool survey, selection, config patch,
# symlinks, confirm dry-run
python intersectional/complete_synonyms.py proposed_biases/coco/3/coco_train.json
PYTHONHASHSEED=0 python intersectional/dryrun_pairs.py \
    --proposals proposed_biases/coco/3/coco_train.json --dump_asked
python intersectional/openset_select.py --asked results/intersectional/openset_asked_attrs.json \
    --min_potential 30 --target_obs 1200
python intersectional/apply_demo_config.py coco_train_openset.json --tag openset --max-prompts 1000000
python intersectional/link_reused_images.py
PYTHONHASHSEED=0 python intersectional/dryrun_pairs.py     # confirm on the selected file

# GPU (sbatch staged at ~/ob_gen_openset.sbatch; edu-long, 4x L40S)
sbatch ~/ob_gen_openset.sbatch
# the job body is intersectional/run_openset.sh: generate -> VQA -> all-pairs scan with FDR,
# raw + clean variants -> results/intersectional/coco_openset{,clean}
```

Generation auto-resumes: a timeout + resubmit loses nothing. `utils/config.py` is restored by
the script on exit (`git checkout utils/config.py`).

## Reading the output

`results/intersectional/coco_openset/.../intersectional_results.md` opens with the FDR
discoveries table. Before claiming a discovery, classify it:
- **prompt-driven** — high leak rate, or it vanishes in the clean variant;
- **definitional** — the two attributes are semantically entangled (age x experience,
  sport x equipment): real coupling, but not a model bias;
- **model bias** — survives the clean variant and is not definitional. These are the
  contribution; show their joint tables.

## Inherited limits

Labels are VQA predictions (LLaVA), taxonomies are the LLM-proposed discrete classes,
non-binary is dropped upstream, clusters are free-text and NOT merged across synonyms
(person vs child vs surfer stay separate pair families), single generator (SDXL) and dataset
(COCO), correlational throughout.
