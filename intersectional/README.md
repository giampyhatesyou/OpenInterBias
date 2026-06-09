# Intersectional bias analysis

An extension of OpenBias from single-attribute bias to **joint (intersectional) bias** of attribute
pairs, e.g. gender x race. It runs after the standard OpenBias pipeline (bias proposal, image
generation, VQA) and reads the cached VQA answers; it does not modify the upstream pipeline.

For the findings see `REPORT.md`. The metric definitions are in `SCHEMA.md`.

## What it computes

For each pair of attributes measured on the same image (same `refer_to`, e.g. two person
attributes), it builds the joint distribution and reports:

- **Joint Intensity** = 1 - normalized joint entropy (how concentrated the model is on particular
  attribute combinations), on the same scale as the OpenBias single-attribute Bias Intensity.
- **Normalized Mutual Information (NMI)** between the two attributes (how dependent they are). This
  is the part single-attribute analysis cannot see: NMI ~ 0 means the joint bias is just the product
  of the two marginals; NMI > 0 means the model couples the two attributes.

NMI is reported with a Miller-Madow small-sample correction, a bootstrap confidence interval, and a
permutation-test p-value. Free-text attribute names are canonicalized to age/gender/race/... and
messy VQA class labels are normalized via `class_map.json`.

## Running

All commands run locally from the repo root, on a machine with a GPU and the model weights cached.

Analysis only (on an existing VQA run):

    python intersectional/run_analysis.py --dataset coco --generator sd-xl \
        --vqa_model llava-1.5-13b --mode generated --cluster person --min_support 30
    python intersectional/make_plots.py --dataset coco --generator sd-xl \
        --vqa_model llava-1.5-13b --mode generated --cluster person --min_support 5

The single-attribute baseline is sparse for pairs, so a focused regeneration of the
person-demographic captions is provided. It generates into separate output directories and restores
the config afterwards (the original baseline is untouched):

    bash intersectional/run_demo.sh smoke    # 50-image sanity check
    bash intersectional/run_demo.sh 6k        # ~3000 demographic images, then VQA, then scoring

This writes results to `results/intersectional/coco_demo/` (raw) and
`results/intersectional/coco_democlean/` (with prompts that already state an attribute removed).

The context-free metric uses one image per caption. The context-aware metric (per-caption average)
needs several images per caption, so it has its own run:

    bash intersectional/run_demo.sh ctx       # 300 captions x 10 images -> results/intersectional/coco_ctxaware

Its report includes a "Context-aware" table (per-pair mean Joint Intensity and NMI over captions).

Tests:

    PYTHONHASHSEED=0 python -m pytest tests/intersectional -q

## Prompt quality

The generation prompt is the raw caption. OpenBias only measures attributes the prompt does not
state (the `present_in_prompt` flag), but that flag is unreliable: some captions name an attribute
("a man in a kitchen") while it is still flagged as not present. When the prompt fixes one attribute,
the measured value on it is not a free choice of the model, and pairing it with another attribute can
produce a spurious joint correlation. `prompt_quality.py` measures this per pair (a `leak` column in
the report) and `--exclude_leaky` drops those observations.

## Files

- `pairing.py` - load VQA answers, form same-`refer_to` pairs, per-caption joint observations
- `scoring.py` - Joint Intensity, NMI, Miller-Madow, bootstrap CI, permutation test
- `baseline_marginals.py` - single-attribute intensities (context)
- `prompt_quality.py` - caption-leakage detection
- `class_map.json` - class normalization
- `run_analysis.py` - CLI, writes `intersectional_results.json` + `.md`
- `make_plots.py` - figures
- `sensitivity.py` - robustness across normalization / canonicalization choices
- `run_demo.sh`, `apply_demo_config.py`, `dryrun_count.py` - the focused regeneration
- `REPORT.md`, `SCHEMA.md`, `ARCHITECTURE_NOTE.md` - results and design notes
