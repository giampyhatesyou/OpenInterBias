# Intersectional bias in SDXL on COCO

We extend OpenBias from single-attribute bias to joint bias of attribute pairs. After scaling the
generation to about 3000 person-demographic images, the picture is consistent: SDXL has strong
single-attribute biases (most clearly on race) but the demographic attributes are largely separable,
i.e. the intersectional coupling between them is small.

## Method

The analysis is post-hoc and reads the cached VQA answers. For each pair of person attributes
measured on the same image we build the joint distribution and compute Joint Intensity
(1 - normalized joint entropy) and Normalized Mutual Information (NMI). NMI is reported with a
Miller-Madow small-sample correction, a bootstrap confidence interval, and a permutation-test
p-value. Free-text attribute names are canonicalized to age/gender/race; class labels are normalized
(class_map.json). Exclusions of unknown/other/non-binary mirror the upstream make_plots, so the
numbers are comparable to the single-attribute baseline.

## Scale-up

The original baseline run was small (max_prompts_per_bias = 2, n-images = 1), so the demographic
pairs had 6 to 48 joint observations - too few to measure dependence. We rendered the
person-demographic captions that the bias-proposal stage already contained but had not generated,
into separate output directories. This gave 3044 VQA-labelled images and raised the support:

| pair          | before | after |
|---------------|-------:|------:|
| gender x race | 6      | 391   |
| age x race    | 10     | 684   |
| age x gender  | 48     | 550   |

## Results

Single-attribute marginals (where the bias actually concentrates):

| attribute | bias intensity | distribution                                            |
|-----------|---------------:|---------------------------------------------------------|
| race      | 0.53           | caucasian 572, african-american 166, asian 22, hispanic 7 |
| age       | 0.23           | middle-aged 529, young 438, old 46                      |
| gender    | 0.05           | male 615, female 356                                    |

Intersectional pairs (Miller-Madow NMI, permutation p, prompt leakage, and the value after removing
leaky prompts):

| pair          | NMI_MM | p     | leakage | clean NMI_MM / p |
|---------------|-------:|------:|--------:|------------------|
| age x race    | 0.014  | 0.008 | 0%      | 0.014 / 0.013    |
| gender x race | 0.009  | 0.17  | 12%     | 0.006 / 0.32     |
| age x gender  | 0.002  | 0.17  | 10%     | 0.002 / 0.18     |

Every NMI is small (at most 0.014 on a 0-1 scale). The model's bias lives in the marginals (a strong
caucasian skew, a milder male skew), not in the attribute combinations. Only age x race is
statistically significant, and the effect size is negligible. Concretely, in that pair
african-american faces skew slightly younger (young 80 vs middle-aged 55) while caucasian faces skew
middle-aged (292 vs young 207); this weak tendency is the single measurable intersectional signal.

## Prompt quality

The generation prompt is the raw COCO caption. OpenBias is meant to measure only attributes the
prompt does not state, but the present_in_prompt flag is unreliable: among the realized pairs, 12%
of gender x race captions and 10% of age x gender captions lexically name an attribute
("a man in a kitchen"). When the prompt fixes an attribute, its value is not a free choice of the
model, so the joint correlation for that pair can be partly an artifact of the prompt. We measure the
leakage per pair (prompt_quality.py) and recompute after dropping the leaky prompts:

- gender x race goes from NMI_MM 0.009 (p 0.17) to 0.006 (p 0.32), i.e. further toward independence.
  Part of the small apparent coupling came from the prompts rather than the model.
- age x race has 0% leakage, so the one significant result is not a prompt artifact.

## Context-aware variant

The results above are context-free (aggregated over all captions). The context-aware variant
averages a per-caption metric and needs more than one image per caption; with the n-images = 1
baseline it is degenerate. We ran 295 demographic captions at 10 images each (`run_demo.sh ctx`) and
computed, per pair, the mean over captions of the per-caption Joint Intensity and NMI:

| pair          | captions | CA Joint Intensity | CA NMI |
|---------------|---------:|-------------------:|-------:|
| age x race    | 235      | 0.26               | 0.13   |
| age x gender  | 158      | 0.28               | 0.07   |
| gender x race | 134      | 0.24               | 0.09   |

Within a fixed prompt the joint distribution is moderately concentrated (Joint Intensity ~0.25), but
the per-caption NMI (~0.07-0.13) sits at the small-sample bias level of MI computed on ~10 images per
caption (on independent synthetic data the same estimator gives ~0.13). So the context-aware view is
consistent with the context-free finding: the attributes are largely independent.

## Limitations

- Labels are VQA predictions, not ground truth; the class taxonomies are discrete and non-binary is
  dropped; the analysis is pairwise, same-`refer_to`, single generator and dataset, and correlational.

## Reproduce

See README.md. In short: `bash intersectional/run_demo.sh 6k` generates the demographic images, runs
the VQA, and scores the pairs raw and prompt-filtered into results/intersectional/coco_demo and
results/intersectional/coco_democlean.
