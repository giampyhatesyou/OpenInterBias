# Reproducing the results

Exact commands in execution order, with the expected output of each step. GPU steps ran on
1–4× NVIDIA L40S; everything else is CPU. What the steps produce: see [RESULTS.md](RESULTS.md).

Always export `PYTHONHASHSEED=0`. CPU statistics (scoring, FDR, bootstrap) are exactly
reproducible (fixed seeds); SDXL/LLaVA are GPU-nondeterministic, so image-dependent numbers may
move by a few percent — the conclusions do not.

## 0. Environment

```bash
python3.9 -m venv ~/openbias && source ~/openbias/bin/activate
pip install -r requirements.txt          # pinned; tested with torch 2.5.1+cu121
pip uninstall -y peft                    # peft 0.17 needs accelerate>=0.34; pipeline never uses it
cp .env.example .env                     # set the OPENBIAS_* dataset/model paths, then source it
export PYTHONHASHSEED=0
```
Model weights in the default HF cache (`~/.cache/huggingface`): SDXL base+refiner,
llava-v1.5-13b (also under `utils/llava/weights/`), bert/bart/roberta.
Pre-flight: `python tools/check_paths.py` must pass.

Unit tests (CPU, no models):
```bash
python -m pytest tests/ -q     # expect: 39 passed, 5 skipped (skips need torch / large data files)
```

## 1. Upstream baseline (stages 1–4)

Stage 1 (LLM bias proposals) is not re-run: we use the authors' output
`proposed_biases/coco/3/coco_train.json` (153 MB, 73,399 captions — kept out of git).

```bash
python generate_images.py --dataset coco --generator sd-xl     # 6,384 images, ~7 s/img/GPU
python run_VQA.py --vqa_model llava-1.5-13b --workers 4 --dataset coco --mode generated --generator sd-xl
python make_plots.py --generator sd-xl --dataset coco --mode generated
```
Expected: 6,384 caption folders in `sd_generated_dataset/coco/train/sd-xl/`; 6,384 keys in
`results/VQA/coco/.../vqa_answers.json`; race is the strongest single-attribute bias
(intensity ≈ 0.5). Note: `max_prompts_per_bias = 2` in `utils/config.py` caps which captions
each bias is generated AND VQA-questioned on, so attribute *pairs* are starved here (max
realized pair support 51) — that is what steps 2–3 fix.

## 2. Closed-set demographic scale-up (stage 5, first version)

`intersectional/apply_demo_config.py` temporarily repoints `utils/config.py` (proposals file,
output dirs, caps); each run restores it with `git checkout utils/config.py`.

```bash
bash intersectional/run_demo.sh 6k     # ~3 h on 4×L40S: 3,044 images + VQA + scoring
bash intersectional/run_demo.sh ctx    # 295 captions × 10 images (context-aware variant)
```
Expected (`results/intersectional/coco_demo*`): supports gender×race 391, age×race 684,
age×gender 550; NMI_MM ≤ 0.014 everywhere; only age×race nominally significant (p ≈ 0.01),
and it does not survive the FDR of step 3.

## 3. Open-set all-pairs scan (main result)

```bash
# 3a. plan: survey the full pool, then coverage-greedy caption selection
python intersectional/apply_demo_config.py coco_train.json --tag openset --max-prompts 1000000
PYTHONHASHSEED=0 python intersectional/dryrun_pairs.py --dump_asked
git checkout utils/config.py
python intersectional/openset_select.py --asked --min_potential 30 --target_obs 1200
# expect: 6,961 captions selected = 4,213 reused + 2,748 new

# 3b. verify the plan BEFORE spending GPU (CPU, ~2 min)
python intersectional/apply_demo_config.py coco_train_openset.json --tag openset --max-prompts 1000000
PYTHONHASHSEED=0 python intersectional/dryrun_pairs.py
# expect: IMAGES = 6961, ~20,076 VQA questions, 38 pairs with >=30 asked captions
git checkout utils/config.py

# 3c. run (resubmit the resumable sbatch until "ALL DONE", or run the script directly)
sbatch intersectional/ob_gen_openset.sbatch        # 1×L40S/8h profile; or:
bash intersectional/run_openset.sh                 # ~5.3 h gen + ~2.5 h VQA on 1×L40S
```
Expected (`results/intersectional/coco_openset{,clean}`): 1,957 realized pairs, FDR family 34,
**15 discoveries raw / 12 after the leakage filter**, with the values in RESULTS.md.
Scoring alone can be re-run on cached VQA answers (CPU):
```bash
python intersectional/run_analysis.py --dataset coco_openset --generator sd-xl \
    --vqa_model llava-1.5-13b --mode generated --min_support 30 --fdr_q 0.05 \
    --proposed_biases proposed_biases/coco/3/coco_train_openset.json   # add --exclude_leaky for the clean variant
```

## 4. Context-aware validation of the discoveries

```bash
# build the validation set: top-30 non-leaky captions per discovered pair (325 captions)
python - <<'PY'
import json
ja=json.load(open("results/intersectional/coco_opensetclean/generated/sd-xl/llava-1.5-13b/joint_answers.json"))
pairs=["vehicle:color|size","dog:breed|size","person:age|style","person:age|attire",
"person:gender|occupation","dog:age|size","person:ability|age","person:age|occupation",
"person:activity|age","person:activity|gender","person:age|race","person:age|gender"]
sel=set()
for p in pairs: sel.update(sorted(ja[p],key=lambda c:-len(ja[p][c]))[:30])
full=json.load(open("proposed_biases/coco/3/coco_train_openset.json"))["bias_proposal"]
json.dump({"bias_proposal":[e for e in full if str(e["caption_id"]) in sel]},
          open("proposed_biases/coco/3/coco_train_ctxopen.json","w"))
PY

python intersectional/apply_demo_config.py coco_train_ctxopen.json --n-images 10 --tag ctxopen --max-prompts 1000000
python generate_images.py --dataset coco --generator sd-xl    # 3,250 images, ~3 h on 2×L40S
python run_VQA.py --vqa_model llava-1.5-13b --workers 2 --dataset coco --mode generated --generator sd-xl
git checkout utils/config.py

rm -rf results/VQA/coco_ctxopen && cp -r results/VQA_ctxopen/coco results/VQA/coco_ctxopen
python intersectional/run_analysis.py --dataset coco_ctxopen --generator sd-xl \
    --vqa_model llava-1.5-13b --mode generated --min_support 30 \
    --proposed_biases proposed_biases/coco/3/coco_train_ctxopen.json
```
Expected: only the dog pairs exceed the within-caption permutation floor (+0.14 / +0.12);
all person pairs ≈ 0 (see RESULTS.md).

## 5. Figures

```bash
python intersectional/make_openset_plots.py   # 6 PNGs in results/intersectional/figures/
```

## Output map

| artifact | where |
|---|---|
| closed-set results (raw / leak-filtered / ctx) | `results/intersectional/coco_demo`, `coco_democlean`, `coco_ctxaware` |
| open-set scan (raw / leak-filtered) | `results/intersectional/coco_openset`, `coco_opensetclean` |
| ctx validation of discoveries | `results/intersectional/coco_ctxopen` |
| selection plan + exact dry-run | `results/intersectional/openset_selection_report.json`, `openset_dryrun_pairs.json` |
| figures | `results/intersectional/figures/` |
