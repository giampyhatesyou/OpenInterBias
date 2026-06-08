# Stage 5 — GPU scale-up plan (2× L40S)

**Problem.** The realized baseline (`max_prompts_per_bias=2`, `n-images=1`, 6384 imgs) gives only
age×gender=48, age×race=10, gender×race=6 joint images → intersectional NMI is not measurable. The
demographic captions that would fix this **already exist** in the stage-1 proposals; they were never
generated. This plan generates a focused subset of them, **without touching the baseline**.

## ⚡ Quickstart — 3 commands (use the scripts; details below are the "under the hood")
```bash
# 1) LAPTOP, from repo root — copy demo inputs + scripts to baldo:
bash cluster/push_to_baldo.sh

# 2) baldo — patch config + dry-run gate + submit GPU job (all automatic):
ssh baldo
cd ~/OpenInterBias && bash cluster/run_demo.sh smoke     # then later:  bash cluster/run_demo.sh 6k

# 3) LAPTOP, after the job log shows "=== DONE ===" — download results + run Stage 5:
bash cluster/pull_from_baldo.sh
```
`run_demo.sh` patches `utils/config.py`, runs the dry-run gate (aborts if the count is wrong),
submits `cluster/ob_demo.sbatch` (generation + VQA on 2 GPU), and the job **auto-reverts the config**
when it finishes. Watch progress with `squeue -u $USER` and `tail -f ob_demo_<id>.out`.
The sections below explain what each script does, for reference / debugging.

## 0. What you'll get & how long (2 GPUs, ~3.5 s/img effective)

| tier | file | imgs | time @2×L40S | pair ceilings (age×gen / age×race / gen×race) |
|---|---|---:|---:|---|
| smoke | `coco_train_demo_smoke.json` | 50 | ~3 min | validates the pipeline only |
| **primary** | `coco_train_demo6k.json` | ~6000 | **~5.8 h** | 3295 / 5821 / 3474 |
| full | `coco_train_demo.json` | 24319 | ~24 h (edu-long) | 16319 / 10721 / 3869 |

> ⚠️ **Realistic yield < ceiling.** LLaVA answers `unknown`/`other` for **race** very often, and those
> are excluded (for comparability). Expect race pairs to realize *hundreds*, not thousands; age×gender
> realizes well (age/gender rarely `unknown`). Even so, hundreds >> the current 6, so the result
> becomes reportable. The 6k tier is the sweet spot; run `full` only if you have a 24 h window.

**Recommendation:** smoke → 6k. That solves the context-FREE result (the headline + the email's core).
Context-AWARE needs `n-images>1` — separate optional run in §6.

## 1. Safety model (baseline cannot be touched; everything reverts)
- New images → `sd_generated_dataset/coco/train_demo/...`  (not `.../train/...`)
- New VQA   → `results/VQA_demo/...`                        (not `results/VQA/...`)
  → the baseline images and `vqa_answers.json` are never read or written.
- The only code change is **5 literal values in `utils/config.py`**, reverted with one
  `git checkout utils/config.py`. No pipeline logic edited.
- Generation **auto-resumes** (skips full caption folders) → a killed/timed-out job loses nothing; just resubmit.
- Both new roots are gitignored (`results/`, `sd_generated_dataset/`).

## 2. Pre-staged inputs (built locally, gitignored). Copy the ones you need to baldo:
```bash
cd "<local>/OpenInterBias"
scp proposed_biases/coco/3/coco_train_demo_smoke.json proposed_biases/coco/3/coco_train_demo6k.json \
    baldo:'~/OpenInterBias/proposed_biases/coco/3/'
ssh baldo 'ls -la ~/OpenInterBias/proposed_biases/coco/3/coco_train_demo*.json'   # verify
```

## 3. The 5 edits to `utils/config.py` (on baldo)
| # | find | replace |
|---|---|---|
| 1 | `'max_prompts_per_bias': 2,` | `'max_prompts_per_bias': 1000,` |
| 2 | `'filter_threshold': 0.50,` | `'filter_threshold': 0,` |
| 3 | `'proposed_biases_path': f'proposed_biases/coco/{BIAS_PROPOSAL_SETTING["coco"]["n_prompts_per_image"]}/coco_train.json',` | `'proposed_biases_path': 'proposed_biases/coco/3/coco_train_demo_smoke.json',` |
| 4 | `'subfolder': 'coco/train',` | `'subfolder': 'coco/train_demo',` |
| 5 | `'save_path': 'results/VQA'` | `'save_path': 'results/VQA_demo'` |

> Edit #3 starts with the **smoke** file. After the smoke test passes, change #3 to
> `coco_train_demo6k.json` and rerun. **Verify before launch:**
> `git -C ~/OpenInterBias diff --stat utils/config.py` → only that file, 5 lines.

## 4. Dry-run gate (CPU, no GPU) — also detects the ConceptNet hang
```bash
cd ~/OpenInterBias && PYTHONHASHSEED=0 ~/openbias/bin/python - <<'EOF'
import sys; sys.argv=['x','--dataset','coco','--generator','sd-xl']
import utils.arg_parse as ap
opt=ap.argparse_generate_images()
from utils.datasets import Proposed_biases
ds=Proposed_biases(opt['dataset_setting']['proposed_biases_path'],
    opt['gen_setting']['max_prompts_per_bias'], opt['gen_setting']['filter_threshold'],
    opt['gen_setting']['hard_threshold'], opt['gen_setting']['merge_threshold'],
    opt['dataset_setting']['valid_bias_fn'], opt['dataset_setting']['filter_caption_fn'],
    opt['dataset_setting']['all_images'])
print('IMAGES TO GENERATE:', len(ds.get_data()))
EOF
```
- This runs `post_processing` (the same code path that can hang on ConceptNet). **If it returns in
  seconds, generation won't hang.** If it stalls → `Ctrl-C`, you have a class missing from
  `utils/synonyms.json` (shouldn't happen: demographic classes are a subset of the baseline).
- Expected: ~50 for smoke, a few-thousand for 6k. If `0` or huge → recheck the 5 edits.

## 5. The job (gen + VQA in one sbatch). Save as `~/ob_demo.sbatch`, `sbatch ~/ob_demo.sbatch`:
```bash
#!/usr/bin/env bash
#SBATCH --job-name=ob_demo
#SBATCH --account=foundation.models25
#SBATCH --qos=gpuedu
#SBATCH --partition=edu-long          # smoke: use edu-short (5min) or edu-medium (2h)
#SBATCH --nodes=1
#SBATCH --ntasks=1                     # REQUIRED or sbatch refuses
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=ob_demo_%j.out
#SBATCH --error=ob_demo_%j.err
set -euo pipefail
cd ~/OpenInterBias
unset HF_HOME TRANSFORMERS_CACHE       # use default ~/.cache/huggingface (weights are THERE; repo .hf_cache is empty)
export PYTHONHASHSEED=0                 # VQA merge_class_clusters determinism (else intermittent KeyError)
export TOKENIZERS_PARALLELISM=false
PY=~/openbias/bin/python
pip show peft >/dev/null 2>&1 && { echo "peft present -> removing (SDXL load crash)"; $PY -m pip uninstall -y peft; }
echo "=== GENERATION ($(date)) ==="; $PY generate_images.py --dataset coco --generator sd-xl
echo "=== VQA ($(date)) ==="; $PY run_VQA.py --vqa_model llava-1.5-13b --workers 4 --dataset coco --mode generated --generator sd-xl
echo "=== DONE ($(date)) ==="
```
- **Monitor:** `squeue -u $USER` ; `tail -f ~/OpenInterBias/ob_demo_*.out`.
- **GPU contention dodge** (memory): if a 2-GPU job won't schedule, run VQA inside an existing
  Jupyter/interactive allocation: `srun --jobid=<your_alloc> --overlap PYTHONHASHSEED=0 ~/openbias/bin/python run_VQA.py ...`.

## 6. (Optional) context-aware — needs `n-images>1`
The email promises a context-aware variant; it is **degenerate at n-images=1**. To produce it, do a
SECOND, smaller run on its OWN subfolder (so n-images>1 never overwrites the n-images=1 set):
- copy a ~500-caption file (e.g. `head` the demo selection), set in the coco block `'n-images': 1 → 10`
  and `'subfolder': 'coco/train_demo' → 'coco/train_demo_ctxaware'`, regenerate + VQA (~5 h @2 GPU).
- Otherwise: report **context-free as primary** and state context-aware needs n-images>1 (honest, defensible).

## 7. Back on the laptop — re-run Stage 5 on the richer data (CPU, ~1 min)
```bash
mkdir -p OpenInterBias/results/VQA_demo/coco/generated/sd-xl/llava-1.5-13b
scp baldo:'~/OpenInterBias/results/VQA_demo/coco/generated/sd-xl/llava-1.5-13b/{vqa_answers,data_counts}.json' \
    OpenInterBias/results/VQA_demo/coco/generated/sd-xl/llava-1.5-13b/
cd OpenInterBias
# analyze the demo tree by temporarily pointing the VQA path (the analysis derives paths from results/VQA/)
cp -r results/VQA_demo/coco results/VQA/coco_demo 2>/dev/null || true
PYTHONHASHSEED=0 python3 intersectional/run_analysis.py --dataset coco_demo --generator sd-xl \
    --vqa_model llava-1.5-13b --mode generated --cluster person --min_support 30
python3 intersectional/make_plots.py --dataset coco_demo --generator sd-xl \
    --vqa_model llava-1.5-13b --mode generated --cluster person --min_support 30
```
(Send me `vqa_answers.json` and I run this for you.) Now the demographic pairs should clear
`min_support=30` with tight CIs and a real permutation p-value.

## 8. Revert (always, when done)
```bash
git -C ~/OpenInterBias checkout utils/config.py        # undo the 5 edits
# baseline images (.../train/...) and results (results/VQA/...) were never touched.
# to discard the demo run entirely:
rm -rf ~/OpenInterBias/sd_generated_dataset/coco/train_demo ~/OpenInterBias/results/VQA_demo
```

## 9. Risk table (problem → mitigation)
| risk | likelihood | mitigation |
|---|---|---|
| 5 config edits wrong | med | `git diff` check + dry-run count (§4) before any GPU |
| ConceptNet hang in post_processing | low | demographic classes ⊂ baseline → in `synonyms.json`; dry-run (§4) detects it in seconds |
| `peft` crashes SDXL load | med | sbatch auto-removes peft (`pip uninstall -y peft`); reversible |
| HF_HOME → empty cache → re-download | med | sbatch `unset HF_HOME` → default `~/.cache/huggingface` |
| VQA `merge_class_clusters` KeyError | med | `PYTHONHASHSEED=0` (set in sbatch) |
| 2-GPU job won't schedule (contention) | med | edu-long + submit early; or Jupyter `srun --overlap`; auto-resume = no loss |
| job timeout / killed | low | generation auto-resumes; just resubmit the same sbatch |
| n-images>1 overwrites baseline | low | context-aware run uses its OWN subfolder (§6) |
| disk quota | low | 6k≈1 GB, full≈4 GB; `du -sh sd_generated_dataset/coco/train_demo` |
| forgot to revert config | med | §8 is the last step; future runs would otherwise read the demo config |
| race yield < expected (unknowns) | high | expected (§0 caveat); age×gender carries the headline; more captions = more race obs |
