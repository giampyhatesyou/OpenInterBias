# Setup checklist — running OpenInterBias on `baldo`

A pragmatic, step-by-step checklist for getting the OpenInterBias fork to run on the `baldo` GPU cluster. Follow it top-to-bottom; do not skip steps unless you've already done them in a previous session.

> **Why this order matters.** OpenInterBias is a fork of OpenBias built for intersectional bias detection (see [`ARCHITECTURE_NOTE.md`](../ARCHITECTURE_NOTE.md) §4). The intersectional stage is **post-hoc and CPU-only**: it reads cached outputs from the upstream baseline (stages 1-4). So the order is fixed:
>
> 1. Reproduce the OpenBias baseline on baldo (this document, steps 1-11).
> 2. Once `vqa_answers.json` exists, run the intersectional stage 5 (CPU-only, see step 12 below). The increment code is **not yet implemented** — `cluster/05_intersectional_analysis.sbatch` is a placeholder waiting for `intersectional/run_analysis.py`.

---

## 0. Prerequisites

You should already have:

- A `baldo` account with allocation on a GPU partition.
- The ability to `git clone` the fork (`https://github.com/giampyhatesyou/OpenInterBias.git`).
- A way to download model weights (Llama-2-7B-chat from Meta / HF, LLaVA-1.5-13B from HF, Stable Diffusion XL from HF). Some of these require accepting a license.

---

## 1. Clone & branch

```bash
ssh user@baldo
cd /your/scratch/area
git clone https://github.com/giampyhatesyou/OpenInterBias.git
cd OpenInterBias
git checkout chore/repo-prep   # or main, depending on what's been merged
```

---

## 2. Python environment

The upstream `requirements.txt` pins specific versions that depend on CUDA / torch versions. Install PyTorch FIRST, matching the cluster's CUDA, then the rest.

```bash
module load python/3.10           # adapt to baldo's module system
module load cuda/11.8             # match the README's tested combination

python -m venv openbias
source openbias/bin/activate
pip install --upgrade pip

# 1. PyTorch matching CUDA — check pytorch.org/get-started/locally for the right index URL
pip install torch==2.2.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Everything else
pip install -r requirements.txt
```

If any pin in `requirements.txt` fails (`fiftyone`, `modelscope`, `promptcap` are the usual suspects), comment it out, install the rest, and revisit. Capture the working `pip freeze` into `runs/<run-name>/env.txt`.

---

## 3. Model weights

| Model | Where it goes | Notes |
|---|---|---|
| **Llama-2-7B-chat** | `weights/llama-2-7b-chat/` (or any path) | Set in `utils/config.py:BIAS_PROPOSAL_SETTING['llama2']` |
| **LLaVA-1.5-13B** | `utils/llava/weights/llava-v1.5-13b/` | Path is hard-coded in `VQA_SETTING['vqa_models']['llava-1.5-13b']` |
| **Stable Diffusion XL** | (downloaded from HF on first run) | Set `HF_HOME` to a path with room — SDXL base + refiner ≈ 25 GB |
| **StyleGAN3 FFHQ** | `utils/stylegan3/weights/stylegan3-r-ffhq-1024x1024.pkl` | Only needed for the unconditional generation experiment |

Memory hint: LLaMA-2-7B fp16 ≈ 14 GB, LLaVA-13B fp16 ≈ 26 GB, SDXL fp16 ≈ 12 GB. A single 40 GB A100 fits any one of these; a 24 GB GPU may need offload (`enable_model_cpu_offload`, currently commented out in [utils/generative_models.py](../utils/generative_models.py)).

---

## 4. Datasets

| Dataset | Layout expected by `utils/datasets.py` | Notes |
|---|---|---|
| **COCO 2017** | `<root>/annotations/{captions_train2017.json, instances_train2017.json}` + `<root>/images/train2017/` | Used in `Coco` class; filtering keeps **single-person** images only |
| **Flickr30k** | `<root>/captions.txt` + `<root>/Images/` | Used in `Flickr_30k` class |
| **WinoBias** | `<root>/professions.txt` | Optional; used for the closed-set evaluation |
| **FFHQ** | `<root>/images/` + (after captioning) `<root>/captions.json` | Only for unconditional StyleGAN3 experiment |

---

## 5. Update `utils/config.py`

Open `utils/config.py` and replace every `/<insert>/<path>/<here>/` placeholder with the real paths on baldo. The 9 placeholders are:

```
BIAS_PROPOSAL_SETTING['llama2']['weights_path']
BIAS_PROPOSAL_SETTING['llama2']['tokenizer_path']
BIAS_PROPOSAL_SETTING['coco']['path']
BIAS_PROPOSAL_SETTING['flickr_30k']['path']
BIAS_PROPOSAL_SETTING['ffhq']['path']
BIAS_PROPOSAL_SETTING['winobias']['path']
VQA_SETTING['coco']['original']['images_path']
VQA_SETTING['flickr_30k']['original']['images_path']
VQA_SETTING['ffhq']['original']['images_path']
```

Verify with the pre-flight tool:

```bash
python tools/check_paths.py
# Exit 0 = all paths real, no placeholders remaining
# Exit 1 = at least one path missing
```

Re-run until exit 0.

---

## 6. Cluster scripts

```bash
cp cluster/_common.env cluster/_common.local.env
$EDITOR cluster/_common.local.env       # fill TODOs

# Edit each .sbatch's #SBATCH partition/account
$EDITOR cluster/01_bias_proposal.sbatch
$EDITOR cluster/02_generate_images.sbatch
$EDITOR cluster/03_run_vqa.sbatch
$EDITOR cluster/04_make_plots.sbatch
```

---

## 7. Create a run folder

```bash
RUN_NAME="$(date +%Y-%m-%d)_baseline_coco-sdxl_v1"
mkdir -p "runs/${RUN_NAME}/logs" "runs/${RUN_NAME}/outputs"
cp configs/baseline.template.yaml "runs/${RUN_NAME}/config.snapshot.yaml"
$EDITOR "runs/${RUN_NAME}/config.snapshot.yaml"          # fill TODOs

# Wire run name into cluster env
sed -i "s|TODO_run-name|${RUN_NAME}|" cluster/_common.local.env
```

---

## 8. Pilot first

**Do not** launch the full COCO baseline before a successful pilot. Use the pilot template:

```bash
RUN_NAME="$(date +%Y-%m-%d)_pilot_coco-sdxl_v1"
mkdir -p "runs/${RUN_NAME}/logs" "runs/${RUN_NAME}/outputs"
cp configs/pilot.template.yaml "runs/${RUN_NAME}/config.snapshot.yaml"
$EDITOR "runs/${RUN_NAME}/config.snapshot.yaml"
sed -i "s|TODO_run-name|${RUN_NAME}|" cluster/_common.local.env

sbatch cluster/pilot_all_stages.sbatch
squeue -u $USER
```

Expected wallclock: 1–4 h depending on how restricted the pilot is.

---

## 9. Full baseline (after pilot succeeds)

```bash
RUN_NAME="$(date +%Y-%m-%d)_baseline_coco-sdxl_v1"
# ... config snapshot already prepared in step 7 ...
sbatch cluster/01_bias_proposal.sbatch
# When jobid<N> completes:
sbatch --dependency=afterok:<N> cluster/02_generate_images.sbatch
# ... chain stages 3 and 4 similarly with --dependency=afterok
```

Or run them as a single submission via `cluster/pilot_all_stages.sbatch` (rename to `baseline_all_stages.sbatch` and remove the pilot caveats).

---

## 10. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Address already in use` on DDP init | `MASTER_PORT=12398` hard-coded in [utils/DDP_manager.py](../utils/DDP_manager.py) | If two jobs share a node, change the port for one. Better fix: parametrise from env in a follow-up PR. |
| `ModuleNotFoundError: llama` on bias_proposals | The `llama/` directory is local to the repo; not installed as a package | Make sure you `cd` into the repo before launching |
| `pycocotools` fails to install | Missing C compiler on cluster | `module load gcc` first |
| `fiftyone` install pulls MongoDB | MongoDB unreachable on baldo compute nodes | Pre-install on a login node OR use the workaround in `requirements.txt` (pymongo<4.9 already pinned) |
| Out of memory on LLaVA | LLaVA-13B in fp16 ≈ 26 GB | Switch to `llava-1.5-7b` for the pilot only |
| Empty `proposed_biases/coco/3/coco_train.json` | LLM produced no valid JSON | Lower temperature, check the system prompt didn't get accidentally edited |

---

## 11. After the run

```bash
# Capture the final state
bash tools/snapshot_run.sh "runs/${RUN_NAME}" "final"
# Verify outputs exist
ls -lR "runs/${RUN_NAME}/outputs/"
# Write RESULT.md
$EDITOR "runs/${RUN_NAME}/RESULT.md"
```

Then commit (NOT the run folder contents, which are gitignored — just any documentation updates) and push.

---

## 12. Next phase — intersectional post-hoc analysis (stage 5)

Once the baseline above produced a valid `vqa_answers.json`, the intersectional stage becomes available. It does **not** require GPUs and does **not** re-run image generation or VQA — it reads the cached outputs and computes joint distributions / metrics per pair of attributes proposed on the same caption.

Workflow once `intersectional/run_analysis.py` is implemented:

```bash
# 1. Create a separate run folder for the intersectional analysis
INTER_RUN="$(date +%Y-%m-%d)_intersectional_$(basename ${RUN_NAME})_v1"
mkdir -p "runs/${INTER_RUN}/logs" "runs/${INTER_RUN}/outputs"

# 2. Use the intersectional template (not the baseline one)
cp configs/intersectional.template.yaml "runs/${INTER_RUN}/config.snapshot.yaml"
$EDITOR "runs/${INTER_RUN}/config.snapshot.yaml"
#   ↑ set consumes_baseline_run = "runs/${RUN_NAME}"
#     set the same dataset/generator/vqa_model as the baseline

# 3. Submit (CPU partition is enough; sbatch enforces no --gres=gpu)
sed -i "s|TODO_run-name|${INTER_RUN}|" cluster/_common.local.env
sbatch cluster/05_intersectional_analysis.sbatch
```

Until the python entry point exists, `cluster/05_intersectional_analysis.sbatch` exits with code 2 and a clear message ("PLACEHOLDER waiting for the increment"). That's intentional — the cluster setup is ready in advance, so the only thing left to do later is land the code.

Reference docs for that phase:
- [`intersectional/SCHEMA.md`](../intersectional/SCHEMA.md) — the candidate / output schema
- [`intersectional/ARCHITECTURE_NOTE.md`](../intersectional/ARCHITECTURE_NOTE.md) — how the post-hoc stage plugs in
- [`docs/SCHEMA_DECISION.md`](SCHEMA_DECISION.md) — open scope questions to settle with the group before coding
