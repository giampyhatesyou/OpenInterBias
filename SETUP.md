# OpenInterBias Setup Guide

This guide explains how to configure OpenInterBias for your machine or cluster.

## Quick Start (Multi-Machine Setup)

The project now uses **environment variables** to manage paths, so you can run the same code on different machines without modifying the source code.

### Step 1: Copy the Environment Template

```bash
cp .env.example .env
```

### Step 2: Edit `.env` with Your Paths

Edit `.env` and set the paths for your machine:

```bash
# Example for local development
export OPENBIAS_LLAMA_PATH="/home/user/models/llama-2-7b-chat"
export OPENBIAS_LLAMA_TOKENIZER_PATH="/home/user/models/llama-2-7b-chat/tokenizer.model"
export OPENBIAS_COCO_PATH="/mnt/data/coco"
export OPENBIAS_FLICKR30K_PATH="/mnt/data/flickr_30k"

# Example for cluster (BALDO, etc.)
# export OPENBIAS_LLAMA_PATH="/scratch/shared/models/llama-2-7b-chat"
# export OPENBIAS_COCO_PATH="/data/datasets/coco"
```

### Step 3: Source Before Running

**Option A:** Source `.env` once, then run scripts

```bash
source .env
python bias_proposals.py --workers 6 --dataset 'coco'
python generate_images.py --dataset coco --generator sd-xl
python run_VQA.py --vqa_model llava-1.5-13b --workers 4 --dataset 'coco' --mode 'generated'
```

**Option B:** Set variables on-the-fly (no sourcing needed)

```bash
OPENBIAS_LLAMA_PATH="/path/to/llama" python bias_proposals.py --workers 6 --dataset 'coco'
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENBIAS_LLAMA_PATH` | Llama-2 model directory | `/models/llama-2-7b-chat` |
| `OPENBIAS_LLAMA_TOKENIZER_PATH` | Llama-2 tokenizer path | `/models/llama-2-7b-chat/tokenizer.model` |
| `OPENBIAS_COCO_PATH` | COCO dataset directory | `/data/coco` |
| `OPENBIAS_FLICKR30K_PATH` | Flickr30k dataset directory | `/data/flickr_30k` |
| `OPENBIAS_FFHQ_PATH` | FFHQ dataset directory | `/data/FFHQ` |
| `OPENBIAS_WINOBIAS_PATH` | WinoBias dataset directory | `/data/winobias` |

## For Cluster/HPC Setup (SLURM)

Update your SBATCH submission scripts to include environment setup:

```bash
#!/bin/bash
#SBATCH --job-name=openbias
#SBATCH --gpus=4

# Load any required modules
module load pytorch cuda

# Set paths for your cluster
export OPENBIAS_LLAMA_PATH="/scratch/shared/models/llama-2-7b-chat"
export OPENBIAS_COCO_PATH="/data/shared/coco"

# Activate environment
source ../openbias/bin/activate

# Run pipeline
python bias_proposals.py --workers 6 --dataset 'coco'
```

## Downloading Models and Datasets

### Llama-2 Model

```bash
# Requires Meta license approval
huggingface-cli login
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat /path/to/save/llama-2-7b-chat
```

### COCO Dataset

Download from [COCO website](https://cocodataset.org/) and structure as:
```
/path/to/coco/
  train2017/
  val2017/
  annotations/
    captions_train2017.json
    captions_val2017.json
```

### Flickr30k

Download from [Flickr30k website](https://shannon.cs.illinois.edu/DenotationGraph/) and place in:
```
/path/to/flickr_30k/
  flickr30k-images/
  results_20130124.token
```

## Troubleshooting

**Error: "no checkpoint files found in /<insert>/<path>/<here>/llama-2-7b-chat"**

→ Your `OPENBIAS_LLAMA_PATH` is not set. Run `source .env` first, then verify:
```bash
echo $OPENBIAS_LLAMA_PATH
ls $OPENBIAS_LLAMA_PATH
```

**Error: Dataset path not found**

→ Check that the dataset path is correct:
```bash
echo $OPENBIAS_COCO_PATH
ls $OPENBIAS_COCO_PATH
```

## Git Workflow (Multi-Machine)

1. **Machine A (Development):**
   ```bash
   # Edit code
   git add .
   git commit -m "Update config to use env vars"
   git push
   ```

2. **Machine B (GPU Cluster):**
   ```bash
   git pull
   cp .env.example .env
   # Edit .env with cluster paths
   source .env
   python bias_proposals.py --workers 6 --dataset 'coco'
   ```

The `.env` file is **not committed** (in `.gitignore`), so each machine can have different paths.

## Notes

- `.env` is in `.gitignore`, so it won't be committed to Git
- `.env.example` is committed, serves as a template
- All paths can be absolute or relative to the project root
- Environment variables take precedence over default placeholders in `config.py`
