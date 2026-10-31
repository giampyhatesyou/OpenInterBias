# Licensing scope and third-party code

The MIT license in [LICENSE](LICENSE) covers only the work written for OpenInterBias by
Andrea Giampietro and Irene Verria:

- `intersectional/`, `tests/`, `tools/`, `.github/`;
- the project documentation (`README.md`, `RESULTS.md`, `REPRODUCE.md`, `CITATION.cff`, this
  file) and `.env.example`;
- the modifications to upstream files delimited by `# CHANGED (fork) - START/END` blocks, and
  the entries added to `utils/synonyms.json` by `intersectional/complete_synonyms.py`.

Everything else comes from the projects below and remains under its original terms.

| Path | Origin | Terms |
|---|---|---|
| Pipeline scripts in the repository root (`bias_proposals.py`, `generate_images.py`, `run_VQA.py`, `make_plots.py`, `captioning.py`, `StyleGan3_generation.py`, `steps.sh`), `utils/` except the two folders below, `proposed_biases/stylegan3_ffhq/add_caption_ids.py`, `requirements.txt` (upstream base), `OpenBias.png`, `docs/OPENBIAS_README.md` | [OpenBias](https://github.com/Picsart-AI-Research/OpenBias) — D'Incà et al., CVPR 2024 | The upstream repository had no license file when this repository was forked from it (its git history contains none). All rights remain with the OpenBias authors. |
| `utils/llava/` | [LLaVA](https://github.com/haotian-liu/LLaVA) — © 2023 Haotian Liu; a few files adapted from FastChat and Stanford Alpaca | Apache License 2.0 (see the file headers) |
| `utils/stylegan3/` | [StyleGAN3](https://github.com/NVlabs/stylegan3) — © 2021 NVIDIA Corporation | NVIDIA Source Code License for StyleGAN3 (non-commercial use only) |
| `llama/` | [Llama 2](https://github.com/meta-llama/llama) — © Meta Platforms, Inc. | Llama 2 Community License Agreement |
| `utils/synonyms.json` (upstream entries) | collected by OpenBias from the [ConceptNet](https://conceptnet.io) API | ConceptNet data is distributed under CC BY-SA 4.0 |

Model weights (Llama 2, Stable Diffusion XL, LLaVA-1.5, BERT/BART/RoBERTa) and datasets (COCO,
Flickr30k, FFHQ, WinoBias) are not included in this repository and are subject to their own
licenses.
