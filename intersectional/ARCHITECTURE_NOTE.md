# Intersectional Architecture Note

This document describes how the intersectional bias detection extension integrates with the existing upstream architecture of the **OpenBias** pipeline.

## 1. Upstream Pipeline Overview
As described in [ARCHITECTURE_NOTE.md](file:///Users/andrea/Desktop/Foundation%20Models/code/OpenInterBias/ARCHITECTURE_NOTE.md), the upstream OpenBias pipeline consists of:
1. **Bias Proposal** (LLM Llama-2 generates single-attribute bias prompts/classes).
2. **Image Generation** (Stable Diffusion generates images for captions).
3. **Bias Assessment** (VQA model evaluates single attributes on generated images).
4. **Scoring & Visualization** (Entropy of single attributes is plotted).

---

## 2. Intersectional Extension Integration

To adhere to the core principles of **respecting the original architecture** and **minimizing destructive edits**, the intersectional extension does not modify any upstream execution files. Instead, it runs as a **post-hoc analysis stage** (Stage 5) that reads from the cached outputs of Stage 3 (VQA answers).

```mermaid
graph TD
    subgraph Upstream Pipeline
        A[Dataset Captions] --> B[bias_proposals.py]
        B --> C[(proposed_biases/)]
        C --> D[generate_images.py]
        D --> E[(sd_generated_dataset/)]
        E & C --> F[run_VQA.py]
        F --> G[(results/ VQA answers)]
    end
    
    subgraph Intersectional Extension
        G --> H[intersectional/run_analysis.py]
        C --> H
        H --> I[(intersectional_results.json)]
        I --> J[intersectional/make_plots.py]
        J --> K[results/intersectional_context_free.png]
    end
```

### Flow Detail
1. **Pairing Proposals**: The script processes `proposed_biases` to group the multiple single-attribute biases proposed for each caption. It finds all unique pairwise combinations of these attributes (e.g., if a caption has Gender, Race, and Age proposed, it forms Gender x Race, Gender x Age, Race x Age).
2. **VQA Prediction Pairing**: For each generated image, the script loads the single VQA predictions from `vqa_answers.json` for both attributes in a pair (e.g., `gender_pred` and `race_pred`) and combines them into a joint observation `(gender_pred, race_pred)`.
3. **Scoring**: The script aggregates the joint observations and computes intersectional metrics (Normalized Joint Entropy and Mutual Information).
4. **Plotting**: Generates plots displaying the intersectional bias intensities.
