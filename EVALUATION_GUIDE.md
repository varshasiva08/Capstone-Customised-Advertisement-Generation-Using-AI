# AdFidelity — Full Evaluation Guide

Complete step-by-step instructions to run every experiment needed for the IEEE Access paper.

---

## Overview: What gets produced

| Output | Paper section | Method |
|--------|---------------|--------|
| CLIP zero-shot validation on FairFace (11k images) | Experimental Setup: Scorer Validation | Zero-shot CLIP on FairFace val set |
| CLIP fidelity scores on generated images (108 images) | Results: Table 1 | Zero-shot CLIP on generated ads |
| LLaVA/DFC fidelity scores (108 images) | Results: Table 2 | Blind LLaVA scoring via Ollama |
| CDVR ablation (with vs without correction loop) | Results: Table 3 | SDXL + CLIP in-loop scoring |
| Scorer agreement (CLIP vs LLaVA) | Results: Table 4 | Cross-comparison of both scorers |
| Bias distribution analysis | Results: Table 5 | Chi-square uniformity test |
| Male subject evaluation (12 images) | Results: Gender generalizability | Zero-shot CLIP |

---

## Phase 0: Environment Setup

### 0.1 — Clone and switch to the evaluation branch

```bash
git clone https://github.com/varshasiva08/Capstone-Customised-Advertisement-Generation-Using-AI.git
cd Capstone-Customised-Advertisement-Generation-Using-AI
git checkout experiments/eval-results-v2
```

### 0.2 — Install Python dependencies

```bash
pip install -r requirements-eval.txt
```

If you have a GPU (CUDA), install PyTorch with CUDA support instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 0.3 — Install and start Ollama (needed for LLaVA DFC scoring)

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows — download from https://ollama.com/download
```

Pull the required models:

```bash
ollama pull phi3:mini
ollama pull llava
```

Start Ollama (leave this running in a separate terminal):

```bash
ollama serve
```

### 0.4 — Set up HuggingFace tokens

Create a `.env` file in the project root:

```
HF_TOKEN_1=hf_your_token_here
```

Get tokens from https://huggingface.co/settings/tokens (free account works).

---

## Phase 1: Download FairFace Dataset

FairFace is used for two purposes:
1. **LoRA fine-tuning** of the generator (train split) — handled separately
2. **CLIP scorer validation** (val split) — benchmarks the scorer's reliability

### 1.1 — Download

**Option A — From HuggingFace (recommended for Kaggle/Colab):**

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceM4/FairFace", "0.25")
```

**Option B — From GitHub (for local setup):**

Go to https://github.com/joojs/fairface and download:
- `fairface-img-margin025-trainval.zip` (~3.4 GB)
- `fairface_label_train.csv`
- `fairface_label_val.csv`

### 1.2 — Extract to the right location

```bash
mkdir -p eval/clip_trainer/fairface_data

unzip fairface-img-margin025-trainval.zip -d eval/clip_trainer/fairface_data/

cp fairface_label_train.csv eval/clip_trainer/fairface_data/
cp fairface_label_val.csv eval/clip_trainer/fairface_data/
```

### 1.3 — Verify the folder structure

```
eval/clip_trainer/fairface_data/
    train/          (~86k images — used for LoRA, NOT for CLIP)
    val/            (~11k images — used for CLIP validation)
    fairface_label_train.csv
    fairface_label_val.csv
```

---

## Phase 2: Validate Zero-Shot CLIP Classifier

> **IMPORTANT:** We use **zero-shot CLIP classification** — NOT fine-tuned classifier heads.
> An earlier approach fine-tuned MLP heads on FairFace embeddings but failed validation
> (models predicted the same class for every image). That approach was abandoned.
> **Do not** train or use the `.pt` classifier files (`age_classifier.pt`, etc.).

### 2.1 — How zero-shot CLIP classification works

For each demographic axis, text prompts are constructed:
- Race: "a photo of a South Asian person", "a photo of an East Asian person", etc.
- Gender: "a photo of a man", "a photo of a woman"
- Age: "a photo of a person in their twenties", "a photo of a person in their thirties", etc.

The generated image is compared against all prompts using CLIP's text-image similarity.
The prompt with the highest similarity score determines the predicted class.
No training is required — this uses off-the-shelf CLIP (openai/clip-vit-base-patch32).

### 2.2 — Run validation on FairFace val set

This benchmarks CLIP's reliability as a demographic classifier on ~11k real photos before using it to score generated images.

Run on Kaggle/Colab (GPU recommended for speed):

```python
# See AdFidelity_Evaluation.ipynb, Step 4 for the full validation script
# Key function:
def classify_batch(images, prompt_dict):
    labels = list(prompt_dict.keys())
    texts = list(prompt_dict.values())
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).cpu()
    preds = probs.argmax(dim=-1).tolist()
    return [labels[p] for p in preds]
```

### 2.3 — Expected results

| Attribute | Accuracy | Macro F1 |
|-----------|----------|----------|
| Gender    | 94.0%    | 0.94     |
| Race      | 63.0%    | 0.63     |
| Age       | 36.0%    | 0.34     |

Results saved to `clip_zeroshot_validation.json`.

These numbers establish the scorer's reliability. Gender and race classification are
strong enough to support the paper's primary claims. Age classification is a known
weakness of CLIP — acknowledged as a limitation in the paper.

---

## Phase 3: Generate Advertisement Images

### 3.1 — Generation setup

Images are generated using Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)
on Kaggle with a T4 GPU.

**Demographic grid:** 3 ethnicities × 3 body types × 4 ages × 3 seeds = 108 images

Each prompt follows the format:
```
professional studio photograph of a real {body_type} {ethnicity} woman in her {age},
wearing a white blazer suit, standing pose, looking at camera, warm beige background,
commercial fashion photography, full body shot, highly detailed face, photorealistic
```

### 3.2 — Run generation + CLIP scoring

See `AdFidelity_Evaluation.ipynb` (Kaggle notebook) for the full generation + scoring cell.
Images are scored with zero-shot CLIP immediately after generation.

Output: `eval/results/clip_fidelity_scores.csv` (108 rows)

### 3.3 — Resumability

The script skips images that already exist on disk. If interrupted, re-run the same cell.

---

## Phase 4: LLaVA/DFC Scoring

Scores the same 108 generated images using LLaVA via Ollama (runs locally).

> **Key difference from CLIP:** LLaVA scoring is "blind" — the model is asked
> "What ethnicity/age/gender do you see?" without being told the intended profile.
> This avoids confirmation bias.

### 4.1 — Run scoring

```bash
cd Capstone-Customised-Advertisement-Generation-Using-AI
python eval/run_llava_scoring.py
```

Requires: Ollama running with LLaVA model pulled.
Saves progress every 5 images (crash-recoverable).

Output: `eval/results/dfc_scores.csv` and `eval/results/dfc_summary.json`

---

## Phase 5: CDVR Ablation

Compares generation with and without the CDVR correction loop.

For each of 36 profiles (1 seed):
- **Without CDVR:** generate once, score, done
- **With CDVR:** generate → CLIP check → if demographics wrong, strengthen prompt → regenerate (up to 3 iterations)

### 5.1 — Run ablation

See `AdFidelity_Evaluation.ipynb` (Kaggle notebook) for the CDVR ablation cell.
Uses SDXL for generation and zero-shot CLIP as the in-loop scorer.

Output: `eval/results/cdvr_ablation.csv` and `eval/results/cdvr_ablation_summary.json`

---

## Phase 6: Scorer Agreement & Bias Analysis

Computes cross-validation between CLIP and LLaVA, plus demographic distribution analysis.

### 6.1 — Run analysis

```bash
python eval/compute_agreement_and_bias.py
```

Output:
- `eval/results/scorer_agreement.csv`
- `eval/results/full_analysis_summary.json`

---

## Phase 7: Male Subject Evaluation (Optional)

12 additional images (3 ethnicities × 4 ages, male) to validate gender generalizability.

See `AdFidelity_Evaluation.ipynb` for the male generation cell.

Output: `eval/results/male_fidelity_scores.csv`

---

## Summary of All Result Files

```
eval/results/
├── generated_images/               # 108 generated advertisement images (female)
├── cdvr_ablation_images/           # 72 before/after CDVR images
├── male_images/                    # 12 male subject images
├── clip_fidelity_scores.csv        # CLIP scores (108 images)
├── dfc_scores.csv                  # LLaVA scores (108 images)
├── dfc_summary.json                # LLaVA summary
├── cdvr_ablation.csv               # CDVR on/off comparison
├── cdvr_ablation_summary.json      # CDVR summary
├── scorer_agreement.csv            # CLIP vs LLaVA per-image
├── full_analysis_summary.json      # Agreement + bias analysis
├── male_fidelity_scores.csv        # Male subject CLIP scores
clip_zeroshot_validation.json       # CLIP validation on FairFace
```

---

## What NOT to Do

- **Do not** train CLIP classifier heads (`train_clip_fidelity.py`) — the fine-tuned
  approach was tested and abandoned. All results use zero-shot CLIP.
- **Do not** generate or use `.pt` model files (age_classifier.pt, etc.) — they
  are not needed and the training produces unreliable results.
- **Do not** use `clip_fidelity_scorer.py` for batch evaluation — it requires the
  abandoned fine-tuned models. Use zero-shot classification instead.
- **Do not** re-run evaluations that are already complete without coordinating
  with the team — results must remain consistent across the paper.

---

## Troubleshooting

**"No HF_TOKEN found"**
→ Create `.env` file with your HuggingFace token.

**"Cannot reach Ollama"**
→ Run `ollama serve` in a separate terminal.

**"FairFace data not found"**
→ Download from HuggingFace: `load_dataset("HuggingFaceM4/FairFace", "0.25")`

**HF rate limit (429 errors)**
→ Wait for credits to reset, or use local SDXL generation on Kaggle instead of the API.

**Kaggle session dies**
→ Generated images save to `/kaggle/working/` and survive kernel restarts. Re-run the cell and it skips existing images.

feature/cpdc-correction
**"Fine-tuned model not found" / missing .pt files**
→ You do not need these. We use zero-shot CLIP, not fine-tuned heads. See "What NOT to Do" above.
=======
**Generation takes too long**
→ Reduce seeds: `--seeds 42` (36 profiles × 1 seed × 2 = 72 images instead of 216). Still statistically usable if you acknowledge reduced variance in the paper.

## Phase 6: Statistical Validation, Legacy Ablation, Regional Prompt, Gender & Bias Reports

### Note: folder rename
`eval/results/lora_study/` was renamed to `eval/results/regional_prompt_study/`
(LoRA was dropped from the final contribution set in favor of regional
prompt synthesis — see Table 8). `STUDY_DIR` in `clip_zeroshot_score.py`
points at the new folder name.

### Table 6 — Bootstrap CI on CPDC gain

Requires `eval/results/cdvr_ablation.csv` to already exist (Phase 3).
No GPU, no Ollama, no HF token needed — pure statistics on existing data.

```bash
python eval/bootstrap_ci.py
```

To also include the CPDC-vs-legacy-CDVR comparison (Table 7 numbers),
run after Table 7's data exists:

```bash
python eval/bootstrap_ci.py --legacy-csv eval/results/cpdc_vs_cdvr/legacy_ablation.csv
```

Output: `eval/results/bootstrap_ci_summary.json`

### Table 7 — CPDC vs Legacy CDVR ablation

Requires `config.yaml` to have `mild`/`strong` correction keys under
both `STF` and `AF` (previously only BTF had these; STF/AF used the
graduated `level_1`–`level_4` CPDC ladder). Confirm before running:

```python
import yaml
cfg = yaml.safe_load(open("config.yaml"))
print(cfg["corrections"]["STF"].keys())  # should include mild, strong
print(cfg["corrections"]["AF"].keys())
```

Requires Ollama running locally with `llava` pulled, and `HF_TOKEN_1`
(optionally `HF_TOKEN_2`/`HF_TOKEN_3` for rotation) for generation.

```bash
python eval/run_cpdc_vs_cdvr.py --seeds 42
```

Add `--skip-generation` to score existing images and skip (not attempt)
any profile with no image on disk — useful if HF credits run out
mid-run.

Output: `eval/results/cpdc_vs_cdvr/legacy_ablation.csv`,
`eval/results/cpdc_vs_cdvr/comparison_summary.json`

**Known limitation:** our run only covers n=21 of the full 36-profile
grid (HF Inference API credits ran out during collection), vs CPDC's
full n=36. The 21 profiles were supplemented to cover every ethnicity ×
body-type combination at least once, but not full age-range coverage.
See Table 7 in EVALUATION_RESULTS.md.

**Scale note:** `eval/fidelity_scorer.py`'s LLaVA scorer returns
`overall` on a 0–10 scale; `cdvr_ablation.csv`'s `overall_fidelity`
(CPDC) is 0–1. Both `run_cpdc_vs_cdvr.py` and `bootstrap_ci.py` divide
the legacy score by 10 before comparing. If extending these scripts,
apply the same normalization or comparisons will be off by 10x.

### Table 8 — Regional Prompt Ablation

Images generated via FLUX.1-schnell (HuggingFace Spaces), 15 per
config (baseline vs + structured regional prompt), South Indian
demographics. Score with:

```bash
python eval/clip_zeroshot_score.py
```

Output: `eval/results/regional_prompt_study/clip_zeroshot_scores.json`

### Table 9 — Gender Fidelity

Scores male + female generated profiles across all three ethnicities
with CLIP zero-shot classification:

```bash
python eval/gender_fidelity_score.py
```

Requires the male/female profile images already generated (see
`eval/results/male_images/`, `eval/results/female_images/`).

### Table 10 — BiasTracker Distribution Report

Aggregates `modules/bias_tracker.py`'s logged per-run fidelity scores
into a distribution report:

```bash
python eval/generate_bias_report.py
```

Output: `eval/results/bias_report.json` (summary),
`eval/results/bias_log.json` (raw per-run log)

### embedding_viz.py (t-SNE/PCA figure) — still blocked

Requires a `real_refs/` folder with real reference photos, one
subfolder per ethnicity (`real_refs/South Asian/`, etc.) — not yet set
up. Needs a small FairFace sample, similar to Phase 1's download.

```bash
python eval/embedding_viz.py \
    --generated-dir eval/results/cpdc_vs_cdvr/generated_images \
    --real-dir eval/real_refs \
    --method both
```main
