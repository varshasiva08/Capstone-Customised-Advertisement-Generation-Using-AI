# AdFidelity — Full Evaluation Guide

Complete step-by-step instructions to run every experiment needed for the IEEE Access paper.

---

## Overview: What gets produced

| Output | Paper section | Script |
|--------|---------------|--------|
| DFC fidelity scores (36 profiles × N seeds × CDVR on/off) | Table 1: Demographic Fidelity Results | `eval/run_full_evaluation.py` |
| CDVR ablation (iteration 0 vs final, with/without correction) | Table 2: CDVR Effectiveness | same |
| CLIP classifier validation (accuracy, F1, confusion matrices on FairFace val) | Section IV-B: CLIP Classifier Validation | `eval/clip_trainer/validate_classifier.py` |
| CLIP fidelity scores on generated images | Table 3: CLIP-based Fidelity | `eval/run_full_evaluation.py` |
| Scorer agreement (LLaVA vs CLIP correlation) | Section IV-C: Cross-Metric Validation | same |
| Bias distribution report | Table 4: Demographic Distribution | same |
| Per-group fidelity breakdown | Figure: Per-Group Fidelity Heatmap | same |

---

## Phase 0: Environment Setup

### 0.1 — Clone and switch to the evaluation branch

```bash
git clone https://github.com/varshasiva08/Capstone-Customised-Advertisement-Generation-Using-AI.git
cd Capstone-Customised-Advertisement-Generation-Using-AI
git checkout experiments/full-evaluation
```

If the branch doesn't exist on remote yet (you haven't pushed), create it locally:

```bash
git checkout main
git checkout -b experiments/full-evaluation
```

Then copy the new files from this guide into the right locations.

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
HF_TOKEN_2=hf_optional_second_token
HF_TOKEN_3=hf_optional_third_token
```

Get tokens from https://huggingface.co/settings/tokens (free account works, but you'll need to accept the FLUX model terms at https://huggingface.co/black-forest-labs/FLUX.1-schnell).

---

## Phase 1: Download FairFace Dataset

This is needed for the CLIP study. FairFace provides ~100k face images labeled with age, gender, and race.

### 1.1 — Download

Go to the FairFace GitHub: https://github.com/joojs/fairface

Download these files (from their Google Drive links on the README):
- `fairface-img-margin025-trainval.zip` (~3.4 GB)
- `fairface_label_train.csv`
- `fairface_label_val.csv`

### 1.2 — Extract to the right location

```bash
mkdir -p eval/clip_trainer/fairface_data

# Unzip the images
unzip fairface-img-margin025-trainval.zip -d eval/clip_trainer/fairface_data/

# Copy the CSVs
cp fairface_label_train.csv eval/clip_trainer/fairface_data/
cp fairface_label_val.csv eval/clip_trainer/fairface_data/
```

### 1.3 — Verify the folder structure

```
eval/clip_trainer/fairface_data/
    train/
        1.jpg, 2.jpg, ...    (~86k images)
    val/
        1.jpg, 2.jpg, ...    (~11k images)
    fairface_label_train.csv
    fairface_label_val.csv
```

Check the CSV columns — you need at minimum: `file`, `age`, `gender`, `race`.

Note: FairFace images may be in subdirectories like `train/1.jpg` — the CSV's `file` column contains the relative path. If your extracted structure has them at `fairface_data/train/1.jpg` then the paths should align. If not, check the CSV and adjust.

---

## Phase 2: Train CLIP Classifier Heads

### 2.1 — Run training

```bash
cd Capstone-Customised-Advertisement-Generation-Using-AI/
python eval/clip_trainer/train_clip_fidelity.py
```

This will:
- Download CLIP ViT-B/32 (~600 MB, automatic, first run only)
- Fine-tune 3 small MLP heads (age, gender, race) on frozen CLIP embeddings
- Takes ~30–60 min on CPU, ~5–10 min on GPU
- Progress prints every 10 batches

### 2.2 — Verify outputs

```bash
ls eval/clip_fidelity_model/
```

You should see:
```
age_classifier.pt
gender_classifier.pt
race_classifier.pt
label_maps.json
```

### 2.3 — Validate the classifier (for the paper)

```bash
python eval/clip_trainer/validate_classifier.py
```

This scores the classifier on FairFace's own validation set and produces:
- Per-class precision, recall, F1 for age/gender/race
- Confusion matrices
- Overall accuracy

Results saved to `eval/clip_trainer/validation_results.json`.

**These numbers go in your paper** — they establish how reliable your automated scorer is before you use it to score generated images.

---

## Phase 3: Generate Images + Score Everything

### 3.1 — Run the full evaluation

```bash
python eval/run_full_evaluation.py --seeds 42 123 456
```

This generates the full 36-profile × 3-seed × 2-condition (CDVR on/off) = **216 images**, then scores each with both LLaVA and CLIP.

**Estimated time:**
- Generation: ~45s per image × 216 = ~2.7 hours (HF API, depends on rate limits)
- LLaVA scoring: ~10s per image × 216 = ~36 min
- CLIP scoring: ~2s per image × 216 = ~7 min
- **Total: ~3.5–4 hours**

### 3.2 — If you get rate-limited or interrupted

The script skips images that already exist on disk. Just re-run the same command:

```bash
python eval/run_full_evaluation.py --seeds 42 123 456
```

It will pick up where it left off.

### 3.3 — If you want to score without regenerating

```bash
python eval/run_full_evaluation.py --skip-generation --seeds 42 123 456
```

### 3.4 — If you haven't trained CLIP yet but want LLaVA scores

```bash
python eval/run_full_evaluation.py --skip-clip --seeds 42 123 456
```

### 3.5 — Verify outputs

```bash
ls eval/results/
```

You should see:
```
generated_images/       — 216 PNG files
manifest.json           — generation metadata
dfc_scores.csv          — LLaVA fidelity scores (216 rows)
clip_scores.csv         — CLIP fidelity scores (216 rows)
scorer_agreement.csv    — per-image LLaVA vs CLIP comparison
summary.json            — aggregate stats ready for paper tables
```

---

## Phase 4: Quick-Test a Single Image (Optional)

To score one image without running the full batch:

### LLaVA scorer:
```bash
python eval/fidelity_scorer.py
```
(Edit the `eval_profiles` list in the script first.)

### CLIP scorer:
```bash
python eval/clip_trainer/clip_fidelity_scorer.py \
    --image outputs/your_image.png \
    --ethnicity "South Asian" \
    --age "30s" \
    --gender "Female"
```

---

## Phase 5: Interpreting Results for the Paper

### What goes where:

**Table 1 — Demographic Fidelity (LLaVA DFC)**
- From `dfc_scores.csv`: group by ethnicity/body_type/age, report mean ± std of `overall` score
- Compare CDVR=True vs CDVR=False rows

**Table 2 — CDVR Effectiveness**
- From `summary.json` → `llava_fidelity.cdvr_on` vs `cdvr_off`
- Report mean fidelity improvement and avg iterations needed

**Table 3 — CLIP Classifier Validation**
- From `eval/clip_trainer/validation_results.json`
- Report per-class F1 and overall accuracy for age, gender, race

**Table 4 — CLIP Fidelity on Generated Images**
- From `clip_scores.csv`: same grouping as Table 1

**Table 5 — Scorer Agreement**
- From `scorer_agreement.csv`: report Pearson correlation between LLaVA and CLIP overall scores

**Table 6 — Bias Distribution**
- Count how many images per ethnicity/body_type/age group were successfully generated
- Chi-square test against uniform distribution

---

## Troubleshooting

**"No HF_TOKEN found"**
→ Create `.env` file with your HuggingFace token.

**"Cannot reach Ollama"**
→ Run `ollama serve` in a separate terminal.

**"FairFace training CSV not found"**
→ Download FairFace dataset (Phase 1) and extract to the correct folder.

**"Fine-tuned model not found"**
→ Run `train_clip_fidelity.py` first (Phase 2).

**HF rate limit (429 errors)**
→ Add HF_TOKEN_2 and HF_TOKEN_3 to `.env` for rotation, or wait and re-run (existing images are skipped).

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
```