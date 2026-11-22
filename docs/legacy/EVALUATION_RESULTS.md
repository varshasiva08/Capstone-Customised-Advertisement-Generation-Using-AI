> **LEGACY - superseded.** Numbers and claims here come from the old LLaVA-scored pipeline and are not valid evidence. See README.md and CHANGES.md.

# AdFidelity — Evaluation Results

> **Branch:** `main` (merged from `feature/cpdc-correction`)
> **Purpose:** Complete experimental evaluation for IEEE Access paper submission
> **Date:** September 2026

---

## Overview

This document contains the full evaluation results for AdFidelity, a system for generating demographically faithful AI advertisements. The evaluation validates four core claims:

1. AdFidelity generates images that faithfully represent requested demographics (ethnicity, age, gender, body type)
2. The CDVR/CPDC correction loop measurably improves demographic fidelity over single-pass generation
3. Structured regional prompt synthesis improves South Asian demographic confidence
4. The system generates uniformly across all demographic groups with no systematic underrepresentation

Two independent scorers (CLIP zero-shot classification and LLaVA vision-language model) cross-validate results, reducing scorer-specific bias.

---

## Experimental Setup

### Demographic Grid (Main Evaluation)
- **Ethnicities:** South Asian, East Asian, African American (3)
- **Body Types:** slim, medium, plus-size (3)
- **Age Groups:** 20s, 30s, 40s, 50s (4)
- **Gender:** Female (fixed for main eval; gender extension evaluated separately)
- **Total profiles:** 3 × 3 × 4 = 36

### Gender Generalizability (Supplementary)
- **Ethnicities:** South Asian, East Asian, African American (3)
- **Body Type:** medium (fixed)
- **Age Groups:** 20s, 30s, 40s, 50s (4)
- **Gender:** Male
- **Total profiles:** 3 × 4 = 12

### Generation
- **Model:** FLUX.1-schnell (black-forest-labs/FLUX.1-schnell)
- **Seeds:** 42, 123, 456 (main evaluation); 42 (CDVR ablation)
- **Inference steps:** 4 (schnell distilled)
- **Guidance scale:** 0.0 (schnell distilled)

### Scorers
- **CLIP:** Zero-shot classification using OpenAI CLIP ViT-B/32, validated on FairFace dataset
- **LLaVA:** Vision-language model via Ollama, open-ended demographic identification

---

## Results Summary

### Table 1: CLIP Classifier Validation on FairFace (n=10,954)

Validates CLIP as a reliable demographic classifier before using it to score generated images.

| Attribute | Accuracy | Macro F1 | Weighted F1 |
|-----------|----------|----------|-------------|
| Gender    | 94.0%    | 0.94     | 0.94        |
| Race      | 63.0%    | 0.63     | 0.63        |
| Age       | 36.0%    | 0.34     | 0.32        |

**Key findings:** Gender classification is highly reliable (94%). Race classification performs well above chance (63% vs 14.3% random for 7 classes). Age classification is weakest (36%), consistent with known difficulty of age estimation from faces — this is a limitation acknowledged in the paper.

**Per-class race performance:**

| Race | Precision | Recall | F1 |
|------|-----------|--------|----|
| Black | 0.83 | 0.85 | 0.84 |
| East Asian | 0.73 | 0.67 | 0.70 |
| Indian | 0.72 | 0.68 | 0.70 |
| Latino_Hispanic | 0.40 | 0.51 | 0.45 |
| Middle Eastern | 0.49 | 0.48 | 0.49 |
| Southeast Asian | 0.50 | 0.75 | 0.60 |
| White | 0.89 | 0.48 | 0.62 |

---

### Table 2: Demographic Fidelity — CLIP Scorer (n=108)

Zero-shot CLIP classification on 108 generated advertisement images (36 profiles × 3 seeds).

| Metric | Accuracy |
|--------|----------|
| Ethnicity match | 96.3% |
| Gender match | 100.0% |
| Age match | 34.3% |
| **Overall fidelity** | **76.9% ± 16.7%** |

**Per-ethnicity breakdown:**

| Ethnicity | Race Accuracy | Overall Fidelity |
|-----------|---------------|------------------|
| South Asian | 100.0% | 76.9% |
| East Asian | 88.9% | 78.7% |
| African American | 100.0% | 75.0% |

---

### Table 3: Demographic Fidelity — LLaVA/DFC Scorer (n=108)

LLaVA vision-language model scoring the same 108 images via open-ended demographic identification.

| Metric | Accuracy |
|--------|----------|
| Ethnicity match | 89.8% |
| Gender match | 100.0% |
| Age match | 95.4% |
| **Mean DFC** | **95.1%** |

---

### Table 4: Scorer Agreement — CLIP vs LLaVA

Cross-validation between two independent scorers on the same 108 images.

| Axis | CLIP | LLaVA | Agreement |
|------|------|-------|-----------|
| Ethnicity | 96.3% | 89.8% | Both strong; CLIP slightly higher |
| Gender | 100.0% | 100.0% | Perfect agreement |
| Age | 34.3% | 95.4% | Divergent — see note below |

**Note on age divergence:** CLIP's zero-shot age classification is inherently weak (validated at only 36% on FairFace), while LLaVA uses open-ended reasoning with fuzzy matching (±10 year tolerance). The divergence reflects scorer methodology, not generation quality. For ethnicity and gender — the primary claims of this paper — both scorers strongly agree.

---

### Table 5: CDVR Ablation (n=36, seed=42)

Compares single-pass generation (no correction) vs CDVR iterative correction loop (up to 3 iterations).

| Metric | Without CDVR | With CDVR | Improvement |
|--------|-------------|-----------|-------------|
| Ethnicity match | 88.9% | 100.0% | **+11.1%** |
| Age match | 36.1% | 38.9% | +2.8% |
| Gender match | 100.0% | 100.0% | +0.0% |
| **Overall fidelity** | **75.0%** | **79.7%** | **+4.6%** |

- **Images needing correction:** 25/36 (69%)
- **Average iterations used:** 2.0

**Key finding:** CDVR achieved **100% ethnicity accuracy** (up from 88.9%), demonstrating that the iterative correction loop is essential for reliable demographic representation. The 69% correction rate confirms that single-pass generation frequently fails to match intended demographics, validating CDVR's necessity.

---

### Table 6: Bias Distribution

Verifies that AdFidelity generates uniformly across all demographic groups (no underrepresentation).

| Dimension | Groups | Count per Group | χ² (vs uniform) |
|-----------|--------|-----------------|------------------|
| Ethnicity | 3 | 36 each | 0.0 (perfectly uniform) |
| Body Type | 3 | 36 each | 0.0 (perfectly uniform) |
| Age | 4 | 27 each | 0.0 (perfectly uniform) |

Generation is uniformly distributed by design (exhaustive grid), confirming no demographic group is underrepresented in the evaluation.

---

### Table 7: Male Subject Evaluation — Gender Generalizability (n=12)

Preliminary evaluation on male subjects to validate that the pipeline is not gender-dependent.

| Metric | Male (n=12) | Female (n=108) |
|--------|-------------|----------------|
| Ethnicity match | 83.3% | 96.3% |
| Gender match | 100.0% | 100.0% |
| Age match | 50.0% | 34.3% |
| **Overall fidelity** | **77.8% ± 21.7%** | **76.9% ± 16.7%** |

**Per-ethnicity breakdown (male):**

| Ethnicity | Race Accuracy | Overall Fidelity |
|-----------|---------------|------------------|
| South Asian | 100.0% | 83.4% |
| East Asian | 100.0% | 83.4% |
| African American | 50.0% | 66.7% |

**Key finding:** Overall fidelity is comparable across genders (77.8% male vs 76.9% female), confirming that the pipeline's demographic fidelity is not gender-dependent. African American male representation shows lower race accuracy (50%) but on a very small sample (n=4), so further evaluation with larger samples is recommended.

---

### Table 8: Output Quality Metrics (from automated evaluation report)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Resolution | 1024 × 1024 px | High resolution |
| Sharpness (Laplacian variance) | 621.59 | Sharp, well-focused |
| Aesthetic Quality Proxy | 89.44 / 100 | High aesthetic quality |
| NIQE | 7.04 | Good (lower is better) |
| BRISQUE | 60.62 | Acceptable (lower is better) |
| CLIP Prompt Adherence | 32.79 | Strong text-image alignment |
| Scene Consistency (SSIM) | 0.78 | Stable across frames |
| Temporal Flicker | 0.045 | Low flicker (video) |
| Background Consistency | 0.87 | Stable background |

---
---

---
---

# Tables 9-13 (revised, human-validated evaluation)

> **Scope.** Tables 1-8 above are unchanged. The legacy banner at the top of this file applies to them, not to this section: Tables 9-13 were re-done with a new design (separate verifier and evaluators, equal-budget control, blind human labels, profile-clustered intervals). The older Tables 9, 10 and 12 are superseded; the older Tables 11 and 13 are kept below only as feature demonstrations.

## Triage of the old Tables 9-13

| Old table | What it claimed | Why it is not valid evidence | Status | Replaced by |
|---|---|---|---|---|
| 9 Bootstrap CI on CPDC gain (n=36) | Gain of +0.046, CI excludes zero | Loop driven by LLaVA confidence with the target shown in the prompt; scorer also acted as verifier; no human labels | Superseded | New Table C |
| 10 CPDC vs legacy CDVR | 0.797 vs 0.735 mean fidelity | Unpaired (n = 21 vs 36), no equal-budget control | Superseded | New Tables B and C |
| 11 Regional prompt ablation | South Asian confidence 0.802 to 0.872 | n = 15, CLIP confidence only, 100% accuracy in both configurations (no accuracy gain shown) | Feature demo only | Not used as evidence |
| 12 Gender fidelity (n=33) | 97% combined gender accuracy | CLIP only, three ethnicities, no human labels | Superseded | New Table D2 |
| 13 BiasTracker report | No group under-represented | Coverage is balanced by construction of the grid; age score is a scorer limit | Feature demo only | New Tables D and E |

## New evaluation design (summary)

- **Generator:** FLUX.1-schnell on Cloudflare Workers AI (free tier). No seed control, so repeated calls are independent random draws.
- **Verifier in the loop:** DeepFace (RetinaFace). Gender passes at P >= 0.8; age within 3 years of the target decade; ethnicity passes if the target is the top class and P >= 0.25, plus a lenient South Asian rule P(indian) >= 0.25 (chosen after the pilot, so it is a post-hoc setting).
- **Arms (48 units each):** vanilla (one image); resample (same prompt, new draw, equal call budget); CPDC v2 (corrects failing axes only, worst axis first, wording capped at level 3, direction-aware for age, stops on pass or stall).
- **Design:** 6 ethnicities x 2 genders x 2 ages (30s, 50s) = 24 profiles; the first 12 repeated three times = 48 units.
- **Graders:** DeepFace (verifier), CLIP ViT-L/14 zero-shot (separate evaluator, run on face crops), and one human rater (145 images, blind).
- **Statistics:** profile-clustered bootstrap, exact McNemar, Wilson intervals, Cohen's kappa, AUC.
- All results below describe **apparent** ethnicity, gender and age as judged by a classifier or a rater. They are not claims about anyone's identity.

## Which new tables the paper needs

| Level | Tables |
|---|---|
| Required | A (setup), C (arms under three graders), D (verifier vs human by ethnicity), E (South Asian detail) |
| Recommended | B (DeepFace-graded arms), D2 (gender and age axes), F (combined verifier), G (cost per success) |
| Optional | H (recovery of failed units), I (does ethnicity wording help) |

---

## New tables

### Table A. Experimental setup

- Units (profile x repeat): **48** from **24** distinct profiles; arms: vanilla, resample, cpdc; up to 4 generator calls per unit; attempt 0 is shared by the three arms.
- Images rated by one human rater: **145** (all unique attempt images).
- Units per target ethnicity:

| ethnicity | profiles | units |
|---|---|---|
| African American | 4 | 12 |
| East Asian | 4 | 12 |
| Latin American | 4 | 4 |
| Middle Eastern | 4 | 4 |
| South Asian | 4 | 12 |
| White | 4 | 4 |

**Reading note:** only the African American, East Asian and South Asian profiles were repeated three times. Latin American, Middle Eastern and White have one unit per profile (4 units each), so results for those three groups are pilot-level. All intervals are bootstrapped over profiles because repeats of a profile are not independent.

*Paper status: Required.*

---

### Table B. Arm comparison graded by the DeepFace verifier (optimistic: the verifier also chose the final image)

| arm | images/unit | ethnicity | gender | age | all axes |
|---|---|---|---|---|---|
| vanilla | 1.0 | 72.9% | 91.7% | 72.9% | 56.2% |
| resample | 2.06 | 87.5% | 100.0% | 87.5% | 75.0% |
| cpdc | 1.96 | 83.3% | 97.9% | 89.6% | 72.9% |

**Reading note:** optimistic by construction. DeepFace both chose the final image and graded it, so this table shows what the verifier believes, not what is true. Table C regrades the same images independently.

*Paper status: Recommended.*

---

### Table C. The same final images graded by three different graders (n = 48 units per arm)

| arm | images/unit | DeepFace: all axes | Human: all axes (age +/-10y) | DeepFace: eth+gender | CLIP: eth+gender | Human: eth+gender |
|---|---|---|---|---|---|---|
| vanilla | 1.0 | 56.2% | 75.0% | 68.8% | 83.3% | 81.2% |
| resample | 2.06 | 75.0% | 77.1% | 87.5% | 87.5% | 83.3% |
| cpdc | 1.96 | 72.9% | 85.4% | 81.2% | 85.4% | 89.6% |

Differences in percentage points with 95% CI (bootstrap over profiles):

| contrast | DeepFace all axes | Human all axes | CLIP eth+gender | Human eth+gender |
|---|---|---|---|---|
| resample - vanilla | +18.8 [+8.3, +30.8] | +2.1 [+0.0, +6.5] | +4.2 [+0.0, +10.9] | +2.1 [+0.0, +6.5] |
| cpdc - vanilla | +16.7 [+8.3, +26.0] | +10.4 [+2.4, +19.2] | +2.1 [+0.0, +6.5] | +8.3 [+1.9, +16.7] |
| cpdc - resample | -2.1 [-9.6, +4.3] | +8.3 [+1.9, +16.7] | -2.1 [-7.4, +0.0] | +6.2 [+0.0, +14.3] |

Units where CPDC differs from resample on human all-axes: better in **4**, worse in **0**, tied in **44** of 48.

**Reading note (main result):** the +18.8 resample gain under DeepFace shrinks to +2.1 under human labels, so most of the verifier-graded gain is the verifier agreeing with itself. CPDC is +8.3 over resample on human all-axes (CI +1.9 to +16.7), but this comes from only 4 of 48 units, and the ranking reverses under DeepFace and CLIP. Report it as suggestive, not established. Human all-axes uses age within 10 years; DeepFace uses 3 years.

*Paper status: Required.*

---

### Table D. Ethnicity verifiers against the human label, by target group

| target | n images | human: reads as target | DeepFace passes | CLIP passes | DeepFace agrees with human | CLIP agrees with human |
|---|---|---|---|---|---|---|
| African American | 31 | 100% | 97% | 100% | 97% | 100% |
| East Asian | 22 | 95% | 100% | 95% | 95% | 91% |
| Latin American | 10 | 20% | 40% | 30% | 80% | 90% |
| Middle Eastern | 19 | 37% | 37% | 58% | 58% | 79% |
| South Asian | 56 | 80% | 20% | 93% | 36% | 80% |
| White | 7 | 100% | 100% | 29% | 100% | 29% |
| All | 145 | 78% | 56% | 83% | 67% | 84% |

Reference: answering "matches" for every image would agree with the human on 77.9% of images. Cohen's kappa with the human: DeepFace 0.29, CLIP 0.50.

**Reading note:** DeepFace passed 20% of South Asian target images that the human rater judged 80% correct, and agrees with the human on only 36% of them. CLIP is better overall (kappa 0.50 vs 0.29) but passes only 29% of White images the human judged correct (n = 7). Latin American (n = 10) and Middle Eastern (n = 19) are mostly judged as not matching by the human, which points to generator weakness for those groups; with one rater and few units this is a pilot observation only.

*Paper status: Required.*

---

### Table D2. Gender and age axes against the human label

| axis | n images | human: matches | DeepFace passes | CLIP matches | human: within 10 years |
|---|---|---|---|---|---|
| gender (target: female) | 89 | 99% | 76% | 100% |  |
| gender (target: male) | 56 | 88% | 100% | 100% |  |
| gender (target: all) | 145 | 94% | 86% | 100% |  |
| age (target: 30s) | 47 |  | 89% | n/a | 98% |
| age (target: 50s) | 98 |  | 51% | n/a | 95% |
| age (target: all) | 145 |  | 63% | n/a | 96% |

Exact-decade age agreement between the human and the target: 44% (within one decade: 96%). CLIP has no age output.

**Reading note:** DeepFace's gender check rejects many female-target images the human accepted (76% pass vs 99% human match). For male targets the human accepted 88% while both automatic checks passed 100%; these disagreements (8 images in total across both genders) are still to be re-checked. Human age is a visual estimate of the decade, so it is reported within 10 years only.

*Paper status: Recommended.*

---

### Table E. South Asian target images: can the verifiers tell good from bad? (n = 56; human-yes 45, human-no 11)

| verifier | passes of human-yes | passes of human-no | AUC of P(target) | kappa |
|---|---|---|---|---|
| DeepFace (lenient South Asian rule) | 10/45 | 1/11 | 0.54 | 0.06 |
| CLIP ViT-L/14 zero-shot | 43/45 | 9/11 | 0.46 | 0.18 |

Human-no South Asian images by profile id: p000 (1), p001 (5), p003 (5) (profiles are 30s/50s female/male South Asian).

**Reading note:** neither verifier can separate good from bad South Asian images (AUC near 0.5). DeepFace is too strict (passes 10 of 45 human-correct images); CLIP is too lenient (passes 9 of 11 human-incorrect images). 10 of the 11 human-incorrect images come from two profiles (p001, p003), so the failure examples are concentrated.

*Paper status: Required.*

---

### Table F. Can a combined verifier beat the raw ones? (ethnicity axis, leave-one-profile-out)

| verifier | accuracy (all 145) | accuracy (South Asian) | AUC (all) | AUC (South Asian) |
|---|---|---|---|---|
| always "matches" | 77.9% | 80.4% | 0.50 | 0.50 |
| DeepFace pass/fail | 66.9% | 35.7% | 0.70 | 0.57 |
| CLIP top class = target | 84.1% | 80.4% | 0.79 | 0.46 |
| logistic: CLIP P(target) (leave-one-profile-out) | 79.3% | 78.6% | 0.70 | 0.44 |
| logistic: CLIP + DeepFace (leave-one-profile-out) | 84.8% | 78.6% | 0.82 | 0.52 |

**Reading note (negative result):** on South Asian images no verifier, raw or combined, beats answering "matches" every time (80.4%). The combined verifier is no better than CLIP alone overall (84.8% vs 84.1% accuracy). No calibrated verifier is claimed. South Asian has 56 images and 11 negatives from 4 profiles, so differences between AUCs of 0.44 to 0.57 are within noise.

*Paper status: Recommended.*

---

### Table G. Cost per success (all axes; final selected image)

| arm | images generated | DeepFace successes | images per DeepFace success | human successes | images per human success |
|---|---|---|---|---|---|
| vanilla | 48 | 27 | 1.78 | 36 | 1.33 |
| resample | 99 | 36 | 2.75 | 37 | 2.68 |
| cpdc | 94 | 35 | 2.69 | 41 | 2.29 |

**Reading note:** vanilla is the cheapest per success. CPDC costs less per human success than resample (2.29 vs 2.68 images), but both cost more than vanilla for an equal or modest gain.

*Paper status: Recommended.*

---

### Table H. Recovery of units whose first image failed the DeepFace verifier

| group | arm | units failing at attempt 0 (DeepFace) | fixed by the end (DeepFace) | judged OK by human at the end | mean images |
|---|---|---|---|---|---|
| South Asian targets | vanilla | 9 | 0 (0%) | 6 (67%) | 1.00 |
| South Asian targets | resample | 9 | 3 (33%) | 7 (78%) | 3.67 |
| South Asian targets | cpdc | 9 | 3 (33%) | 8 (89%) | 3.22 |
| other targets | vanilla | 12 | 0 (0%) | 7 (58%) | 1.00 |
| other targets | resample | 12 | 6 (50%) | 7 (58%) | 3.25 |
| other targets | cpdc | 12 | 5 (42%) | 10 (83%) | 3.17 |

**Reading note (optional):** among units whose first image failed DeepFace, the human judged more CPDC final images acceptable than resample (South Asian 8/9 vs 7/9; other targets 10/12 vs 7/12), small counts only.

*Paper status: Optional.*

---

### Table I. Does ethnicity wording help? Retry images in units whose attempt-0 image failed DeepFace's ethnicity check

| retry type | retry images | ethnicity passes (DeepFace) | human says ethnicity matches |
|---|---|---|---|
| cpdc, ethnicity wording applied | 21 | 3 (14%) | 14 (67%) |
| cpdc, other axis wording | 9 | 3 (33%) | 4 (44%) |
| resample (no wording change) | 35 | 9 (26%) | 23 (66%) |

DeepFace pass rate of the image produced in each CPDC correction round, by the axis that was corrected:

| correction axis (cpdc rounds) | rounds | ethnicity pass | gender pass | age pass |
|---|---|---|---|---|
| age | 18 | 78% | 56% | 50% |
| ethnicity | 21 | 14% | 100% | 86% |
| gender | 7 | 71% | 57% | 43% |

**Reading note (optional):** correction rounds that applied ethnicity wording passed DeepFace's ethnicity check 14% of the time (3/21), no better than plain resampling (26%, 9/35); human judgement is similar (67% vs 66%). So CPDC's gain, if real, is not explained by ethnicity wording; do not claim that mechanism.

*Paper status: Optional.*


---

## Findings supported by the data

1. **Verifier-graded gains overstate real gains.** Resample vs vanilla is +18.8 points under DeepFace and +2.1 [0.0, +6.5] under human labels (Table C).
2. **DeepFace is biased as a verifier on South Asian faces.** It passed 10 of 45 images the human judged correct (kappa 0.06, AUC 0.54; Table E). It also rejects many correct female-target images (Table D2).
3. **A second automatic evaluator helps but is not ground truth.** CLIP agrees with the human on 84.1% of ethnicity judgements vs 66.9% for DeepFace and 77.9% for answering "matches" always, but it fails on White images and cannot separate good from bad South Asian images (AUC 0.46; Tables D and E).
4. **CPDC is suggestive, not established.** +8.3 points over resample on human all-axes (CI +1.9 to +16.7), from 4 of 48 units, reversing under the automatic graders (Table C). Its ethnicity wording did not measurably help (Table I).
5. **Negative result.** A combined CLIP + DeepFace verifier does no better than "always matches" on South Asian images (Table F). No calibrated verifier or learned policy is claimed.

## Limitations

- One human rater (the author), so no inter-rater kappa yet; a second rater is needed. Ethnicity labels are inherently subjective, especially for Latin American and Middle Eastern images.
- 48 units; CPDC vs resample differs in only 4 units.
- Latin American, Middle Eastern and White have 4 units each; only African American, East Asian and South Asian were repeated.
- DeepFace is both verifier and grader in the configuration, so DeepFace numbers are verifier-graded (Table B).
- The lenient South Asian rule was chosen after the pilot.
- 8 gender disagreements between the human labels and the automatic checks are unresolved.
- Generator has no seed control, and some calls may have fallen back to Hugging Face; confirm that no fallback images are in `results/main`.
- Human age is a visual estimate and is scored within 10 years.

## Reproduction of new tables

```
python analyze_human_eval.py --results results\main\results.csv --attempts results\main\attempts.csv --ratings results\main\rating\ratings_person1.csv --key results\main\rating\rating_key_DO_NOT_SHARE.csv --map rating_map.csv --clip clip_scores.csv --out new_tables.md
```

Table F needs scikit-learn (`pip install scikit-learn`). CLIP scores come from `clip_verifier_check.py` (`clip_scores.csv`).

---

## Old Table 11 (kept as a feature demonstration, not evidence)

> Shows that the regional prompt builder runs and produces images; it does not show improved fidelity (accuracy is 100% in both configurations, n = 15, CLIP confidence only).

### Old Table 11: Regional Prompt Ablation — South Indian Demographics (n=15 per config)

Isolates the contribution of structured regional prompt synthesis for South Indian demographic generation.

| Configuration | Race Accuracy | Mean SA Confidence | n |
|---------------|---------------|--------------------|---|
| Baseline FLUX (generic "Indian woman") | 100% | 0.802 | 15 |
| + Structured Regional Prompt | 100% | **0.872** | 15 |

**What this table shows:** While FLUX.1-schnell achieves 100% South Asian classification for both configurations, structured regional prompt synthesis increases mean South Asian confidence by +8.7% (0.802 → 0.872). This means regional prompts produce more demographically distinctive outputs — the model's confidence in the correct demographic increases even when it already gets the ethnicity right. This is the primary evidence for the regional prompt synthesis contribution.

**Scored using:** Zero-shot CLIP ViT-B/32 (`eval/clip_zeroshot_score.py`)
**Images generated using:** FLUX.1-schnell via HuggingFace Spaces (15 images per config)

---

## Old Table 13 (kept as a feature demonstration, not evidence)

> Shows that the BiasTracker logs coverage. Coverage is balanced because the grid was balanced; it is not a fairness result. The fairness finding is in new Tables D and E.

### Old Table 13: BiasTracker Distribution Report (n=108)

Monitors demographic coverage across all generated outputs to detect systematic underrepresentation.

| Dimension | Groups | Distribution | Underrepresented? |
|-----------|--------|--------------|-------------------|
| Ethnicity | African American, East Asian, South Asian | 33.3% each | None |
| Body Type | medium, plus-size, slim | 33.3% each | None |
| Age | 20s, 30s, 40s, 50s | 25.0% each | None |

**Average fidelity across all logged runs:**

| Metric | Score |
|--------|-------|
| Gender fidelity | 10.0/10 |
| Ethnicity fidelity | 9.63/10 |
| Age fidelity | 3.43/10 |
| Overall | 7.69/10 |

**What this table shows:** BiasTracker confirms no demographic group is systematically underrepresented — all groups appear at expected frequencies. The age fidelity score (3.43/10) is consistent with CLIP's validated age accuracy (Table 1: 36%) and is a scorer limitation, not a generation failure. BiasTracker runs continuously during app use to flag demographic drift in real deployments.

**Generated by:** `eval/generate_bias_report.py` using `modules/bias_tracker.py`

---
## File Structure

```
eval/
├── results/
│   ├── generated_images/             # 108 generated advertisement images (female)
│   │   ├── South-Asian_slim_20s_seed42.png
│   │   ├── South-Asian_slim_20s_seed123.png
│   │   └── ... (108 total)
│   ├── male_images/                  # 12 generated images (male) — Table 7
│   │   ├── South-Asian_20s_male_seed42.png
│   │   └── ... (12 total)
│   ├── clip_fidelity_scores.csv      # Table 2 raw scores
│   ├── male_fidelity_scores.csv      # Table 7 raw scores
│   ├── dfc_scores.csv                # Table 3 raw scores
│   ├── dfc_summary.json              # Table 3 summary
│   ├── scorer_agreement.csv          # Table 4 raw
│   ├── full_analysis_summary.json    # Table 4 summary
│   ├── cdvr_ablation.csv             # Table 5 raw
│   ├── cdvr_ablation_summary.json    # Table 5 summary
│   ├── cdvr_ablation_images/         # Before/after CDVR images
│   │   ├── South-Asian_slim_20s_seed42_noCDVR.png
│   │   ├── South-Asian_slim_20s_seed42_CDVR.png
│   │   └── ... (72 total)
│   ├── bootstrap_ci_summary.json     # old Table 9 (superseded by new Table C)
│   ├── cpdc_vs_cdvr/
│   │   ├── comparison_summary.json   # old Table 10 (superseded by new Table C)
│   │   └── legacy_ablation.csv       # old Table 10 raw
│   ├── regional_prompt_study/
│   │   └── clip_zeroshot_scores.json # old Table 11 (feature demo)
│   ├── bias_report.json              # old Table 13 (feature demo)
│   └── bias_log.json                 # old Table 13 raw log
├── clip_trainer/
│   ├── train_clip_fidelity.py        # CLIP classifier training script
│   ├── clip_fidelity_scorer.py       # CLIP scoring (fixed race mapping)
│   └── validate_classifier.py        # Table 1 validation
├── run_full_evaluation.py            # Master evaluation script (Tables 2-5, 7)
├── run_llava_scoring.py              # LLaVA/DFC scoring script (Table 3)
├── compute_agreement_and_bias.py     # Agreement + bias analysis (Table 4)
├── fidelity_scorer.py                # Original DFC scorer
├── clip_zeroshot_score.py            # Table 11 scorer
├── gender_fidelity_score.py          # Table 12 scorer
├── generate_bias_report.py           # Table 13 generator
├── bootstrap_ci.py                   # Table 9 generator
├── run_cpdc_vs_cdvr.py               # Table 10 generator
├── embedding_viz.py                  # Figure: embedding space
├── llm_judge.py                      # LLaVA-guided CPDC correction
├── generate_lora_study.py            # Table 11 image generation
clip_zeroshot_validation.json         # CLIP validation on FairFace (11k images) — Table 1
metrics_20260824_215800.json          # Output quality metrics — Table 8
EVALUATION_GUIDE.md                   # Step-by-step reproduction guide
```

---

## Reproduction

| Table | Script | Runtime | Requirements |
|-------|--------|---------|--------------|
| Table 1 | `eval/clip_trainer/validate_classifier.py` | ~10 min | FairFace dataset |
| Tables 2-5, 7 | `eval/run_full_evaluation.py` | ~4 hours | GPU, Ollama+LLaVA |
| old Table 9 (superseded) | `eval/bootstrap_ci.py` | ~1 min | `cdvr_ablation.csv` |
| old Table 10 (superseded) | `eval/run_cpdc_vs_cdvr.py` | ~1 min | `cdvr_ablation.csv` |
| old Table 11 (feature demo) | `eval/clip_zeroshot_score.py` | ~2 min | Generated images |
| old Table 12 (superseded by new Table D2) | `eval/gender_fidelity_score.py` | ~2 min | Male/female images |
| old Table 13 (feature demo) | `eval/generate_bias_report.py` | ~1 min | `clip_fidelity_scores.csv` |
| New Tables A-I | `analyze_human_eval.py` (see Reproduction of new tables above) | ~1 min | `results/main/*`, human ratings, `clip_scores.csv` |

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for complete step-by-step instructions.

---

## Citation

If you use this evaluation framework or results, please cite:

```
@article{adfidelity2026,
  title={AdFidelity: Demographically Faithful AI Advertisement Generation with Closed-Loop Verification},
  year={2026}
}
```