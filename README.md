# AdFidelity

Generate ad images of a person described in a brief, **verify** the image with a
face-attribute model (ethnicity, gender, age), and **correct the prompt** until it
matches. The evaluated path uses no LLM and no paid API.

This repository accompanies the paper *Confidence-Proportional Demographic
Correction and Verifier Validity in Generated Advertisement Images*.

## What is and is not evaluated

| Part | Status |
|---|---|
| Brief parser, DeepFace verifier, CPDC loop (`profiler/`, `scoring/`, `modules/cpdc.py`, `modules/correction_loop.py`) | Evaluated in the paper |
| Product description, tagline, storyboard and video (`modules/product_describe.py`, `narrative.py`, `ad_compositor.py`, `motion.py`) | Demo only, not evaluated; they need a local Ollama |

## Install
```
pip install -r requirements.txt          # app
pip install -r requirements-eval.txt     # experiments
```
Create a `.env` file (never commit it):
```
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
```
For more free-tier quota add further accounts as `CLOUDFLARE_ACCOUNT_ID_2`, `CLOUDFLARE_API_TOKEN_2`, and so on up to `_6`; `generator/flux_pipeline.py` rotates through them when one hits its daily limit.
The demo-only modules need `ollama pull llava` and `ollama pull phi3:mini`; the verification pipeline does not.

## Run the app
```
streamlit run app.py
```
Describe the person, confirm the parsed profile (editable), choose verify mode and attempts, generate. Each attempt shows per-axis pass/fail, the best attempt is selected, and the prompt trace is available. Body type is used in the prompt but is **not** verified. "Demo (sample images)" runs the interface offline.

## Reproduce the experiments
The paper uses three batches: `main`, `extra1`, `extra2`. Each is produced by `eval/run_experiment.py` and its settings are stored in `results/<batch>/summary.json` under `meta`.
```
python eval/run_experiment.py --out results/<batch> --arms vanilla resample cpdc --ethnicities ... --genders ... --ages ... --repeats N --max-calls 4
python eval/run_experiment.py --out results/dry --dry-run        # logic check only, no models or API
```
Images come from a generator without seed control, so a new run gives new images, not the paper's. The rated images and labels are provided instead.

### Human rating protocol
1. `python eval/make_rating_set.py results/<batch>/rating` builds a blind, shuffled set and a private key. **Never run it on `results/main`; it rebuilds the folder.**
2. Each rater runs `streamlit run eval/rating_app.py -- results/<batch>/rating` and answers all four questions per image. The app has no pre-selected answers.
3. Send raters only the blind images and sheet. Never send the `*_key*.csv` files.
4. `python eval/make_image_map.py results/<batch>` maps rated images back to units, arms and attempts by file hash.
5. Gender disagreements between raters are resolved by a third rater on a blind set built with `python eval/make_gender_check_set.py` (the 21 image ids are listed in the script).

### Final analysis
Expected layout:
```
results/<batch>/results.csv, attempts.csv
results/<batch>/rating/ratings_person1.csv, ratings_person2.csv
rating_map.csv                    (project root, main batch)
results/extra1|extra2/image_map.csv
rater3/ratings_person3.csv         rater3_key_PRIVATE.csv
```
```
python eval/analyze_final.py
```
writes `results/final/*.csv` and `results/final/tables.md` (arm table, contrasts with profile-cluster bootstrap CIs, exact McNemar, rater agreement with kappa, DeepFace-versus-rater agreement by group, robustness check). Every number in the paper's results section comes from this file. `python eval/check_ratings_vs_target.py results/<batch>/rating person1` is a read-only sanity check of one rater against the targets; do not edit labels after looking at it.

CLIP agreement (`clip_verifier_check.py`, `clip_scores.csv`) covers the original 145 images only.

## Test
```
python -m pytest -q
python tests/test_product_integration.py
python eval/parser_accuracy.py
```
The parser cases are a regression suite, not an independent accuracy estimate.

## Layout
```
app.py                    Streamlit UI
profiler/                 brief parser
scoring/                  DeepFace verifier, thresholds, human-rating helpers
modules/                  CPDC, correction loop, demo-only product/video modules
generator/                FLUX client with multi-account rotation
eval/                     experiment runner, rating app, statistics, analysis
eval/legacy, docs/legacy  earlier LLaVA-era work, kept as history; not valid evidence
results/                  per-batch data and results/final
```

## Privacy and release
Do not publish `.env`, any `*_key*.csv`, `rater3_key_PRIVATE.csv`, or `_archive/`. The keys map images to targets and would unblind the ratings. Release them only after review, if at all.
