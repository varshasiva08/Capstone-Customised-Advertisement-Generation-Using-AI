# Changes
* Removed LLaVA / Ollama / phi3 from the verification pipeline (parser, profile suggestions, scoring, correction loop). Old LLaVA scoring scripts moved to `eval/legacy/` and docs to `docs/legacy/` (invalid evidence; see banners).
* Product description, storyboard and video modules (`modules/product_describe.py`, `narrative.py`, `ad_compositor.py`, `motion.py`) are the teammates' original code, unchanged; they still use Ollama (llava / phi3:mini).
* New deterministic parser (135 regression cases), gender and extra ethnicities added.
* New `scoring/` verifier (DeepFace, RetinaFace) with pass thresholds; verifier and evaluators are separate; human rating tools.
* CPDC v2: error-proportional, direction-aware, target-aware correction with stall-based early stop; new control arms (resample, best-of-n, fixed escalation).
* New paired experiment runner with bootstrap CIs and exact McNemar.
* Regional cues are non-physical (heritage/attire), not skin-tone.
* Fixed old CLIP scorer label mapping (it merged White/Middle Eastern/Latino/Southeast Asian into other classes).
* Streamlit UI rebuilt: editable parsed profile, verification gallery, prompt trace, no Ollama requirement.


* Multi-account generation fix: `generator/adapter.py` now imports `generator.flux_pipeline` (with account rotation) instead of the non-rotating copy in `eval/generator/`.
* Experiment grew from 48 units / 24 profiles to 80 units / 52 profiles / 272 rated images (batches `main`, `extra1`, `extra2`).
* Two independent raters on all 272 images; a third rater resolves the 21 gender disagreements.
* `eval/rating_app.py` fixed: no pre-selected answers, all four questions required.
* New `eval/analyze_final.py` (arm table, contrasts, McNemar, rater agreement, per-group DeepFace validity, robustness) writing `results/final/`; new helper scripts `check_ratings_vs_target.py`, `make_image_map.py`, `make_gender_check_set.py`.
* Retired `analyze_human_eval.py`, `second_rater_kappa.py`, `list_gender_disagreements.py` (replaced by `analyze_final.py`); stale `new_tables.md` removed.
* Paper reframed around verifier validity; CPDC is the testbed.

