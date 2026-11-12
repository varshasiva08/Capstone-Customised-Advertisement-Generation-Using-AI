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
