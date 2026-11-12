# AdFidelity

Generate ad images of a person described in a brief, then **verify** the result
with a face-attribute model (ethnicity, gender, age) and **correct the prompt**
until it matches - no LLM, no local server required.

## Install
```
pip install -r requirements.txt          # app
pip install -r requirements-eval.txt     # experiments
```
Product description, taglines and storyboard/video use a local Ollama (`ollama pull llava`, `ollama pull phi3:mini`); the verification pipeline does not.
Set CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN in `.env` for FLUX.

## Run
```
streamlit run app.py
```
Flow: describe the person -> confirm the parsed profile (editable) -> pick
verify mode and attempts -> generate. Each attempt shows per-axis pass/fail,
the best attempt is selected, and the prompt trace is available. Body type is
used in the prompt but is **not** auto-verified. "Demo (sample images)" runs the
interface offline with bundled sample photos.

## How it works
* `profiler/text_parser.py` - deterministic brief parser with evidence, low-confidence and unsupported reporting.
* `scoring/` - DeepFace-based verifier, pass/fail + normalised error per axis, cache, human-rating tools.
* `modules/cpdc.py`, `modules/correction_loop.py` - policies: vanilla, best_of_n, resample, fixed_escalation, cpdc (+ ablations).
* `eval/run_experiment.py` - paired experiment runner (`--dry-run` simulates the pipeline for logic checks only).

## Test
```
python -m pytest tests
python tests/test_product_integration.py
python eval/parser_accuracy.py
```
Parser cases are a regression suite written alongside the rules, not an
independent accuracy estimate. Verifier estimates (especially age and ethnicity)
are approximate; validate with human ratings (`scoring/human.py`) before
reporting results.
