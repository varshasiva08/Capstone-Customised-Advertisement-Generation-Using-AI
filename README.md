# AdFidelity — Demographic-Faithful Advertisement Generation

AdFidelity is an end-to-end AI pipeline that generates customised advertisement images and videos from a free-text brand brief, with explicit demographic control and fidelity verification.

## What it does

A brand describes their target audience in plain English ("plus-size South Asian woman in her 40s for a jewellery campaign"). AdFidelity parses that brief, builds a demographically-faithful prompt, generates a studio-quality advertisement image via FLUX.1-schnell, optionally composites a product, animates the figure using Wan2.2-I2V, and appends a branded end card — all in one pipeline.

## System Architecture

```
Brand Brief (free text)
        ↓
[text_parser.py]  Hybrid Ollama + keyword parser → demographic profile
        ↓
[prompt_builder.py]  Structured prompt synthesis + CDVR correction loop
        ↓
[flux_pipeline.py]  FLUX.1-schnell via Cloudflare Workers AI (free) → advertisement image
        ↓
[product_describe.py + prompt_builder.py]  Optional product integration (description-baked)
        ↓
[motion.py]  Wan2.2-I2V via free HF Space + demographically-aware motion prompt → video
        ↓
[ad_compositor.py]  Brand card with tagline + product image → final ad video
        ↓
[bias_tracker.py]  Demographic distribution monitoring across generated outputs
        ↓
[eval/fidelity_scorer.py]  LLaVA-based offline fidelity evaluation (DFC)
```

## Key Contributions

- **Structured demographic prompt synthesis** — converts free-text brand briefs into demographically constrained image generation prompts via a hybrid rule-based + LLM parser
- **CDVR (Contextual Demographic Verification and Refinement) loop** — iterative prompt correction with escalating severity when generated images fail fidelity checks
- **Demographically-aware motion prompt builder** — constructs contextual motion descriptions from the profile and clothing type (saree vs. suit vs. kurta), rather than using a fixed driving video
- **Ad compositor** — seamless forward-loop video with brand card, tagline (auto-generated or custom), and product image
- **BiasTracker** — monitors demographic distribution across generated ad batches, flags underrepresented groups

## Setup

### Prerequisites
- Python 3.10+
- Ollama running locally with phi3:mini and llava pulled
- A free Cloudflare account (no credit card) for image generation
- HuggingFace tokens are **optional** — only used as a paid fallback

### Install

```bash
pip install -r requirements.txt
ollama pull phi3:mini
ollama pull llava
```

### Environment

Create a `.env` file. Only the Cloudflare pair is required; the HF tokens are
optional paid fallbacks:

```
# --- Free image generation: Cloudflare Workers AI (FLUX.1-schnell) ---
# Sign up free at https://dash.cloudflare.com/sign-up/workers-and-pages
# then Workers AI -> "Use REST API" -> create a token and copy the Account ID.
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_token

# --- Optional paid fallback (HF Inference API) ---
HF_TOKEN_1=hf_your_token_here
HF_TOKEN_2=hf_second_token   # optional, for rotation
HF_TOKEN_3=hf_third_token    # optional, for rotation
```

### Provider switches

All three AI steps are provider-gated in `config.yaml`, so any step can be
moved back to the paid path with a one-line change:

| Step | Free default | Paid fallback |
|---|---|---|
| Image | `generation.provider: cloudflare` | `hf` |
| Product description | `product_describe.provider: ollama` | `hf` |
| Video | `wan.provider: space` (free public HF Space) | `fal-ai` |

### Run

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                    # Streamlit UI — sidebar controls, chat interface
├── config.yaml               # Prompt template, options, correction tokens
├── generator/
│   └── flux_pipeline.py      # FLUX.1-schnell via Cloudflare Workers AI (HF fallback)
├── profiler/
│   ├── text_parser.py        # Hybrid brief parser (Ollama + keyword rules)
│   ├── prompt_builder.py     # Prompt synthesis + CDVR correction loop
│   └── profile_gen.py        # 6-profile generator for vague briefs
├── modules/
│   ├── motion.py             # Wan2.2-I2V animation + motion prompt builder
│   ├── ad_compositor.py      # Brand card, tagline, forward-loop video
│   ├── product_describe.py   # Vision-based product description
│   ├── product_overlay.py    # GrabCut-based product compositing (static)
│   └── bias_tracker.py       # Demographic distribution monitoring
└── eval/
    └── fidelity_scorer.py    # LLaVA-based offline DFC evaluation + batch runner
```

## Evaluation

To run fidelity scoring on a batch of generated images:

```bash
python eval/fidelity_scorer.py
```

Edit the `eval_profiles` list in the script to match your evaluation set. Results are saved to `eval/fidelity_results.csv`.

## Notes

- Free by default: images via Cloudflare Workers AI (10,000 Neurons/day), product descriptions via local Ollama, and video via the free public Wan2.1 HF Space (queue times vary). The HF `fal-ai` path is a paid fallback only.
- Image generation is a fresh roll each time: the free Cloudflare FLUX.1-schnell REST endpoint rejects the `seed` parameter (HTTP 400), so outputs are not reproducible there and the Seed control was removed from the UI. The internal seed still names the output file and is honoured by the HF fallback.
- The product in the generated image is FLUX's reinterpretation of the uploaded product description — close but not pixel-identical to the uploaded photo. This is a deliberate tradeoff for natural integration.
- Ollama (phi3:mini) must be running before launching the app.
