# Module 4 — Product Integration: Summary

## What this module does
Adds a product (handbag, sunglasses, jewelry, etc.) to the AI-generated model photo, so the final ad shows the model naturally wearing/carrying it — instead of the placeholder that previously did nothing.

## How it works
Instead of pasting a product cutout onto a finished photo (tried first, looked fake — see "approaches tried" below), the final approach **bakes the product into the image generation itself**:

```
User uploads product photo
        ↓
Vision AI (Gemma-3-27B) describes it in words
        e.g. "cream faux leather tote bag with chain strap"
        ↓
Description gets woven into the FLUX prompt, with pose instructions
matched to how that product type is normally worn/carried
        ↓
FLUX generates the model already wearing/carrying the product
        (correct shadows, occlusion, and integration — because it's
        one coherent image, not two images glued together)
```

## New / changed files
| File | What changed |
|---|---|
| `modules/product_describe.py` | **New.** Sends the uploaded product photo to a vision model, gets back a short text description. |
| `profiler/prompt_builder.py` | Added `product_description` and `product_category` params. Builds a pose-specific clause (different phrasing for handbags vs. sunglasses vs. jewelry vs. other) and injects it into the prompt. |
| `config.yaml` | Base prompt now explicitly requests a front-facing pose with visible face (was previously ambiguous, sometimes generated side-profile shots). |
| `app.py` | Product upload UI now live (was disabled placeholder). Added a "Product type" dropdown. Wired the describe → prompt → generate flow together. |

## Approaches tried, and why we landed here
1. **Corner-placement compositing** (`rembg`/OpenCV background removal + paste in a corner) — worked technically, but looked like a sticker, not something she was wearing. Abandoned for realism reasons.
2. **True inpainting** (mask her hand, regenerate just that region) — technically the "correct" way to do exact-pixel product placement, but the models needed for this aren't available on Hugging Face's free API (checked directly — confirmed "not deployed by any provider"). Would require running a model fully locally, which is slow on a CPU-only Mac. Set aside due to time constraints.
3. **Description-based generation** (current approach) — what's implemented now. Trades exact pixel-for-pixel product fidelity for realistic, natural integration (correct shadows/occlusion/pose) using tools that actually run on our setup.

## Known limitation (important — flag this in the writeup)
**The product in the final image is the AI's own reinterpretation of the description, not your literal uploaded photo.** Color/shape/style will be close but not identical to the exact product image uploaded. This is a deliberate tradeoff: true pixel-exact placement needs local inpainting infrastructure we didn't have time to build reliably. Works best with simple, rigid, clearly-shaped products (bags, sunglasses, jewelry) — struggles more with unusual shapes or busy backgrounds in the uploaded photo.

## How to test it
1. `pip install -r requirements.txt`
2. Run the app, enable image generation in Settings
3. Enter a brief → say yes to product → upload a product photo → pick the matching "Product type" → Generate
4. Watch the status messages during generation — shows what description the AI extracted from your upload before generating

## Possible next steps (if more time)
- Local inpainting for exact product fidelity (needs GPU or patience with CPU)
- Expand `product_category` options / auto-detect category instead of manual dropdown
- Tune pose phrasing further per product type based on more test generations
