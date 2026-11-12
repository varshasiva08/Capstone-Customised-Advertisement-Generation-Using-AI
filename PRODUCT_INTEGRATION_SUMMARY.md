# Module 4 — Product Integration: Summary

## What this module does

Adds a product (handbag, sunglasses, jewellery, etc.) to the generated advertisement image so the model appears to naturally wear or carry it, rather than having it pasted on afterward.

## How it works

```
User uploads product photo
        ↓
[product_describe.py] Local vision model (Ollama llava; HF Gemma-3 fallback)
  → Describes the product in words
  e.g. "burgundy structured leather handbag with gold hardware"
        ↓
[prompt_builder.py] Product clause injected into FLUX prompt
  → Pose phrasing matched to product category:
    handbag  → "hanging from her shoulder, carried naturally"
    jewelry  → "wearing necklace, clearly visible, realistic scale"
    sunglasses → "worn on face, fitted naturally, resting on bridge of nose"
        ↓
FLUX.1-schnell generates model already wearing/carrying the product
  → Correct shadows, lighting, occlusion — one coherent image
```

## Key contribution

Product integration is description-based rather than compositing-based. This means the product is generated as part of the scene with correct lighting, shadow, and occlusion — not pasted on afterward as a separate layer. The prompt category system ensures pose phrasing is contextually appropriate per product type.

The category is inferred **from the vision description of the actual uploaded photo first** (falling back to the brief, then a neutral default) — so a necklace or sunglasses is no longer phrased as a handbag. Each category's clause then states the physical cues the model otherwise misses: an explicit scale anchor, a visible grip/wearing point, a contact shadow, and partial occlusion by the hand or body.

## Files

| File | Role |
|---|---|
| `modules/product_describe.py` | Sends uploaded photo to the local vision model (Ollama llava), returns text description |
| `profiler/product_category.py` | Shared product vocabulary + inference (brief keywords and photo description) |
| `profiler/prompt_builder.py` | Injects product clause into prompt with category-matched pose phrasing |
| `modules/product_overlay.py` | GrabCut-based static compositing (fallback, available but not primary) |

## Known limitation

The product in the final image is FLUX's reinterpretation of the text description — close in colour, shape, and style but not pixel-identical to the uploaded photo. This is a deliberate tradeoff: description-based generation produces natural integration (correct shadows, pose, occlusion) that compositing cannot match. Works best with simple, clearly-shaped products (bags, sunglasses, jewellery).

## Auto-detection

The product category is auto-detected from the brand brief keywords (e.g. "jewellery" → `jewelry`, "bag" → `handbag`). The user is not asked to confirm the category manually — it is used silently in the prompt builder.
