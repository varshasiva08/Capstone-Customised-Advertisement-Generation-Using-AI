# Module 5 — Motion / Animation: Summary

## What this module does

Takes the final advertisement image and animates it into a short video using Wan2.2-I2V — free via the public Hugging Face Space (`wan.provider: space`) by default, with the paid fal-ai path retained as a fallback. No local GPU required. No driving video needed.

## How it works

```
Final ad image (from FLUX, product baked in)
        ↓
[motion.py] build_motion_prompt()
  → Constructs a demographically-aware motion description
    from the user's profile and clothing type
  e.g. saree → "graceful walk, fabric draping naturally"
       suit  → "confident professional stride, upright posture"
        ↓
Wan2.2-I2V Lightning (4-step distilled) — free public HF Space
(Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom) by default, fal-ai fallback
  → Returns a ~3.5s clip of the figure moving
        ↓
[ad_compositor.py] apply_ad_overlay()
  → Loops video to target duration (5/10/15/20s) with seamless crossfade
  → Fades into brand card: brand name + tagline + product image
  → Output: final .mp4 advertisement
```

## Key contribution

The motion prompt is not hardcoded. It is constructed dynamically from the demographic profile and clothing context. A woman in a saree receives a different motion description than one in a blazer suit — producing more contextually appropriate animation per demographic target.

Two motion styles are available:
- **Sway & turn** — gentle body sway in place, slight torso turn to show the outfit/product. Best for jewellery, bags, accessories.
- **Walk & pose** — walks three slow steps toward camera, stops in a natural pose. Best for clothing, sarees, suits.

Both explicitly block unwanted motion: `no dancing, no jumping, no spinning, no dramatic gestures`.

## Files

| File | Role |
|---|---|
| `modules/motion.py` | Motion prompt builder + Wan2.2-I2V API call |
| `modules/ad_compositor.py` | Video looping, brand card, fade transition |
| `config.yaml` | `wan:` section — model/space ID and provider (`space` free / `fal-ai` paid) |
| `.env` | Cloudflare pair for images; HF tokens only needed for paid fallbacks |

## Migration from Viggle

This module originally used Viggle's motion-transfer API with a pre-recorded driving video. Replaced with Wan2.2-I2V for:
- No driving video dependency
- Prompt-guided motion (Viggle had no prompt input)
- Free default — no per-credit API cost on the public Space (queue times vary)
- Better identity preservation (no body shape drift from driving video performer)

## Known limitations

- Default path is the free public HF Space — free but shared-queue (waits vary); the paid fal-ai fallback costs roughly $0.05–0.10 per generation
- Generation takes 60–120 seconds per video
- Motion quality depends on prompt adherence — Wan occasionally interprets prompts loosely
- Video duration from Wan2.1 is fixed (~5–10s server-side); duration control is achieved by looping in the compositor

## How to test

1. No token needed for the free Space path (only `CLOUDFLARE_*` for images)
2. Run the app, complete a brief, reach the Generate stage
3. Enable the "Generate video" toggle in the sidebar
4. Select motion style and duration
5. Click Generate — status messages show progress
