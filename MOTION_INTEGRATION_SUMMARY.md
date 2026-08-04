# Module 5 — Motion / Animation: Summary

## What this module does
Takes the final advertisement image (model + product already baked in by Module 4) and animates it into a short walking video, so the ad can be delivered as a video instead of only a static photo — instead of the placeholder that previously did nothing.

## How it works
```
Final ad image (from Module 4, product already baked in)
        ↓
Submitted to Viggle's motion-transfer API (model: V4_Preview) along
with a pre-recorded walking driving video
        ↓
Viggle returns an animated video of the character walking, plus a
matching alpha mask isolating her from Viggle's own rendered background
        ↓
The animated figure is composited onto a flat color plate sampled from
the corners of the original ad photo (same background color the model
was generated on), so the final video keeps the same clean studio look
as the static image
```

## New / changed files
| File | What changed |
|---|---|
| `modules/motion.py` | **Implemented** (was a `NotImplementedError` placeholder). Full submit → poll → download → composite pipeline against the Viggle API. |
| `config.yaml` | Added a `viggle:` section — API base URL, model version, background mode, path to the driving video. |
| `.env` | Added `VIGGLE_API_KEY` (not committed — already covered by `.gitignore`). |
| `requirements.txt` | Added `opencv-python`, `imageio`, `imageio-ffmpeg` — used by `motion.py` (and, it turns out, already needed by `modules/product_overlay.py` but never listed). |
| `app.py` | "Animation style" dropdown enabled (was `disabled=True`). Added an explicit "Also animate this ad (uses Viggle credits)" checkbox. Wired `animate_image()` into `render_output()` with its own status spinner and error handling, matching the existing image-generation pattern. Added a warning when animation is combined with the 6-profile bulk-generation flow, since that would fire 6 separate paid Viggle jobs from one click. |

## Approaches tried, and why we landed here
1. **Leave the product in the source photo, let Viggle animate it natively** — what's implemented now. Tested directly: Viggle's `V4_Preview` model carries a held product through motion transfer reasonably faithfully on its own, without needing any extra compositing logic on our end.
2. **Strip the product out before animating, reattach it afterward** — built and tested first, before landing on approach 1. Precisely inpaints the product out of the source photo (using a GrabCut-derived silhouette mask rather than a plain bounding box — a plain rectangular mask blends background color into the surrounding clothing across its corners and produces a visible diagonal smear artifact), animates the now-product-free character, then re-composites the product back in per frame, tracked to the wrist via MediaPipe's `PoseLandmarker` (the current Tasks API — the older `mp.solutions.pose` API is deprecated and unreliable in current `mediapipe` pip releases). Abandoned as the default once testing showed approach 1 already handles the product well enough on its own, and this path adds real fragility of its own: wrist tracking on a synthetic (AI-generated) driving video occasionally lost the landmark mid-clip and froze the product in place, causing it to visibly float away from the hand.
3. **Hallucination behavior observed from Viggle itself, not from our code**: in isolated test runs, Viggle sometimes rendered a rough approximation of the product *and* a second, different hallucinated object near it — this happened both with the product left in the photo and, once, even after it was stripped out. Not reliably reproducible; most `V4_Preview` runs with the product left in rendered correctly.

## Known limitations (important — flag this in the writeup)
- **Body shape drift.** Viggle's motion transfer retargets the character's proportions toward the driving video performer's build as an inherent part of how it maps motion onto a new character — the further apart the two builds are, the more visible the drift. Switching to `model: V4_Preview` (from the older default `V3_Preview`) reduces this but does not eliminate it. The only remaining lever under our control is sourcing a driving video with a body type closer to the generated model's, which we don't currently have tooling to guarantee or automate.
- **Only one motion is actually implemented.** The "Animation style" dropdown still shows all three original placeholder options (Subtle sway / Camera pan / Zoom in) for UI consistency, but every option currently triggers the same walking animation under the hood — only one driving video has been sourced and validated so far.
- **Costs real money per generation, including failed attempts.** Viggle bills per second of output video (~$0.01/sec) regardless of whether the result turns out usable. A job-caching layer (skip resubmission if a completed job already exists) was built and validated during prototyping, but is **not** wired into the production app — every click of "Also animate this ad" in the current `app.py` submits a fresh paid job.
- **Aspect-ratio sensitivity.** The driving video and the source photo should share a reasonably similar aspect ratio; a marked mismatch produced a squashed, tiny-in-frame, visibly distorted result in testing.
- **Occasional motion shakiness**, observed on at least one driving video swap; root cause wasn't fully isolated — plausibly related to the aspect-ratio issue above, but not confirmed as the same cause.

## How to test it
1. Ensure `.env` has `VIGGLE_API_KEY` set, and `config.yaml`'s `viggle.driving_video` points to a real file on disk (e.g. `driving_videos/walking_forward_back.mp4`).
2. `pip install -r requirements.txt`
3. Run the app, enable image generation in Settings
4. Enter a brief → generate a profile → optionally add a product → check **"Also animate this ad (uses Viggle credits)"** → Generate
5. Watch the status messages during generation — shows Viggle submission/render progress before the composited video appears inline, underneath the static image

## Possible next steps (if more time)
- Source and validate additional driving videos so "Subtle sway" / "Camera pan" / "Zoom in" become functionally distinct instead of all mapping to the same walk
- Wire the job-caching layer into the production app itself (not just the prototype), so accidental Streamlit re-runs can't silently re-charge
- Systematically investigate the shakiness and aspect-ratio sensitivity, ideally against a small set of driving videos of known-good, consistent aspect ratios
- Revisit the strip/reattach fallback (Approach 2 above) as an opt-in path for specific products that don't render well natively, rather than an all-or-nothing choice