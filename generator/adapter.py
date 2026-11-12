"""Generator adapters: one `generate(prompt, seed) -> PIL.Image` interface."""
from __future__ import annotations

import glob
import os
import random

from PIL import Image

MOCK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "mock_samples")


def make_generate_fn(cfg: dict, provider: str = None, profile: dict = None):
    """
    provider: None -> config generation.provider ("cloudflare" | "hf");
              "mock" -> sample images bundled in assets/mock_samples (UI / smoke tests ONLY).
    """
    provider = provider or (cfg.get("generation", {}) or {}).get("provider", "cloudflare")
    if provider == "mock":
        return _mock(profile)

    from eval.generator.flux_pipeline import generate_image

    def gen(prompt: str, seed: int):
        import time
        last = None
        for attempt in range(3):
            try:
                img, _device, _res, _steps = generate_image(prompt, int(seed), cfg)
                return img
            except RuntimeError as e:
                msg = str(e).lower()
                # daily quota / auth problems will not fix themselves: stop at once
                if "429" in msg or "daily free allocation" in msg or "401" in msg or "403" in msg:
                    raise
                last = e                                  # timeouts / connection errors: wait, retry
                print(f"[generator] transient error, retry {attempt + 1}/3 in 30s: {str(e)[:120]}")
                time.sleep(30)
        raise last
    gen.supports_seed = provider != "cloudflare"          # Cloudflare REST rejects `seed`
    return gen


def _mock(profile):
    files = sorted(glob.glob(os.path.join(MOCK_DIR, "*.jpg")))
    if not files:
        raise RuntimeError("no mock samples found in assets/mock_samples")
    want = ""
    if profile:
        want = (profile.get("ethnicity", "").replace(" ", "-") + "_" + profile.get("age", "")).lower()
    pool = [f for f in files if want and os.path.basename(f).lower().startswith(want)] or files

    def gen(prompt: str, seed: int):
        return Image.open(random.Random(seed).choice(pool)).convert("RGB")
    gen.supports_seed = True
    gen.is_mock = True
    return gen
