"""
Image generation for AdFidelity.

Two providers, selected by ``generation.provider`` in config.yaml:

  * "cloudflare" - Cloudflare Workers AI, model
                   ``@cf/black-forest-labs/flux-1-schnell``.
                   FREE: 10,000 Neurons/day on the Workers Free plan, no credit
                   card. Needs ``CLOUDFLARE_ACCOUNT_ID`` and
                   ``CLOUDFLARE_API_TOKEN`` in .env.
                   NOTE: the live REST schema accepts only ``prompt`` and
                   ``steps`` — ``seed``/dimensions are rejected with HTTP 400,
                   so outputs are not reproducible on this provider.
  * "hf"         - Hugging Face Inference API
                   (black-forest-labs/FLUX.1-schnell) with HF_TOKEN_1/2/3
                   rotation. Kept as a paid fallback; honours ``seed``.

Both providers return ``(PIL.Image, provider_label, (width, height), steps)``
so the rest of the app is provider-agnostic.
"""

import base64
import os
from io import BytesIO

import requests
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image

load_dotenv()

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
CF_MODEL_ID = "@cf/black-forest-labs/flux-1-schnell"

HF_TOKENS = [t for t in (
    os.getenv("HF_TOKEN_1"),
    os.getenv("HF_TOKEN_2"),
    os.getenv("HF_TOKEN_3"),
) if t]

_fallback_token_idx = 0


def _get_token_idx():
    try:
        if "hf_token_idx" not in st.session_state:
            st.session_state.hf_token_idx = 0
        return st.session_state.hf_token_idx
    except Exception:
        return _fallback_token_idx


def _set_token_idx(idx):
    global _fallback_token_idx
    try:
        st.session_state.hf_token_idx = idx
    except Exception:
        _fallback_token_idx = idx


def _generate_cloudflare(prompt: str, seed: int, config: dict):
    """Generate one image via Cloudflare Workers AI."""
    del seed
    cfg = (config or {}).get("generation", {}) or {}
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise RuntimeError(
            "Cloudflare credentials missing. Add CLOUDFLARE_ACCOUNT_ID and "
            "CLOUDFLARE_API_TOKEN to .env. Free signup (no card): "
            "https://dash.cloudflare.com/sign-up/workers-and-pages -> "
            "Workers AI -> Use REST API."
        )

    model = cfg.get("cloudflare_model") or CF_MODEL_ID
    steps = int(cfg.get("num_inference_steps", 4) or 4)
    steps = max(1, min(steps, 8))
    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{model}"
    )
    payload = {"prompt": prompt, "steps": steps}

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )

    try:
        data = resp.json()
    except ValueError:
        data = {}

    if resp.status_code != 200 or not data.get("success", False):
        detail = data.get("errors") or data.get("error") or resp.text[:300]
        raise RuntimeError(
            f"Cloudflare Workers AI error ({resp.status_code}): {detail}"
        )

    result = data.get("result") or {}
    if result.get("usage"):
        print(f"[flux_pipeline] Cloudflare usage: {result['usage']}")
    img_b64 = result.get("image")
    if not img_b64:
        raise RuntimeError(
            "Cloudflare Workers AI returned no image "
            f"(result keys: {list(result.keys())})"
        )

    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
    return img, "cloudflare", img.size, steps


def _generate_hf(prompt: str, seed: int):
    """Generate one image via the HF Inference API."""
    if not HF_TOKENS:
        raise RuntimeError("No HF_TOKEN_1/2/3 found in environment (.env)")

    errors = []
    start = _get_token_idx()
    for offset in range(len(HF_TOKENS)):
        idx = (start + offset) % len(HF_TOKENS)
        client = InferenceClient(model=MODEL_ID, token=HF_TOKENS[idx], timeout=300)
        try:
            img = client.text_to_image(prompt, seed=seed)
            _set_token_idx(idx)
            return img, "hf-api", (1024, 1024), 4
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            errors.append(f"account {idx + 1} -> {status}")
            if status not in (429, 402, 403):
                raise

    raise RuntimeError(
        f"All {len(HF_TOKENS)} HF accounts failed: {', '.join(errors)}"
    )


def generate_image(prompt: str, seed: int, config: dict = None):
    """Generate one image using the configured provider."""
    cfg = ((config or {}).get("generation") or {})
    provider = (cfg.get("provider") or "cloudflare").strip().lower()
    errors = []

    if provider == "cloudflare":
        try:
            return _generate_cloudflare(prompt, seed, config)
        except Exception as e:  # noqa: BLE001 - fall back to HF if configured
            errors.append(f"cloudflare -> {e}")
            print(f"[flux_pipeline] cloudflare failed: {e}")
            if HF_TOKENS:
                print("[flux_pipeline] falling back to Hugging Face...")
                try:
                    return _generate_hf(prompt, seed)
                except Exception as e2:  # noqa: BLE001
                    errors.append(f"hf -> {e2}")
    elif provider == "hf":
        try:
            return _generate_hf(prompt, seed)
        except Exception as e:  # noqa: BLE001
            errors.append(f"hf -> {e}")
    else:
        raise RuntimeError(
            f"Unknown generation.provider '{provider}' "
            "(expected 'cloudflare' or 'hf')"
        )

    raise RuntimeError("Image generation failed -> " + " | ".join(errors))
