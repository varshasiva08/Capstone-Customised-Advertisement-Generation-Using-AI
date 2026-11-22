"""
Product Describer - turns an uploaded product photo into a short text
description, which is injected into the FLUX generation prompt so the model
is drawn *holding* the product from the start (rather than a product cutout
being pasted onto a finished photo afterward).

Providers (config: product_describe.provider):
  * "ollama" - local LLaVA via the Ollama server. FREE and offline; no HF
               credits, no API keys. Pull the model once:  ollama pull llava
  * "hf"     - Gemma-3 vision via the Hugging Face Inference API (paid
               fallback). Reuses the HF_TOKEN_1/2/3 rotation.

Note: this describes the product in words - it does not transplant the exact
uploaded pixels into the output. FLUX draws a new product matching the
description, integrated naturally into the pose/lighting/shadows.
"""

import base64
import os
from io import BytesIO

import requests
import yaml
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image

load_dotenv()

DFC_MODEL_ID = "google/gemma-3-27b-it"  # HF fallback only

HF_TOKENS = [t for t in (
    os.getenv("HF_TOKEN_1"),
    os.getenv("HF_TOKEN_2"),
    os.getenv("HF_TOKEN_3"),
) if t]

DESCRIBE_INSTRUCTION = (
    "You are looking at a single product photo for an advertisement. "
    "Answer in ONE short phrase (about 25 words max) that begins with the "
    "product type and includes: its material, colour, shape, its approximate "
    "size relative to a person (small / medium / large), and how it is worn "
    "or carried. Do not include brand names or logos, and do not describe "
    "the background. Examples: 'medium black leather structured handbag with "
    "gold buckle and top handle, carried by hand'; 'thin gold chain necklace "
    "with a small round pendant, worn around the neck'. "
    "Reply with ONLY the phrase, nothing else."
)


def _load_config(config: dict = None) -> dict:
    if config is not None:
        return config
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001 - config is optional here
        print(f"[product_describe] could not read config.yaml: {e}")
        return {}


def _image_to_base64(image: Image.Image) -> str:
    """Raw base64 JPEG - the form Ollama's /api/chat expects."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_to_data_url(image: Image.Image) -> str:
    """data: URL form - the form the HF chat API expects."""
    return f"data:image/jpeg;base64,{_image_to_base64(image)}"


def _clean_description(text: str) -> str:
    """
    Vision models often wrap the answer in a preamble sentence and/or
    markdown bold, e.g.:
        "Here's a description in one short phrase:\n**Cream tote bag.**"
    This extracts just the clean phrase.
    """
    import re

    # Prefer text inside **bold** markers if present - that's usually
    # the actual answer, with the preamble being plain text around it.
    bold_match = re.search(r"\*\*(.+?)\*\*", text, re.DOTALL)
    if bold_match:
        text = bold_match.group(1)

    # Drop a leading "Here's ... :" style preamble if bold wasn't used
    text = re.split(r":\s*\n", text)[-1] if ":\n" in text else text
    text = text.strip().strip('"\'').rstrip(".")
    # Collapse any remaining newlines/extra whitespace
    text = " ".join(text.split())
    return text


def _describe_ollama(image: Image.Image, cfg: dict) -> str:
    """Describe via the local Ollama server (free, offline)."""
    host = (cfg.get("host") or "http://localhost:11434").rstrip("/")
    model = cfg.get("model") or "llava"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": DESCRIBE_INSTRUCTION,
            "images": [_image_to_base64(image)],
        }],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 120},
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "")
        description = _clean_description(content)
        if description:
            return description
        print("[product_describe] ollama returned an empty description")
    except Exception as e:  # noqa: BLE001 - fall back gracefully
        print(f"[product_describe] ollama error: {type(e).__name__}: {e}")
    return "stylish accessory"


def _describe_hf(image: Image.Image) -> str:
    """Describe via the HF Inference API (paid fallback)."""
    if not HF_TOKENS:
        raise RuntimeError("No HF_TOKEN_1/2/3 found in environment (.env)")

    data_url = _image_to_data_url(image)
    errors = []

    for token in HF_TOKENS:
        client = InferenceClient(token=token, timeout=60)
        try:
            completion = client.chat.completions.create(
                model=DFC_MODEL_ID,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": DESCRIBE_INSTRUCTION},
                    ],
                }],
                max_tokens=60,
            )
            description = completion.choices[0].message.content.strip()
            description = _clean_description(description)
            return description
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            print(f"[product_describe] HfHubHTTPError (status {status}): {e}")
            errors.append(f"status {status}")
            continue
        except Exception as e:  # noqa: BLE001 - want to fall back on any failure
            print(f"[product_describe] {type(e).__name__}: {e}")
            errors.append(str(e))
            continue

    # Fallback: don't crash the whole generation over a describe failure
    return "stylish accessory"


def describe_product(product_image: Image.Image, config: dict = None) -> str:
    """
    Turn the uploaded product image into a short text description usable
    inside an image-generation prompt.

    Uses the configured provider (default: local Ollama llava). Falls back
    to a generic phrase if the call fails, so the app can still proceed with
    generation instead of hard-erroring.
    """
    cfg = _load_config(config).get("product_describe", {}) or {}
    provider = (cfg.get("provider") or "ollama").strip().lower()

    if provider in ("hf", "huggingface"):
        try:
            return _describe_hf(product_image)
        except Exception as e:  # noqa: BLE001
            print(f"[product_describe] hf error: {e}")
            return "stylish accessory"

    return _describe_ollama(product_image, cfg)


# ---------- CLI test ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python product_describe.py <path_to_product_image>")
        sys.exit(1)
    img = Image.open(sys.argv[1])
    print("Description:", describe_product(img))
