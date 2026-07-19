"""
Product Describer - uses LLaVA (via Hugging Face Inference API) to turn
an uploaded product photo into a short text description, which can then
be injected into the FLUX generation prompt so the model is drawn
*holding* the product from the start (rather than a product cutout being
pasted onto a finished photo afterward).

Reuses the same HF_TOKEN_1/2/3 rotation approach as generator/flux_pipeline.py.

Note: this describes the product in words - it does not transplant the
exact uploaded pixels into the output. FLUX draws a new product matching
the description, integrated naturally into the pose/lighting/shadows.
"""

import base64
import os
from io import BytesIO

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image

load_dotenv()

DFC_MODEL_ID = "google/gemma-3-27b-it"  # confirmed live: Scaleway, Featherless AI, DeepInfra

HF_TOKENS = [t for t in (
    os.getenv("HF_TOKEN_1"),
    os.getenv("HF_TOKEN_2"),
    os.getenv("HF_TOKEN_3"),
) if t]

DESCRIBE_INSTRUCTION = (
    "Describe this product in one short phrase suitable for an image "
    "generation prompt: include its type, material, color, and shape. "
    "Do not include any brand names or logos. "
    "Example: 'black leather structured handbag with gold buckle and top handle'. "
    "Reply with ONLY the phrase, nothing else."
)


def _image_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


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


def describe_product(product_image: Image.Image) -> str:
    """
    Send the uploaded product image to LLaVA and get back a short text
    description usable inside an image-generation prompt.

    Falls back to a generic phrase if the call fails, so the app can
    still proceed with generation instead of hard-erroring.
    """
    if not HF_TOKENS:
        raise RuntimeError("No HF_TOKEN_1/2/3 found in environment (.env)")

    data_url = _image_to_data_url(product_image)
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


# ---------- CLI test ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python product_describe.py <path_to_product_image>")
        sys.exit(1)
    img = Image.open(sys.argv[1])
    print("Description:", describe_product(img))
