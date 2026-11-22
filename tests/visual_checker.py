"""
tests/visual_checker.py

Automated visual-integration checker for Product Integration.

Mirrors the pattern already used in eval/fidelity_scorer.py (vision model
as judge) - instead of scoring demographic fidelity, this scores whether
the product was integrated realistically: face visible, product visible,
facing camera, plausible placement/shadow.

This replaces manual screenshot-and-judge-by-eye logging with an automated
score + CSV log, using the same Gemma-3-27B model already wired up in
modules/product_describe.py.

Usage (single image):
    from tests.visual_checker import score_visual_integration
    result = score_visual_integration("outputs/output_seed42.png",
                                       product_category="handbag")

Usage (CLI, batch over a folder):
    python3 tests/visual_checker.py outputs/
"""

import base64
import csv
import json
import os
import re
import sys
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

load_dotenv()

JUDGE_MODEL_ID = "google/gemma-3-27b-it"  # same model as modules/product_describe.py

HF_TOKENS = [t for t in (
    os.getenv("HF_TOKEN_1"),
    os.getenv("HF_TOKEN_2"),
    os.getenv("HF_TOKEN_3"),
) if t]

JUDGE_INSTRUCTION = (
    "You are QA-checking an AI-generated fashion advertisement photo. "
    "Look at the image and answer honestly. Reply ONLY with compact JSON, "
    "no extra text, in exactly this format:\n"
    '{{"face_visible": true/false, '
    '"facing_camera": true/false, '
    '"product_visible": true/false, '
    '"product_plausible_placement": true/false, '
    '"overall_realism_score": X, '
    '"notes": "..."}}\n\n'
    "Definitions:\n"
    "- face_visible: is the person's face clearly visible (not turned away, "
    "not obscured)?\n"
    "- facing_camera: is the person's body oriented toward the camera "
    "(not a side/back profile)?\n"
    "- product_visible: is a {category} clearly visible somewhere on the "
    "person?\n"
    "- product_plausible_placement: does the {category} look like it is "
    "naturally worn/carried (not floating, not pasted, correct location "
    "for that item type)?\n"
    "- overall_realism_score: 0-10, how realistic does the whole image look "
    "as a professional ad photo?"
)


def _image_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _extract_json(text: str) -> dict:
    """Vision models often wrap JSON in markdown fences or add preamble."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()
    return json.loads(text)


def score_visual_integration(image_path: str, product_category: str = "product") -> dict:
    """
    Send a generated image to the vision model and get back a structured
    judgment of whether the product was integrated realistically.

    Returns a dict with the judged fields, plus 'pass' (bool) and 'error'
    (str or None). Never raises - failures come back as a zero/False result
    with 'error' populated, so a batch run doesn't stop on one bad image.
    """
    if not HF_TOKENS:
        return _error_result("No HF_TOKEN_1/2/3 found in environment (.env)")
    if not os.path.exists(image_path):
        return _error_result(f"image not found: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as e:
        return _error_result(f"could not open image: {e}")

    data_url = _image_to_data_url(img)
    instruction = JUDGE_INSTRUCTION.format(category=product_category)

    for token in HF_TOKENS:
        client = InferenceClient(token=token, timeout=60)
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL_ID,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": instruction},
                    ],
                }],
                max_tokens=200,
            )
            raw = completion.choices[0].message.content
            parsed = _extract_json(raw)
            return _finalize_result(parsed)
        except Exception as e:  # noqa: BLE001 - try next token / fail gracefully
            last_error = str(e)
            continue

    return _error_result(f"all tokens failed, last error: {last_error}")


def _finalize_result(parsed: dict) -> dict:
    required = ["face_visible", "facing_camera", "product_visible",
                "product_plausible_placement", "overall_realism_score"]
    for key in required:
        parsed.setdefault(key, False if key != "overall_realism_score" else 0)

    # Pass criteria: all boolean checks true AND realism score >= 6/10.
    # Tune this threshold based on what you observe in practice.
    parsed["pass"] = bool(
        parsed["face_visible"] and parsed["facing_camera"]
        and parsed["product_visible"] and parsed["product_plausible_placement"]
        and float(parsed.get("overall_realism_score", 0)) >= 6
    )
    parsed["error"] = None
    return parsed


def _error_result(message: str) -> dict:
    return {
        "face_visible": False, "facing_camera": False, "product_visible": False,
        "product_plausible_placement": False, "overall_realism_score": 0,
        "notes": "", "pass": False, "error": message,
    }


# ---------------------------------------------------------------------------
# CLI - batch score every image in a folder, write a CSV log automatically
# ---------------------------------------------------------------------------

def score_folder(folder: str, product_category: str = "product",
                  output_csv: str = None) -> list:
    if output_csv is None:
        output_csv = os.path.join(folder, "visual_test_results.csv")

    images = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not images:
        print(f"No images found in {folder}")
        return []

    results = []
    for name in images:
        path = os.path.join(folder, name)
        print(f"Scoring {name}...")
        r = score_visual_integration(path, product_category)
        r["image"] = name
        r["timestamp"] = datetime.now().isoformat()
        results.append(r)
        status = "PASS" if r["pass"] else ("ERROR" if r["error"] else "FAIL")
        print(f"  {status} — realism {r['overall_realism_score']}/10"
              + (f" — {r['error']}" if r["error"] else ""))

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["image", "pass", "face_visible", "facing_camera",
                      "product_visible", "product_plausible_placement",
                      "overall_realism_score", "notes", "error", "timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved → {output_csv}")

    n = len(results)
    passed = sum(1 for r in results if r["pass"])
    print(f"\n{passed}/{n} images passed automated visual integration check")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tests/visual_checker.py <folder_of_images> [product_category]")
        sys.exit(1)
    folder = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else "product"
    score_folder(folder, category)
