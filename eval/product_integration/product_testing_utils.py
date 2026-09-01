"""
Shared utilities for the Product Integration evaluation suite.

All five test files use this module for:
- locating project paths
- validating/opening product and generated images
- calling the same Gemma-3-27B vision evaluator used by the project's
  product-description module
- parsing structured scores
- printing consistent PASS/FAIL results

This file is a helper; it is not a test case.
"""

from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import InferenceClient

load_dotenv()

MODEL_ID = "google/gemma-3-27b-it"
HF_TOKENS = [
    token for token in (
        os.getenv("HF_TOKEN_1"),
        os.getenv("HF_TOKEN_2"),
        os.getenv("HF_TOKEN_3"),
    )
    if token
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "eval" / "product_integration" / "results"


def validate_image(path: str | Path) -> tuple[bool, str]:
    """Return (True, message) when path is a readable supported image."""
    p = Path(path)
    if not p.exists():
        return False, f"file not found: {p}"
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return False, f"unsupported image format: {p.suffix}"
    try:
        with Image.open(p) as image:
            image.verify()
        with Image.open(p) as image:
            size = image.size
        return True, f"{p.name} ({size[0]}x{size[1]})"
    except Exception as exc:
        return False, f"unreadable image: {exc}"


def image_to_data_url(path: str | Path) -> str:
    """Convert an image file to a JPEG data URL for the HF vision API."""
    with Image.open(path) as image:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json(text: str) -> dict[str, Any]:
    """Robustly extract a JSON object from a model response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse evaluator JSON: {text[:500]}")


def _clamp_score(value: Any) -> int:
    try:
        return max(1, min(5, int(round(float(value)))))
    except Exception:
        return 1


def vision_evaluate(
    original_product: str | Path,
    generated_image: str | Path,
    criteria: list[tuple[str, str]],
    context: str = "",
) -> dict[str, Any]:
    """
    Compare an original product image with a generated advertisement.

    criteria is a list of (key, question) pairs. Each criterion receives
    a 1-5 score and a short reason.
    """
    ok1, msg1 = validate_image(original_product)
    ok2, msg2 = validate_image(generated_image)
    if not ok1:
        raise FileNotFoundError(msg1)
    if not ok2:
        raise FileNotFoundError(msg2)
    if not HF_TOKENS:
        raise RuntimeError(
            "No HF_TOKEN_1/2/3 found in .env. "
            "The visual tests require the project's Hugging Face token setup."
        )

    criterion_text = "\n".join(
        f'{i}. "{key}": {question}'
        for i, (key, question) in enumerate(criteria, 1)
    )
    keys = [key for key, _ in criteria]

    schema_scores = ", ".join(f'"{key}": X' for key in keys)
    schema_reasons = ", ".join(f'"{key}_reason": "short reason"' for key in keys)

    prompt = f"""
You are evaluating PRODUCT INTEGRATION in an AI-generated advertisement.

IMAGE 1 is the ORIGINAL PRODUCT uploaded by the user.
IMAGE 2 is the GENERATED ADVERTISEMENT.

{context}

Compare IMAGE 1 with the product as it appears in IMAGE 2.
Judge ONLY the product and its integration. Do not judge the person's
demographic attributes, beauty, or the overall advertisement design.

Score every criterion from 1 to 5:
5 = excellent / clearly correct
4 = good / minor imperfection
3 = acceptable / noticeable imperfection
2 = poor
1 = failed / absent

Criteria:
{criterion_text}

Return JSON ONLY in this exact structure:
{{
  "scores": {{{schema_scores}}},
  "reasons": {{{schema_reasons}}},
  "overall_reason": "one short overall observation"
}}
"""

    content = [
        {"type": "image_url", "image_url": {"url": image_to_data_url(original_product)}},
        {"type": "image_url", "image_url": {"url": image_to_data_url(generated_image)}},
        {"type": "text", "text": prompt},
    ]

    errors = []
    for token in HF_TOKENS:
        try:
            client = InferenceClient(token=token, timeout=90)
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": content}],
                max_tokens=700,
            )
            raw = response.choices[0].message.content.strip()
            parsed = _extract_json(raw)

            raw_scores = parsed.get("scores", {})
            scores = {key: _clamp_score(raw_scores.get(key, 1)) for key in keys}
            reasons = {
                key: str(parsed.get("reasons", {}).get(f"{key}_reason", ""))
                for key in keys
            }

            return {
                "scores": scores,
                "reasons": reasons,
                "overall_reason": str(parsed.get("overall_reason", "")),
            }
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    raise RuntimeError("All Hugging Face evaluation attempts failed: " + " | ".join(errors))


def run_single_criterion_test(
    test_id: str,
    title: str,
    product_path: str,
    output_path: str,
    category: str,
    criterion_key: str,
    criterion_question: str,
) -> dict[str, Any]:
    """Run one visual criterion and print a screenshot-friendly result."""
    print("\n" + "=" * 72)
    print(f"{test_id}  {title}")
    print("=" * 72)
    print(f"Product : {product_path}")
    print(f"Output  : {output_path}")
    print(f"Category: {category}")

    result = vision_evaluate(
        product_path,
        output_path,
        [(criterion_key, criterion_question)],
        context=f"The expected product category is: {category}.",
    )

    score = result["scores"][criterion_key]
    status = "PASS" if score >= 3 else "FAIL"
    reason = result["reasons"][criterion_key]

    print(f"\n{test_id} RESULT: {status}")
    print(f"{criterion_key}: {score}/5")
    print(f"Reason: {reason}")
    return {
        "test_id": test_id,
        "title": title,
        "category": category,
        "score": score,
        "status": status,
        "reason": reason,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def run_multi_criterion_test(
    test_id: str,
    title: str,
    product_path: str,
    output_path: str,
    category: str,
    criteria: list[tuple[str, str]],
) -> dict[str, Any]:
    """Run a group of product-integration criteria."""
    print("\n" + "=" * 72)
    print(f"{test_id}  {title}")
    print("=" * 72)
    print(f"Product : {product_path}")
    print(f"Output  : {output_path}")
    print(f"Category: {category}")

    result = vision_evaluate(
        product_path,
        output_path,
        criteria,
        context=f"The expected product category is: {category}.",
    )

    scores = result["scores"]
    passed = sum(score >= 3 for score in scores.values())
    status = "PASS" if passed == len(scores) else "FAIL"

    print()
    for key, _ in criteria:
        print(f"{key}: {scores[key]}/5 — {result['reasons'][key]}")
    print(f"\n{test_id} RESULT: {status} ({passed}/{len(scores)} criteria passed)")
    print(f"Overall: {result['overall_reason']}")

    return {
        "test_id": test_id,
        "title": title,
        "category": category,
        "scores": scores,
        "reasons": result["reasons"],
        "status": status,
        "passed_criteria": passed,
        "total_criteria": len(scores),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def print_input_test(
    product_path: str,
    category: str,
) -> dict[str, Any]:
    """PI-01: validate the original product file used in the live upload."""
    print("\n" + "=" * 72)
    print("PI-01  Dynamic Product Upload / Input Validation")
    print("=" * 72)
    print(f"Product : {product_path}")
    print(f"Category: {category}")

    ok, message = validate_image(product_path)
    status = "PASS" if ok else "FAIL"
    print(f"\nPI-01 RESULT: {status}")
    print(f"Input check: {message}")

    return {
        "test_id": "PI-01",
        "title": "Dynamic Product Upload / Input Validation",
        "category": category,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# def save_result(test_result: dict[str, Any]) -> Path:
#     """Save one result as JSON under eval/product_integration/results."""
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#     stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     path = RESULTS_DIR / f"{test_result['test_id']}_{stamp}.json"
#     path.write_text(json.dumps(test_result, indent=2), encoding="utf-8")
#     print(f"Result saved: {path}")
#     return path

def save_result(test_result: dict[str, Any]) -> Path:
    """Save one result as JSON under eval/product_integration/results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_test_id = test_result["test_id"].replace("/", "_")

    path = RESULTS_DIR / f"{safe_test_id}_{stamp}.json"
    path.write_text(json.dumps(test_result, indent=2), encoding="utf-8")

    print(f"Result saved: {path}")
    return path


def save_batch_result(test_id: str, results: list[dict[str, Any]]) -> Path:
    """Save a robustness batch result."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{test_id}_{stamp}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBatch result saved: {path}")
    return path
