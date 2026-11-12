"""
eval/llm_judge.py — LLaVA-Guided Demographic Prompt Correction

Extends CPDC (modules/cpdc.py) with LLaVA-generated correction suggestions.
CPDC decides WHETHER and HOW MUCH to correct (graduated levels).
This module decides WHAT to correct — the specific prompt edits.

Integration point: generator/flux_pipeline.py
Called after each failed CPDC check, before the next generation attempt.

Regional Indian demographic focus:
  South Indian    — Dravidian features, dark brown complexion
  North Indian    — Indo-Aryan features, wheatish complexion
  East Indian     — Bengali features, medium brown complexion
  Northeast Indian — Mongoloid features, light complexion

Requires:
  - Ollama running locally: ollama serve
  - LLaVA pulled:           ollama pull llava

Standalone test:
  python eval/llm_judge.py path/to/image.png south_indian
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Regional descriptor reference — used to validate LLaVA suggestions
# and as fallback corrections when LLaVA output can't be parsed.
# ---------------------------------------------------------------------------

REGIONAL_DESCRIPTORS = {
    "south_indian": {
        "features":   "Dravidian facial features, broad nose, dark eyes",
        "complexion": "dark brown complexion, deep warm undertone",
        "clothing":   "silk saree, gold temple jewelry",
        "keywords":   ["dravidian", "south indian", "tamil", "telugu", "dark complexion"],
    },
    "north_indian": {
        "features":   "Indo-Aryan facial features, sharp nose, almond eyes",
        "complexion": "wheatish complexion, warm golden undertone",
        "clothing":   "salwar kameez, traditional dupatta",
        "keywords":   ["indo-aryan", "north indian", "punjabi", "hindi", "wheatish"],
    },
    "east_indian": {
        "features":   "Bengali facial features, soft eyes, medium build",
        "complexion": "medium brown complexion, neutral undertone",
        "clothing":   "cotton saree, simple jewelry",
        "keywords":   ["bengali", "east indian", "odia", "medium complexion"],
    },
    "northeast_indian": {
        "features":   "Mongoloid facial features, epicanthic fold, high cheekbones",
        "complexion": "light complexion, cool undertone",
        "clothing":   "mekhela chador, traditional northeast attire",
        "keywords":   ["mongoloid", "northeast indian", "assamese", "meitei", "light complexion"],
    },
}


# ---------------------------------------------------------------------------
# Core judge — evaluates image and returns correction suggestions
# ---------------------------------------------------------------------------

def evaluate_and_suggest(
    image_path: str,
    target_region: str,
    current_prompt: str,
    correction_level: int = 2,
) -> dict:
    """
    Evaluate a generated image for regional Indian demographic fidelity
    and suggest specific prompt corrections.

    Args:
        image_path:       Path to the generated image.
        target_region:    One of: south_indian, north_indian,
                          east_indian, northeast_indian.
        current_prompt:   The FLUX prompt that produced this image.
        correction_level: CPDC correction level (1-4). Higher = more
                          aggressive suggestions injected into prompt.

    Returns:
        {
            "perceived_region":    str,   # what LLaVA thinks the region is
            "fidelity_score":      float, # 0.0-1.0
            "issues":              list,  # what's wrong
            "prompt_corrections":  list,  # phrases to append to prompt
            "corrected_prompt":    str,   # full corrected prompt ready to use
            "pass":                bool,  # True if fidelity_score >= threshold
            "confidence":          float, # LLaVA's self-reported confidence
        }
    """
    if target_region not in REGIONAL_DESCRIPTORS:
        raise ValueError(
            f"Unknown region '{target_region}'. "
            f"Valid: {list(REGIONAL_DESCRIPTORS)}"
        )

    if not os.path.exists(image_path):
        return _fallback_result(target_region, current_prompt, correction_level,
                                reason="image not found")

    descriptors = REGIONAL_DESCRIPTORS[target_region]
    region_label = target_region.replace("_", " ").title()

    judge_prompt = (
        f"Look at this AI-generated portrait image carefully.\n"
        f"Target demographic: {region_label} Indian woman\n\n"
        f"Evaluate the image on these criteria:\n"
        f"1. Does the person's skin tone match {region_label}? "
        f"Expected: {descriptors['complexion']}\n"
        f"2. Do the facial features match {region_label}? "
        f"Expected: {descriptors['features']}\n"
        f"3. What Indian region does this person most look like?\n\n"
        f"Respond ONLY in valid JSON, no extra text:\n"
        f'{{'
        f'"perceived_region": "<region name>", '
        f'"skin_tone_correct": <true/false>, '
        f'"features_correct": <true/false>, '
        f'"fidelity_score": <0.0-1.0>, '
        f'"confidence": <0.0-1.0>, '
        f'"issues": ["<issue1>", "<issue2>"], '
        f'"corrections_needed": ["<specific visual fix1>", "<specific visual fix2>"]'
        f'}}'
    )

    try:
        import ollama
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        response = ollama.chat(
            model="llava",
            messages=[{
                "role":    "user",
                "content": judge_prompt,
                "images":  [img_bytes],
            }]
        )
        raw = response["message"]["content"]
        parsed = _parse_json(raw)

    except ImportError:
        return _fallback_result(target_region, current_prompt, correction_level,
                                reason="ollama not installed")
    except Exception as e:
        return _fallback_result(target_region, current_prompt, correction_level,
                                reason=str(e))

    # Build prompt corrections from LLaVA output + regional descriptor fallbacks
    prompt_corrections = _build_corrections(
        parsed, descriptors, correction_level
    )

    fidelity = float(parsed.get("fidelity_score", 0.5))
    passes = fidelity >= _pass_threshold(correction_level)

    corrected_prompt = current_prompt
    if not passes and prompt_corrections:
        correction_str = ", ".join(prompt_corrections)
        corrected_prompt = f"{current_prompt}, {correction_str}"

    return {
        "perceived_region":   parsed.get("perceived_region", "unknown"),
        "fidelity_score":     fidelity,
        "confidence":         float(parsed.get("confidence", 0.5)),
        "issues":             parsed.get("issues", []),
        "prompt_corrections": prompt_corrections,
        "corrected_prompt":   corrected_prompt,
        "pass":               passes,
    }


# ---------------------------------------------------------------------------
# Correction builder — converts LLaVA issues into FLUX prompt tokens
# ---------------------------------------------------------------------------

def _build_corrections(
    parsed: dict,
    descriptors: dict,
    correction_level: int,
) -> list[str]:
    """
    Build a list of prompt correction tokens based on LLaVA's output
    and the correction level from CPDC.

    Level 1: add complexion descriptor only (mild)
    Level 2: add complexion + features
    Level 3: add complexion + features + clothing
    Level 4: add all descriptors + explicit regional label (max correction)
    """
    corrections = []

    # Always add what CPDC says is wrong based on level
    if correction_level >= 1:
        corrections.append(descriptors["complexion"])

    if correction_level >= 2:
        corrections.append(descriptors["features"])

    if correction_level >= 3:
        corrections.append(descriptors["clothing"])

    if correction_level >= 4:
        # Maximum correction — explicit regional grounding
        region_label = list(descriptors.values())[0]  # not used, explicit below
        corrections.append("authentic regional Indian appearance")
        corrections.append("photorealistic portrait")

    # Add any specific fixes LLaVA identified, up to 2
    llava_fixes = parsed.get("corrections_needed", [])
    for fix in llava_fixes[:2]:
        if fix and len(fix) < 60:  # sanity check — ignore long hallucinated text
            corrections.append(fix)

    return corrections


# ---------------------------------------------------------------------------
# Threshold — what fidelity score counts as passing, per correction level
# ---------------------------------------------------------------------------

def _pass_threshold(correction_level: int) -> float:
    """
    Higher correction levels mean we've already tried and failed —
    lower the bar slightly to avoid infinite loops.
    """
    thresholds = {1: 0.70, 2: 0.65, 3: 0.60, 4: 0.55}
    return thresholds.get(correction_level, 0.65)


# ---------------------------------------------------------------------------
# Fallback — when LLaVA is unavailable, use rule-based corrections only
# ---------------------------------------------------------------------------

def _fallback_result(
    target_region: str,
    current_prompt: str,
    correction_level: int,
    reason: str = "",
) -> dict:
    """
    Rule-based fallback when LLaVA is unavailable.
    Uses REGIONAL_DESCRIPTORS directly — no VLM needed.
    This is equivalent to the old CDVR behavior.
    """
    descriptors = REGIONAL_DESCRIPTORS[target_region]
    corrections = _build_corrections({}, descriptors, correction_level)
    corrected_prompt = current_prompt
    if corrections:
        corrected_prompt = f"{current_prompt}, {', '.join(corrections)}"

    return {
        "perceived_region":   "unknown (llava unavailable)",
        "fidelity_score":     0.0,
        "confidence":         0.0,
        "issues":             [f"LLaVA unavailable: {reason}"],
        "prompt_corrections": corrections,
        "corrected_prompt":   corrected_prompt,
        "pass":               False,
    }


# ---------------------------------------------------------------------------
# JSON parser — robust against LLaVA markdown wrapping
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Batch evaluation — re-runs scoring on regional Indian grid
# Extends run_llava_scoring.py to cover South/North/East/Northeast Indian
# ---------------------------------------------------------------------------

REGIONAL_INDIAN_PROFILES = [
    {"ethnicity": "South Indian",     "body_type": "slim",   "age": "30s", "region": "south_indian"},
    {"ethnicity": "South Indian",     "body_type": "medium", "age": "20s", "region": "south_indian"},
    {"ethnicity": "North Indian",     "body_type": "slim",   "age": "30s", "region": "north_indian"},
    {"ethnicity": "North Indian",     "body_type": "medium", "age": "20s", "region": "north_indian"},
    {"ethnicity": "East Indian",      "body_type": "slim",   "age": "30s", "region": "east_indian"},
    {"ethnicity": "East Indian",      "body_type": "medium", "age": "20s", "region": "east_indian"},
    {"ethnicity": "Northeast Indian", "body_type": "slim",   "age": "30s", "region": "northeast_indian"},
    {"ethnicity": "Northeast Indian", "body_type": "medium", "age": "20s", "region": "northeast_indian"},
]


def run_regional_evaluation(images_dir: str, output_csv: str = None) -> list:
    """
    Score a folder of regionally labeled Indian ad images.
    Images should be named: South-Indian_slim_30s_seed42.png etc.

    This extends the existing run_llava_scoring.py evaluation to
    regional Indian demographics for the paper's Table X.
    """
    import csv
    from datetime import datetime

    if output_csv is None:
        output_csv = os.path.join(images_dir, "regional_indian_scores.csv")

    images = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    print(f"[LLM Judge] Scoring {len(images)} regional Indian images...")

    results = []
    for i, fname in enumerate(images):
        image_path = os.path.join(images_dir, fname)

        # Parse region from filename: South-Indian_slim_30s_seed42.png
        region = _parse_region_from_filename(fname)
        if not region:
            print(f"  [{i+1}] SKIP — can't parse region from {fname}")
            continue

        print(f"  [{i+1}/{len(images)}] {fname} -> target: {region}...", end=" ")

        result = evaluate_and_suggest(
            image_path=image_path,
            target_region=region,
            current_prompt="",   # not needed for eval-only mode
            correction_level=1,
        )

        row = {
            "image":            fname,
            "target_region":    region,
            "perceived_region": result["perceived_region"],
            "fidelity_score":   result["fidelity_score"],
            "confidence":       result["confidence"],
            "pass":             result["pass"],
            "issues":           "; ".join(result["issues"]),
            "timestamp":        datetime.now().isoformat(),
        }
        results.append(row)
        print(f"fidelity={result['fidelity_score']:.2f} pass={result['pass']}")

    if results:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[LLM Judge] Saved -> {output_csv}")

        # Print summary per region
        print("\n=== Regional Fidelity Summary ===")
        for region in REGIONAL_DESCRIPTORS:
            region_rows = [r for r in results if r["target_region"] == region]
            if region_rows:
                avg = sum(float(r["fidelity_score"]) for r in region_rows) / len(region_rows)
                pass_rate = sum(1 for r in region_rows if r["pass"] in [True, "True"]) / len(region_rows)
                print(f"  {region:20s}: avg_fidelity={avg:.2f}  pass_rate={pass_rate:.0%}")

    return results


def _parse_region_from_filename(fname: str) -> Optional[str]:
    """South-Indian_slim_30s_seed42.png -> south_indian"""
    fname_lower = fname.lower().replace("-", "_")
    for region in REGIONAL_DESCRIPTORS:
        if region in fname_lower:
            return region
    return None


# ---------------------------------------------------------------------------
# CLI — test on a single image
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python eval/llm_judge.py <image_path> <region>")
        print("       region: south_indian | north_indian | east_indian | northeast_indian")
        print("\nExample:")
        print("  python eval/llm_judge.py eval/results/test_south.png south_indian")
        sys.exit(1)

    image_path   = sys.argv[1]
    target_region = sys.argv[2]
    test_prompt  = "a portrait photo of an Indian woman in traditional attire"

    print(f"\n[LLM Judge] Testing on: {image_path}")
    print(f"[LLM Judge] Target region: {target_region}\n")

    result = evaluate_and_suggest(
        image_path=image_path,
        target_region=target_region,
        current_prompt=test_prompt,
        correction_level=2,
    )

    print("=== Result ===")
    print(f"Perceived region:  {result['perceived_region']}")
    print(f"Fidelity score:    {result['fidelity_score']:.2f}")
    print(f"Confidence:        {result['confidence']:.2f}")
    print(f"Pass:              {result['pass']}")
    print(f"Issues:            {result['issues']}")
    print(f"Corrections added: {result['prompt_corrections']}")
    print(f"\nCorrected prompt:\n  {result['corrected_prompt']}")
