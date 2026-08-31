"""
eval/fidelity_scorer.py - Demographic Fidelity Check (DFC)

Runs LLaVA offline via Ollama to score how well a generated advertisement
image matches its intended demographic profile.

Used for:
  - Offline evaluation batch (generates Table 1 in the paper)
  - Wired into the live pipeline as a CDVR trigger
  - Feeds age_confidence into CPDC (modules/cpdc.py) as the age-axis
    error signal - CLIP's age classifier is unreliable (36% accuracy,
    see EVALUATION_RESULTS.md Table 3), so CPDC uses LLaVA's own
    self-reported confidence for that axis instead.

Requires:
  - Ollama running locally: ollama serve
  - LLaVA pulled:           ollama pull llava
"""

import json
import os
import re
import csv
from datetime import datetime


# ---------------------------------------------------------------------------
# Single image scorer
# ---------------------------------------------------------------------------

def score_fidelity(image_path: str, demographic_profile: dict) -> dict:
    """
    Score a generated image against its intended demographic profile
    using LLaVA via Ollama.

    Args:
        image_path:          Absolute or relative path to the image file.
        demographic_profile: Dict with keys: ethnicity, body_type, age.

    Returns:
        Dict with keys: age_score, age_confidence, gender_score,
        gender_confidence, ethnicity_score, ethnicity_confidence,
        overall, notes. Scores are 0-10, confidences are 0.0-1.0.
        Returns a zero-score dict on parse failure rather than raising.
    """
    try:
        import ollama
    except ImportError:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")

    if not os.path.exists(image_path):
        return _zero_scores(f"image not found: {image_path}")

    prompt = (
        f"Look at this advertisement image carefully.\n"
        f"The intended demographic profile is: {json.dumps(demographic_profile)}\n\n"
        f"Score on each dimension from 0 to 10, AND give your confidence (0.0-1.0) "
        f"that your assessment is correct:\n"
        f"1. Age match - does the model appear to be in their {demographic_profile.get('age','?')}?\n"
        f"2. Gender match - is this a woman?\n"
        f"3. Ethnicity/skin tone match - does the appearance match {demographic_profile.get('ethnicity','?')}?\n"
        f"4. Overall demographic fidelity - how well does the image match the full profile?\n\n"
        f"Respond in JSON only, no extra text:\n"
        f'{{"age_score": X, "age_confidence": X, '
        f'"gender_score": X, "gender_confidence": X, '
        f'"ethnicity_score": X, "ethnicity_confidence": X, '
        f'"overall": X, "notes": "..."}}'
    )

    try:
        response = ollama.chat(
            model="llava",
            messages=[{
                "role":    "user",
                "content": prompt,
                "images":  [image_path],
            }]
        )
        raw = response["message"]["content"]
        return _parse_json(raw)

    except Exception as e:
        return _zero_scores(f"llava error: {e}")


def _parse_json(raw: str) -> dict:
    """
    Robustly extract JSON from LLaVA's response.
    LLaVA sometimes wraps JSON in markdown or adds preamble text.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object from text
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return _zero_scores(f"could not parse response: {raw[:100]}")


def _zero_scores(notes: str) -> dict:
    return {
        "age_score":            0,
        "age_confidence":       0.0,
        "gender_score":         0,
        "gender_confidence":    0.0,
        "ethnicity_score":      0,
        "ethnicity_confidence": 0.0,
        "overall":              0,
        "notes":                notes,
    }


# ---------------------------------------------------------------------------
# Batch scorer - generates Table 1 for the paper
# ---------------------------------------------------------------------------

def score_batch(image_folder: str, profiles: list,
                output_csv: str = None) -> list:
    """
    Score all images in a folder against their corresponding profiles.

    Expects images named: output_0.png, output_1.png, ... (matching profile indices).
    Saves results to a CSV file for direct use as paper Table 1.

    Args:
        image_folder: Folder containing the generated images.
        profiles:     List of demographic profile dicts, one per image.
        output_csv:   Path to save the CSV results. Defaults to
                      {image_folder}/fidelity_results.csv.

    Returns:
        List of result dicts (one per image).
    """
    if output_csv is None:
        output_csv = os.path.join(image_folder, "fidelity_results.csv")

    results = []
    total   = len(profiles)

    print(f"[DFC] Scoring {total} images...")

    for i, profile in enumerate(profiles):
        # Support both output_0.png and output_seed*.png naming
        candidates = [
            os.path.join(image_folder, f"output_{i}.png"),
            os.path.join(image_folder, f"eval_{i}.png"),
        ]
        image_path = next((p for p in candidates if os.path.exists(p)), None)

        if image_path is None:
            print(f"  [{i+1}/{total}] SKIP - no image found for profile {i}")
            continue

        print(f"  [{i+1}/{total}] Scoring {os.path.basename(image_path)}...")
        scores = score_fidelity(image_path, profile)

        result = {
            "index":                i,
            "image":                os.path.basename(image_path),
            "ethnicity":            profile.get("ethnicity", ""),
            "body_type":            profile.get("body_type", ""),
            "age":                  profile.get("age", ""),
            "age_score":            scores["age_score"],
            "age_confidence":       scores.get("age_confidence", 0.0),
            "gender_score":         scores["gender_score"],
            "gender_confidence":    scores.get("gender_confidence", 0.0),
            "ethnicity_score":      scores["ethnicity_score"],
            "ethnicity_confidence": scores.get("ethnicity_confidence", 0.0),
            "overall":              scores["overall"],
            "notes":                scores.get("notes", ""),
            "timestamp":            datetime.now().isoformat(),
        }
        results.append(result)
        print(f"         overall={scores['overall']}/10  "
              f"ethnicity={scores['ethnicity_score']}/10  "
              f"age={scores['age_score']}/10")

    if results:
        _save_csv(results, output_csv)
        _print_summary(results)

    return results


def _save_csv(results: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[DFC] Results saved -> {path}")


def _print_summary(results: list) -> None:
    n = len(results)
    avg = lambda key: round(sum(r[key] for r in results) / n, 2)
    print(f"\n[DFC] Summary ({n} images):")
    print(f"  Avg overall fidelity:   {avg('overall')}/10")
    print(f"  Avg ethnicity match:    {avg('ethnicity_score')}/10")
    print(f"  Avg age match:          {avg('age_score')}/10")
    print(f"  Avg gender match:       {avg('gender_score')}/10")


# ---------------------------------------------------------------------------
# CLI - run as script for batch evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Usage:
        python eval/fidelity_scorer.py

    Edit the profiles list below to match your evaluation batch.
    Images should be named output_0.png, output_1.png, etc. in eval/images/.
    """
    eval_profiles = [
        {"ethnicity": "South Asian",      "body_type": "slim",      "age": "20s"},
        {"ethnicity": "South Asian",      "body_type": "medium",    "age": "30s"},
        {"ethnicity": "South Asian",      "body_type": "plus-size", "age": "40s"},
        {"ethnicity": "East Asian",       "body_type": "slim",      "age": "20s"},
        {"ethnicity": "East Asian",       "body_type": "medium",    "age": "30s"},
        {"ethnicity": "East Asian",       "body_type": "plus-size", "age": "40s"},
        {"ethnicity": "African American", "body_type": "slim",      "age": "20s"},
        {"ethnicity": "African American", "body_type": "medium",    "age": "30s"},
        {"ethnicity": "African American", "body_type": "plus-size", "age": "40s"},
        # Add more as needed - 30 total recommended for the paper
    ]

    score_batch(
        image_folder="eval/images",
        profiles=eval_profiles,
        output_csv="eval/fidelity_results.csv",
    )
