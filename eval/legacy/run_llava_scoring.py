"""
run_llava_scoring.py — Score generated images with LLaVA via Ollama
====================================================================
Reads each generated image, asks LLaVA to identify the person's
ethnicity, age, and gender, then compares against the intended profile.

Usage:
    cd Capstone-Customised-Advertisement-Generation-Using-AI
    python eval/run_llava_scoring.py
"""

import os
import json
import csv
import base64
import re
import time

try:
    import ollama
    USE_OLLAMA_LIB = True
except ImportError:
    import urllib.request
    USE_OLLAMA_LIB = False
    print("ollama package not found, using raw HTTP requests")


IMAGES_DIR = os.path.join("eval", "results", "generated_images")
OUTPUT_CSV = os.path.join("eval", "results", "dfc_scores.csv")
OUTPUT_JSON = os.path.join("eval", "results", "dfc_summary.json")


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_llava(image_path, question):
    """Send an image + question to LLaVA via Ollama and get a text response."""
    if USE_OLLAMA_LIB:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        response = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": question,
                "images": [img_bytes],
            }],
        )
        return response["message"]["content"]
    else:
        # Fallback: raw HTTP to Ollama API
        b64 = image_to_base64(image_path)
        payload = json.dumps({
            "model": "llava",
            "messages": [{
                "role": "user",
                "content": question,
                "images": [b64],
            }],
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]


def parse_profile_from_filename(filename):
    """Extract target demographics from filename like South-Asian_slim_30s_seed42.png"""
    name = filename.replace(".png", "")
    parts = name.rsplit("_seed", 1)
    seed = int(parts[1]) if len(parts) > 1 else 0
    demo_parts = parts[0].rsplit("_", 2)

    if len(demo_parts) == 3:
        ethnicity = demo_parts[0].replace("-", " ")
        body_type = demo_parts[1]
        age = demo_parts[2]
    else:
        ethnicity, body_type, age = "Unknown", "Unknown", "Unknown"

    return {
        "ethnicity": ethnicity,
        "body_type": body_type,
        "age": age,
        "gender": "Female",
        "seed": seed,
    }


def score_one_image(image_path, profile):
    """Score a single image against its intended demographic profile."""
    question = (
        "Look at this photograph carefully. Answer these three questions about "
        "the person shown. Be specific and concise.\n"
        "1. What is their likely ethnicity or racial background? "
        "(e.g., South Asian, East Asian, African American, White, Latino)\n"
        "2. What is their approximate age range? "
        "(e.g., 20s, 30s, 40s, 50s)\n"
        "3. What is their gender? (Male or Female)\n"
        "Format your answer exactly as:\n"
        "Ethnicity: <answer>\n"
        "Age: <answer>\n"
        "Gender: <answer>"
    )

    try:
        response = ask_llava(image_path, question)
    except Exception as e:
        return {
            "llava_response": str(e),
            "pred_ethnicity": "error",
            "pred_age": "error",
            "pred_gender": "error",
            "ethnicity_match": False,
            "age_match": False,
            "gender_match": False,
            "overall_dfc": 0.0,
        }

    # Parse response
    resp_lower = response.lower()

    # Ethnicity matching
    target_eth = profile["ethnicity"].lower()
    eth_keywords = {
        "south asian": ["south asian", "indian", "desi", "subcontinental"],
        "east asian": ["east asian", "chinese", "japanese", "korean", "asian"],
        "african american": ["african american", "black", "african", "dark-skinned"],
    }

    pred_ethnicity = "Unknown"
    ethnicity_match = False
    for eth_name, keywords in eth_keywords.items():
        if any(kw in resp_lower for kw in keywords):
            pred_ethnicity = eth_name
            if target_eth in eth_name or eth_name in target_eth:
                ethnicity_match = True
            break

    # Special case: "Asian" without qualifier
    if pred_ethnicity == "Unknown" and "asian" in resp_lower:
        pred_ethnicity = "Asian (unspecified)"
        if "asian" in target_eth:
            ethnicity_match = True

    # Age matching
    target_age = profile["age"].replace("s", "")  # "30s" -> "30"
    pred_age = "Unknown"
    age_match = False

    age_patterns = [
        r'(\d{2})s', r'(\d{2})\s*-\s*\d{2}', r'(\d{2})\s*to\s*\d{2}',
        r'mid[- ]?(\d{2})', r'late[- ]?(\d{2})', r'early[- ]?(\d{2})',
        r'(\d{2})\s*years?', r'around\s*(\d{2})',
    ]
    for pattern in age_patterns:
        m = re.search(pattern, resp_lower)
        if m:
            detected_age = int(m.group(1))
            pred_age = f"{(detected_age // 10) * 10}s"
            target_decade = int(target_age)
            if abs(detected_age - target_decade) <= 10:
                age_match = True
            break

    # Gender matching
    pred_gender = "Unknown"
    gender_match = False
    if "female" in resp_lower or "woman" in resp_lower:
        pred_gender = "Female"
        gender_match = (profile["gender"].lower() == "female")
    elif "male" in resp_lower or " man " in resp_lower:
        pred_gender = "Male"
        gender_match = (profile["gender"].lower() == "male")

    overall = sum([ethnicity_match, age_match, gender_match]) / 3.0

    return {
        "llava_response": response[:500],
        "pred_ethnicity": pred_ethnicity,
        "pred_age": pred_age,
        "pred_gender": pred_gender,
        "ethnicity_match": ethnicity_match,
        "age_match": age_match,
        "gender_match": gender_match,
        "overall_dfc": round(overall, 3),
    }


def main():
    if not os.path.exists(IMAGES_DIR):
        print(f"ERROR: {IMAGES_DIR} not found")
        print("Make sure generated images are in eval/results/generated_images/")
        return

    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])
    print(f"Found {len(images)} images to score\n")

    if not images:
        print("No images found!")
        return

    # Check if partial results exist
    existing = {}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["image"]] = row
        print(f"Found {len(existing)} existing scores, will skip those\n")

    results = list(existing.values())

    for i, filename in enumerate(images):
        if filename in existing:
            print(f"  [{i+1}/{len(images)}] SKIP (cached): {filename}")
            continue

        image_path = os.path.join(IMAGES_DIR, filename)
        profile = parse_profile_from_filename(filename)

        print(f"  [{i+1}/{len(images)}] {filename}...", end=" ")

        scores = score_one_image(image_path, profile)

        row = {
            "image": filename,
            "seed": profile["seed"],
            "target_ethnicity": profile["ethnicity"],
            "target_body_type": profile["body_type"],
            "target_age": profile["age"],
            "target_gender": profile["gender"],
            **scores,
        }
        results.append(row)

        eth_mark = "Y" if scores["ethnicity_match"] else "N"
        age_mark = "Y" if scores["age_match"] else "N"
        gen_mark = "Y" if scores["gender_match"] else "N"
        print(f"Race:{eth_mark} Age:{age_mark} Gender:{gen_mark} DFC:{scores['overall_dfc']:.0%}")

        # Save progress every 5 images
        if (i + 1) % 5 == 0:
            _save_csv(results)

        time.sleep(0.5)  # gentle pacing for Ollama

    # Final save
    _save_csv(results)

    # Summary
    matched = [r for r in results if "overall_dfc" in r and r["overall_dfc"] != "error"]
    dfc_values = [float(r["overall_dfc"]) for r in matched if r["overall_dfc"]]

    if dfc_values:
        eth_acc = sum(1 for r in matched if r.get("ethnicity_match") in [True, "True"]) / len(matched)
        age_acc = sum(1 for r in matched if r.get("age_match") in [True, "True"]) / len(matched)
        gen_acc = sum(1 for r in matched if r.get("gender_match") in [True, "True"]) / len(matched)
        mean_dfc = sum(dfc_values) / len(dfc_values)

        summary = {
            "scorer": "LLaVA (via Ollama)",
            "total_images": len(results),
            "scored": len(matched),
            "mean_dfc": round(mean_dfc, 3),
            "ethnicity_accuracy": round(eth_acc, 3),
            "age_accuracy": round(age_acc, 3),
            "gender_accuracy": round(gen_acc, 3),
        }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"LLaVA DFC SCORING COMPLETE")
        print(f"{'='*60}")
        print(f"Scored: {len(matched)} images")
        print(f"Mean DFC:         {mean_dfc:.1%}")
        print(f"Ethnicity match:  {eth_acc:.1%}")
        print(f"Age match:        {age_acc:.1%}")
        print(f"Gender match:     {gen_acc:.1%}")
        print(f"\nResults: {OUTPUT_CSV}")
        print(f"Summary: {OUTPUT_JSON}")


def _save_csv(results):
    if not results:
        return
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    keys = results[0].keys()
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
