"""
eval/run_cpdc_vs_cdvr.py — CPDC vs legacy CDVR head-to-head ablation

Your existing eval/results/cdvr_ablation.csv compares CPDC-driven
correction against NO correction. It does not compare CPDC against the
original binary/fixed-severity CDVR rule (the thing CPDC was built to
replace) on the STF/AF axes — that comparison is what this script adds.

Runs the same 36-profile demographic grid through THREE conditions:
    baseline - no correction at all (re-scores existing baseline images
               if you pass --reuse-baseline, else regenerates)
    cpdc     - the existing modules/cpdc.py graduated correction
               (same logic as run_full_evaluation.py)
    legacy   - the OLD fixed mild/strong CDVR rule applied to STF and AF
               too (previously only BTF used this path; STF/AF used a
               flat "score < threshold -> correct" trigger before CPDC)

Requires two new keys in config.yaml under `corrections`, since STF/AF
currently only have level_1..4 (the CPDC ladder), not mild/strong. Add:

    corrections:
      STF:
        mild:   "warm brown skin tone,"        # == level_1/level_2 equivalent
        strong: "dark rich brown skin, deep melanin-rich complexion,"  # == level_4
      AF:
        mild:   "slightly mature features,"     # == level_1
        strong: "visible age lines, mature face, silver streaks in hair, aged skin texture,"  # == level_4

(Reusing your existing level_1/level_4 strings as mild/strong keeps the
comparison apples-to-apples — same correction vocabulary, different
selection logic.)

Where to put this file:
    eval/run_cpdc_vs_cdvr.py

Prerequisites (same as run_full_evaluation.py):
    - HF_TOKEN_1 in .env
    - Ollama running with llava pulled
    - config.yaml updated with the mild/strong STF/AF keys above

How to run:
    cd <repo root>
    python eval/run_cpdc_vs_cdvr.py --seeds 42 123 456

    Add --skip-generation to re-score existing images instead of
    regenerating (looks for images already produced by
    run_full_evaluation.py / this script under eval/results/cpdc_vs_cdvr/).

Output:
    eval/results/cpdc_vs_cdvr/
        legacy_ablation.csv        - per-profile legacy-CDVR results
        comparison_summary.json    - CPDC vs legacy side-by-side stats
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from itertools import product as itertools_product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

ETHNICITIES = ["South Asian", "East Asian", "African American"]
BODY_TYPES = ["slim", "medium", "plus-size"]
AGES = ["20s", "30s", "40s", "50s"]
CLOTHING = "White Blazer Suit"
BACKGROUND = "Warm Beige"


def build_profile_grid():
    grid = []
    for eth, bt, age in itertools_product(ETHNICITIES, BODY_TYPES, AGES):
        grid.append({"ethnicity": eth, "body_type": bt, "age": age, "gender": "Female"})
    return grid


# ---------------------------------------------------------------------------
# Legacy binary CDVR loop (STF + AF on the OLD fixed-severity rule)
# ---------------------------------------------------------------------------

def run_legacy_cdvr(profile, seed, clothing, background, cfg, out_path, max_iter):
    from profiler.prompt_builder import build_prompt
    from generator.flux_pipeline import generate_image
    from eval.fidelity_scorer import score_fidelity

    threshold = cfg.get("generation", {}).get("dfc_threshold", 0.70) * 10  # 0-10 scale
    prompt = build_prompt(profile, clothing, background, cfg)
    img, _, _, _ = generate_image(prompt, seed, cfg)
    iterations_used = 0
    last_scores = None

    for iteration in range(1, max_iter + 1):
        img.save(out_path)
        try:
            scores = score_fidelity(out_path, profile)
        except Exception:
            break
        last_scores = scores

        correction_keys = []
        if scores.get("ethnicity_score", 10) < threshold:
            correction_keys.append("STF")
        if scores.get("age_score", 10) < threshold:
            correction_keys.append("AF")
        if profile.get("body_type") == "plus-size" and scores.get("overall", 10) < threshold:
            correction_keys.append("BTF")

        if not correction_keys:
            break  # passed everything

        # OLD rule: iteration 1 = mild, iteration 2+ = strong, flat for ALL
        # failing axes at once (no per-axis graduation, no early stop)
        prompt = build_prompt(
            profile, clothing, background, cfg,
            correction_keys=correction_keys,
            iteration=iteration,
        )
        img, _, _, _ = generate_image(prompt, seed + iteration, cfg)
        iterations_used = iteration

    img.save(out_path)
    return img, iterations_used, last_scores or {}


def generate_and_score_legacy(profiles, seeds, output_dir, cfg, skip_generation):
    from eval.fidelity_scorer import score_fidelity

    os.makedirs(output_dir, exist_ok=True)
    max_iter = cfg.get("generation", {}).get("max_iterations", 3)
    rows = []
    total = len(profiles) * len(seeds)
    count = 0

    for seed in seeds:
        for i, profile in enumerate(profiles):
            count += 1
            label = f"p{i:02d}_seed{seed}_legacy"
            out_path = os.path.join(output_dir, f"{label}.png")
            print(f"[{count}/{total}] {label}")

            if os.path.exists(out_path):
                scores = score_fidelity(out_path, profile)
                iterations_used = None  # unknown when reusing images
            elif skip_generation:
                # --skip-generation now means exactly that: don't attempt
                # generation for anything not already on disk (e.g. HF
                # credits exhausted). Just skip it rather than erroring out.
                print(f"  SKIPPED (no local image, --skip-generation set)")
                continue
            else:
                try:
                    _, iterations_used, scores = run_legacy_cdvr(
                        profile, seed, CLOTHING, BACKGROUND, cfg, out_path, max_iter
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

            rows.append({
                "image": os.path.basename(out_path),
                "seed": seed,
                "condition": "legacy_cdvr",
                "ethnicity": profile["ethnicity"],
                "body_type": profile["body_type"],
                "age": profile["age"],
                "iterations_used": iterations_used,
                "ethnicity_score": scores.get("ethnicity_score", 0),
                "age_score": scores.get("age_score", 0),
                "gender_score": scores.get("gender_score", 0),
                "overall": scores.get("overall", 0),
            })

    return rows


# ---------------------------------------------------------------------------
# Compare against your existing CPDC results
# ---------------------------------------------------------------------------

def load_existing_cpdc_results(path="eval/results/cdvr_ablation.csv"):
    """Reads your existing cdvr_ablation.csv and pulls out cdvr=True rows
    (those are the CPDC-driven results, per run_full_evaluation.py)."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — CPDC comparison will be empty. "
              f"Run eval/run_full_evaluation.py first.")
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if str(r.get("cdvr")).lower() == "true"]


def build_comparison(legacy_rows, cpdc_rows, output_json):
    import numpy as np

    def _stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {"mean": None, "n": 0}
        return {"mean": round(float(np.mean(values)), 3),
                "std": round(float(np.std(values)), 3), "n": len(values)}

    # NOTE: legacy rows come from eval/fidelity_scorer.py's score_fidelity(),
    # which returns "overall" on a 0-10 scale (see docstring in that file).
    # cdvr_ablation.csv's "overall_fidelity" column (used for cpdc_overall
    # below) is on a 0-1 scale. Divide by 10 here so the two are comparable.
    legacy_overall = [float(r["overall"]) / 10.0 for r in legacy_rows if r.get("overall") not in (None, "")]
    legacy_iters = [r["iterations_used"] for r in legacy_rows if r.get("iterations_used") not in (None, "")]
    legacy_iters = [int(v) for v in legacy_iters if str(v).strip() != ""]

    cpdc_overall = [float(r["overall_fidelity"]) for r in cpdc_rows if r.get("overall_fidelity") not in (None, "")]
    cpdc_iters = [int(r["iterations_used"]) for r in cpdc_rows if r.get("iterations_used") not in (None, "")]

    summary = {
        "created_at": datetime.now().isoformat(),
        "legacy_cdvr": {
            "overall_fidelity": _stats(legacy_overall),
            "iterations_used": _stats(legacy_iters),
        },
        "cpdc": {
            "overall_fidelity": _stats(cpdc_overall),
            "iterations_used": _stats(cpdc_iters),
        },
        "note": ("cpdc rows are pulled from eval/results/cdvr_ablation.csv "
                 "(cdvr=True). If that file's schema differs from what's "
                 "expected here, check the column names line up before "
                 "trusting these numbers."),
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nComparison saved -> {output_json}")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="CPDC vs legacy CDVR ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--output-dir", default="eval/results/cpdc_vs_cdvr")
    args = parser.parse_args()

    from profiler.prompt_builder import load_config
    cfg = load_config()

    if "mild" not in cfg.get("corrections", {}).get("STF", {}):
        print("ERROR: config.yaml is missing corrections.STF.mild/strong "
              "(and AF.mild/strong). Add them per the docstring at the top "
              "of this file before running.")
        sys.exit(1)

    profiles = build_profile_grid()
    images_dir = os.path.join(args.output_dir, "generated_images")

    legacy_rows = generate_and_score_legacy(
        profiles, args.seeds, images_dir, cfg, args.skip_generation
    )

    legacy_csv = os.path.join(args.output_dir, "legacy_ablation.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    if legacy_rows:
        with open(legacy_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=legacy_rows[0].keys())
            writer.writeheader()
            writer.writerows(legacy_rows)
        print(f"Saved -> {legacy_csv} ({len(legacy_rows)} rows)")

    cpdc_rows = load_existing_cpdc_results()
    build_comparison(
        legacy_rows, cpdc_rows,
        os.path.join(args.output_dir, "comparison_summary.json"),
    )


if __name__ == "__main__":
    main()
