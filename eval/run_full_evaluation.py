"""
run_full_evaluation.py - Master Evaluation Runner
===================================================
Generates the full 36-combination demographic grid (3 ethnicities x 3 body types x 4 ages)
with CDVR on and CDVR off, then scores every output with both LLaVA DFC and CLIP fidelity.

CDVR now uses CPDC (Confidence-Proportional Demographic Correction,
modules/cpdc.py) instead of a fixed mild/strong escalation: correction
severity is chosen from a continuous error signal per axis (CLIP
target-class confidence for ethnicity, LLaVA self-reported confidence for
age), and a diminishing-returns detector stops correcting an axis once
further attempts stop improving, rather than always burning the full
iteration budget.

Produces:
    eval/results/
        generated_images/          - all generated PNGs
        dfc_scores.csv             - LLaVA-based fidelity scores
        clip_scores.csv            - CLIP-based fidelity scores
        cdvr_ablation.csv          - iteration-0 vs final scores
        bias_distribution.json     - BiasTracker output
        scorer_agreement.csv       - LLaVA vs CLIP correlation
        summary.json               - aggregate stats for paper tables

Usage:
    cd adfidelity/
    python eval/run_full_evaluation.py --seeds 42 123 456

    Add --skip-generation to score existing images without regenerating.
    Add --skip-llava to skip LLaVA scoring (if Ollama isn't available).
    Add --skip-clip to skip CLIP scoring (if model not trained yet).

Prerequisites:
    - HF_TOKEN_1 in .env (for FLUX generation)
    - Ollama running with llava pulled (for DFC scoring)
    - CLIP model trained (eval/clip_fidelity_model/*.pt)
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from itertools import product as itertools_product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# -- Demographic grid ---------------------------------------------------------

ETHNICITIES = ["South Asian", "East Asian", "African American"]
BODY_TYPES  = ["slim", "medium", "plus-size"]
AGES        = ["20s", "30s", "40s", "50s"]
CLOTHINGS   = ["White Blazer Suit"]  # fix clothing for controlled comparison
BACKGROUNDS = ["Warm Beige"]         # fix background for controlled comparison

def build_profile_grid():
    """Returns list of 36 demographic profile dicts."""
    grid = []
    for eth, bt, age in itertools_product(ETHNICITIES, BODY_TYPES, AGES):
        grid.append({
            "ethnicity": eth,
            "body_type": bt,
            "age": age,
            "gender": "Female",
        })
    return grid


# -- Step 1: Generate images ---------------------------------------------------

def generate_all(profiles, seeds, output_dir, clothing, background):
    """Generate images for every profile x seed, with and without CDVR."""
    from profiler.prompt_builder import build_prompt, load_config
    from generator.flux_pipeline import generate_image
    from modules.cpdc import AxisTracker
    from eval.fidelity_scorer import score_fidelity
    cfg = load_config()
    os.makedirs(output_dir, exist_ok=True)

    manifest = []  # tracks what was generated and where

    total = len(profiles) * len(seeds) * 2  # x2 for CDVR on/off
    count = 0

    for seed in seeds:
        for i, profile in enumerate(profiles):
            for cdvr_enabled in [False, True]:
                count += 1
                label = f"p{i:02d}_seed{seed}_cdvr{'ON' if cdvr_enabled else 'OFF'}"
                out_path = os.path.join(output_dir, f"{label}.png")

                if os.path.exists(out_path):
                    print(f"  [{count}/{total}] SKIP (exists): {label}")
                    manifest.append({
                        "index": i, "seed": seed, "cdvr": cdvr_enabled,
                        "profile": profile, "image": out_path, "iterations": 0,
                    })
                    continue

                print(f"  [{count}/{total}] Generating: {label}")

                try:
                    # Iteration 0: no corrections
                    prompt = build_prompt(profile, clothing, background, cfg)
                    img, device, res, steps = generate_image(prompt, seed, cfg)

                    iterations_used = 0

                    if cdvr_enabled:
                        # CPDC-driven CDVR loop (up to max_iterations)
                        max_iter = cfg.get("generation", {}).get("max_iterations", 3)
                        trackers = {
                            "STF": AxisTracker(axis="STF"),
                            "AF":  AxisTracker(axis="AF"),
                        }

                        for iteration in range(1, max_iter + 1):
                            img.save(out_path)

                            try:
                                llava_scores = score_fidelity(out_path, profile)
                            except Exception:
                                break

                            age_conf = llava_scores.get(
                                "age_confidence", llava_scores.get("age_score", 0) / 10)
                            eth_conf = llava_scores.get(
                                "ethnicity_confidence", llava_scores.get("ethnicity_score", 0) / 10)

                            correction_levels = {}
                            for conf, key in [(age_conf, "AF"), (eth_conf, "STF")]:
                                stop = trackers[key].update(conf)
                                if stop:
                                    continue  # diminishing returns - stop correcting this axis
                                level = trackers[key].current_level()
                                if level > 0:
                                    correction_levels[key] = level

                            # Body type keeps the old binary rule (no trained
                            # confidence classifier exists for it)
                            correction_keys = []
                            if (profile.get("body_type") == "plus-size"
                                    and llava_scores.get("overall", 0) < 7):
                                correction_keys = ["BTF"]

                            if not correction_levels and not correction_keys:
                                break  # passed, or every failing axis has stalled

                            prompt = build_prompt(
                                profile, clothing, background, cfg,
                                correction_levels=correction_levels,
                                correction_keys=correction_keys,
                                iteration=iteration,
                            )
                            img, device, res, steps = generate_image(prompt, seed + iteration, cfg)
                            iterations_used = iteration

                    img.save(out_path)
                    manifest.append({
                        "index": i, "seed": seed, "cdvr": cdvr_enabled,
                        "profile": profile, "image": out_path,
                        "iterations": iterations_used,
                    })

                except Exception as e:
                    print(f"    ERROR: {e}")
                    manifest.append({
                        "index": i, "seed": seed, "cdvr": cdvr_enabled,
                        "profile": profile, "image": None,
                        "error": str(e),
                    })

    return manifest


# -- Step 2: Score with LLaVA DFC ----------------------------------------------

def score_with_llava(manifest, output_csv):
    """Score all generated images with LLaVA-based DFC."""
    from eval.fidelity_scorer import score_fidelity

    results = []
    valid = [m for m in manifest if m.get("image") and os.path.exists(m["image"])]

    for i, entry in enumerate(valid):
        print(f"  [LLaVA {i+1}/{len(valid)}] {os.path.basename(entry['image'])}")
        try:
            scores = score_fidelity(entry["image"], entry["profile"])
        except Exception as e:
            print(f"    ERROR: {e}")
            scores = {"age_score": 0, "gender_score": 0, "ethnicity_score": 0, "overall": 0, "notes": str(e)}

        results.append({
            **{k: v for k, v in entry.items() if k != "profile"},
            **entry["profile"],
            "scorer": "llava",
            **scores,
        })

    _save_csv(results, output_csv)
    return results


# -- Step 3: Score with CLIP ---------------------------------------------------

def score_with_clip(manifest, output_csv):
    """Score all generated images with CLIP fidelity scorer."""
    from eval.clip_trainer.clip_fidelity_scorer import CLIPFidelityScorer

    scorer = CLIPFidelityScorer()
    results = []
    valid = [m for m in manifest if m.get("image") and os.path.exists(m["image"])]

    for i, entry in enumerate(valid):
        print(f"  [CLIP {i+1}/{len(valid)}] {os.path.basename(entry['image'])}")
        try:
            scores = scorer.score(entry["image"], entry["profile"])
        except Exception as e:
            print(f"    ERROR: {e}")
            scores = {"age_score": 0, "gender_score": 0, "race_score": 0, "overall_fidelity": 0}

        results.append({
            **{k: v for k, v in entry.items() if k != "profile"},
            **entry["profile"],
            "scorer": "clip",
            **scores,
        })

    _save_csv(results, output_csv)
    return results


# -- Step 4: Compute agreement between scorers ---------------------------------

def compute_agreement(llava_results, clip_results, output_csv):
    """Compute per-image agreement between LLaVA and CLIP fidelity scores."""
    # Index CLIP results by image path for fast lookup
    clip_by_image = {r["image"]: r for r in clip_results}

    rows = []
    for lr in llava_results:
        img = lr.get("image")
        cr = clip_by_image.get(img)
        if not cr:
            continue

        rows.append({
            "image": os.path.basename(img),
            "ethnicity": lr.get("ethnicity", ""),
            "body_type": lr.get("body_type", ""),
            "age": lr.get("age", ""),
            "cdvr": lr.get("cdvr", ""),
            "llava_overall": lr.get("overall", 0),
            "clip_overall": cr.get("overall_fidelity", 0),
            "llava_ethnicity": lr.get("ethnicity_score", 0),
            "clip_race": cr.get("race_score", 0),
            "llava_age": lr.get("age_score", 0),
            "clip_age": cr.get("age_score", 0),
        })

    _save_csv(rows, output_csv)

    # Print correlation
    if rows:
        import numpy as np
        llava_vals = [r["llava_overall"] for r in rows]
        clip_vals = [r["clip_overall"] for r in rows]
        if len(set(llava_vals)) > 1 and len(set(clip_vals)) > 1:
            corr = np.corrcoef(llava_vals, clip_vals)[0, 1]
            print(f"\n  Pearson correlation (LLaVA vs CLIP overall): {corr:.3f}")
        else:
            print("\n  Insufficient variance for correlation.")

    return rows


# -- Step 5: Build summary stats ------------------------------------------------

def build_summary(llava_results, clip_results, manifest, output_json):
    """Aggregate stats for paper tables."""
    import numpy as np

    def _stats(values):
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": round(float(np.mean(values)), 3),
            "std": round(float(np.std(values)), 3),
            "min": round(float(np.min(values)), 3),
            "max": round(float(np.max(values)), 3),
            "n": len(values),
        }

    # CDVR ablation
    cdvr_on = [r for r in llava_results if r.get("cdvr") is True]
    cdvr_off = [r for r in llava_results if r.get("cdvr") is False]

    # Per-group breakdown
    group_stats = {}
    for r in llava_results:
        key = f"{r.get('ethnicity', '?')}_{r.get('body_type', '?')}_{r.get('age', '?')}"
        group_stats.setdefault(key, []).append(r.get("overall", 0))

    summary = {
        "created_at": datetime.now().isoformat(),
        "total_images": len(manifest),
        "successful_generations": len([m for m in manifest if m.get("image")]),
        "llava_fidelity": {
            "all": _stats([r.get("overall", 0) for r in llava_results]),
            "cdvr_on": _stats([r.get("overall", 0) for r in cdvr_on]),
            "cdvr_off": _stats([r.get("overall", 0) for r in cdvr_off]),
        },
        "clip_fidelity": {
            "all": _stats([r.get("overall_fidelity", 0) for r in clip_results]),
        },
        "cdvr_iterations": _stats([m.get("iterations", 0) for m in manifest if m.get("cdvr") is True]),
        "per_group_llava_overall": {k: _stats(v) for k, v in group_stats.items()},
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved -> {output_json}")
    return summary


# -- Helpers ---------------------------------------------------------------------

def _save_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved -> {path} ({len(rows)} rows)")


# -- Main ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AdFidelity full evaluation runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for generation (default: 42 123 456)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip image generation, score existing images only")
    parser.add_argument("--skip-llava", action="store_true",
                        help="Skip LLaVA/DFC scoring")
    parser.add_argument("--skip-clip", action="store_true",
                        help="Skip CLIP scoring")
    parser.add_argument("--output-dir", default="eval/results",
                        help="Output directory (default: eval/results)")
    args = parser.parse_args()

    results_dir = args.output_dir
    images_dir  = os.path.join(results_dir, "generated_images")
    os.makedirs(results_dir, exist_ok=True)

    profiles = build_profile_grid()
    print(f"Demographic grid: {len(profiles)} profiles x {len(args.seeds)} seeds x 2 (CDVR on/off)")
    print(f"Total images to generate: {len(profiles) * len(args.seeds) * 2}\n")

    # -- Step 1: Generate ---------------------------------------------------
    manifest_path = os.path.join(results_dir, "manifest.json")
    if args.skip_generation and os.path.exists(manifest_path):
        print("=== Step 1: Loading existing manifest ===")
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        print("=== Step 1: Generating images ===")
        manifest = generate_all(
            profiles, args.seeds, images_dir,
            clothing=CLOTHINGS[0], background=BACKGROUNDS[0],
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved -> {manifest_path}\n")

    # -- Step 2: LLaVA DFC scoring -------------------------------------------
    llava_results = []
    llava_csv = os.path.join(results_dir, "dfc_scores.csv")
    if not args.skip_llava:
        print("=== Step 2: Scoring with LLaVA DFC ===")
        llava_results = score_with_llava(manifest, llava_csv)
    elif os.path.exists(llava_csv):
        import csv as csv_mod
        with open(llava_csv) as f:
            llava_results = list(csv_mod.DictReader(f))

    # -- Step 3: CLIP scoring -------------------------------------------------
    clip_results = []
    clip_csv = os.path.join(results_dir, "clip_scores.csv")
    if not args.skip_clip:
        print("\n=== Step 3: Scoring with CLIP ===")
        clip_results = score_with_clip(manifest, clip_csv)
    elif os.path.exists(clip_csv):
        import csv as csv_mod
        with open(clip_csv) as f:
            clip_results = list(csv_mod.DictReader(f))

    # -- Step 4: Agreement -----------------------------------------------------
    if llava_results and clip_results:
        print("\n=== Step 4: Scorer agreement ===")
        compute_agreement(
            llava_results, clip_results,
            os.path.join(results_dir, "scorer_agreement.csv"),
        )

    # -- Step 5: Summary --------------------------------------------------------
    print("\n=== Step 5: Building summary ===")
    build_summary(
        llava_results, clip_results, manifest,
        os.path.join(results_dir, "summary.json"),
    )

    print("\n=== Evaluation complete ===")
    print(f"All results in: {results_dir}/")


if __name__ == "__main__":
    main()
