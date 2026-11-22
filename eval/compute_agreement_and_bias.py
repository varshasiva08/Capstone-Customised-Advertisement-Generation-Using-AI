"""
compute_agreement_and_bias.py — Final analysis for paper
==========================================================
1. Computes agreement between CLIP and LLaVA scorers
2. Computes bias distribution across demographics

Usage:
    cd Capstone-Customised-Advertisement-Generation-Using-AI
    python eval/compute_agreement_and_bias.py
"""

import os
import csv
import json

CLIP_CSV = os.path.join("eval", "results", "clip_fidelity_scores.csv")
LLAVA_CSV = os.path.join("eval", "results", "dfc_scores.csv")
OUTPUT_DIR = os.path.join("eval", "results")


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_bool(val):
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "y")


def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main():
    # ── Load data ─────────────────────────────────────────────────────
    if not os.path.exists(CLIP_CSV):
        print(f"ERROR: {CLIP_CSV} not found")
        return
    if not os.path.exists(LLAVA_CSV):
        print(f"ERROR: {LLAVA_CSV} not found")
        return

    clip_rows = load_csv(CLIP_CSV)
    llava_rows = load_csv(LLAVA_CSV)

    print(f"CLIP scores:  {len(clip_rows)} rows")
    print(f"LLaVA scores: {len(llava_rows)} rows")

    # Index LLaVA by image name (without .png extension for matching)
    llava_by_image = {}
    for r in llava_rows:
        key = r.get("image", "").replace(".png", "")
        llava_by_image[key] = r

    # ══════════════════════════════════════════════════════════════════
    # 1. SCORER AGREEMENT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  SCORER AGREEMENT: CLIP vs LLaVA")
    print(f"{'='*60}")

    agreement_rows = []
    both_eth_match = 0
    both_eth_mismatch = 0
    disagree_eth = 0
    both_age_match = 0
    both_age_mismatch = 0
    disagree_age = 0
    both_gen_match = 0
    both_gen_mismatch = 0
    disagree_gen = 0
    matched = 0

    clip_overall_vals = []
    llava_overall_vals = []

    for cr in clip_rows:
        clip_key = cr.get("image", "")
        if clip_key in llava_by_image:
            lr = llava_by_image[clip_key]
            matched += 1

            c_eth = to_bool(cr.get("ethnicity_match", False))
            l_eth = to_bool(lr.get("ethnicity_match", False))
            c_age = to_bool(cr.get("age_match", False))
            l_age = to_bool(lr.get("age_match", False))
            c_gen = to_bool(cr.get("gender_match", False))
            l_gen = to_bool(lr.get("gender_match", False))

            if c_eth and l_eth: both_eth_match += 1
            elif not c_eth and not l_eth: both_eth_mismatch += 1
            else: disagree_eth += 1

            if c_age and l_age: both_age_match += 1
            elif not c_age and not l_age: both_age_mismatch += 1
            else: disagree_age += 1

            if c_gen and l_gen: both_gen_match += 1
            elif not c_gen and not l_gen: both_gen_mismatch += 1
            else: disagree_gen += 1

            c_overall = to_float(cr.get("overall_fidelity"))
            l_overall = to_float(lr.get("overall_dfc"))
            if c_overall is not None:
                clip_overall_vals.append(c_overall)
            if l_overall is not None:
                llava_overall_vals.append(l_overall)

            agreement_rows.append({
                "image": clip_key,
                "target_ethnicity": cr.get("target_ethnicity", ""),
                "target_body_type": cr.get("target_body_type", ""),
                "target_age": cr.get("target_age", ""),
                "clip_ethnicity_match": c_eth,
                "llava_ethnicity_match": l_eth,
                "clip_age_match": c_age,
                "llava_age_match": l_age,
                "clip_gender_match": c_gen,
                "llava_gender_match": l_gen,
                "clip_overall": c_overall,
                "llava_overall": l_overall,
            })

    if matched == 0:
        print("No matching images found between CLIP and LLaVA results!")
        print("Check that filenames match between the two CSVs.")
        return

    # Compute agreement percentages
    eth_agree = (both_eth_match + both_eth_mismatch) / matched
    age_agree = (both_age_match + both_age_mismatch) / matched
    gen_agree = (both_gen_match + both_gen_mismatch) / matched
    overall_agree = (eth_agree + age_agree + gen_agree) / 3

    # Compute Pearson correlation if possible
    pearson = None
    if len(clip_overall_vals) == len(llava_overall_vals) and len(clip_overall_vals) > 1:
        n = len(clip_overall_vals)
        mean_c = sum(clip_overall_vals) / n
        mean_l = sum(llava_overall_vals) / n
        cov = sum((c - mean_c) * (l - mean_l) for c, l in zip(clip_overall_vals, llava_overall_vals)) / n
        std_c = (sum((c - mean_c)**2 for c in clip_overall_vals) / n) ** 0.5
        std_l = (sum((l - mean_l)**2 for l in llava_overall_vals) / n) ** 0.5
        if std_c > 0 and std_l > 0:
            pearson = cov / (std_c * std_l)

    print(f"\nMatched images: {matched}")
    print(f"\nPer-axis agreement (both agree on match/mismatch):")
    print(f"  Ethnicity: {eth_agree:.1%}  (agree-match:{both_eth_match} agree-miss:{both_eth_mismatch} disagree:{disagree_eth})")
    print(f"  Age:       {age_agree:.1%}  (agree-match:{both_age_match} agree-miss:{both_age_mismatch} disagree:{disagree_age})")
    print(f"  Gender:    {gen_agree:.1%}  (agree-match:{both_gen_match} agree-miss:{both_gen_mismatch} disagree:{disagree_gen})")
    print(f"  Overall:   {overall_agree:.1%}")

    if pearson is not None:
        print(f"\nPearson correlation (overall scores): {pearson:.3f}")

    # CLIP summary
    clip_eth_acc = sum(1 for r in clip_rows if to_bool(r.get("ethnicity_match"))) / len(clip_rows) if clip_rows else 0
    clip_age_acc = sum(1 for r in clip_rows if to_bool(r.get("age_match"))) / len(clip_rows) if clip_rows else 0
    clip_gen_acc = sum(1 for r in clip_rows if to_bool(r.get("gender_match"))) / len(clip_rows) if clip_rows else 0

    llava_eth_acc = sum(1 for r in llava_rows if to_bool(r.get("ethnicity_match"))) / len(llava_rows) if llava_rows else 0
    llava_age_acc = sum(1 for r in llava_rows if to_bool(r.get("age_match"))) / len(llava_rows) if llava_rows else 0
    llava_gen_acc = sum(1 for r in llava_rows if to_bool(r.get("gender_match"))) / len(llava_rows) if llava_rows else 0

    print(f"\nSide-by-side accuracy:")
    print(f"  {'Metric':<20} {'CLIP':>10} {'LLaVA':>10}")
    print(f"  {'Ethnicity':<20} {clip_eth_acc:>9.1%} {llava_eth_acc:>9.1%}")
    print(f"  {'Age':<20} {clip_age_acc:>9.1%} {llava_age_acc:>9.1%}")
    print(f"  {'Gender':<20} {clip_gen_acc:>9.1%} {llava_gen_acc:>9.1%}")

    # Save agreement CSV
    agreement_csv = os.path.join(OUTPUT_DIR, "scorer_agreement.csv")
    if agreement_rows:
        with open(agreement_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=agreement_rows[0].keys())
            writer.writeheader()
            writer.writerows(agreement_rows)
        print(f"\n  Saved → {agreement_csv}")

    # ══════════════════════════════════════════════════════════════════
    # 2. BIAS DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  BIAS DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")

    # Count generations per group
    eth_counts = {}
    bt_counts = {}
    age_counts = {}
    group_dfc = {}

    for r in llava_rows:
        eth = r.get("target_ethnicity", "?")
        bt = r.get("target_body_type", "?")
        age = r.get("target_age", "?")
        dfc = to_float(r.get("overall_dfc", 0))

        eth_counts[eth] = eth_counts.get(eth, 0) + 1
        bt_counts[bt] = bt_counts.get(bt, 0) + 1
        age_counts[age] = age_counts.get(age, 0) + 1

        key = f"{eth}_{bt}_{age}"
        if key not in group_dfc:
            group_dfc[key] = []
        if dfc is not None:
            group_dfc[key].append(dfc)

    print(f"\nGeneration count by ethnicity:")
    for k, v in sorted(eth_counts.items()):
        print(f"  {k}: {v}")

    print(f"\nGeneration count by body type:")
    for k, v in sorted(bt_counts.items()):
        print(f"  {k}: {v}")

    print(f"\nGeneration count by age:")
    for k, v in sorted(age_counts.items()):
        print(f"  {k}: {v}")

    # Check for uniform distribution
    total = len(llava_rows)
    expected_per_eth = total / len(eth_counts) if eth_counts else 0
    expected_per_bt = total / len(bt_counts) if bt_counts else 0
    expected_per_age = total / len(age_counts) if age_counts else 0

    # Chi-square test (manual, no scipy needed)
    def chi_square(observed, expected):
        return sum((o - expected)**2 / expected for o in observed) if expected > 0 else 0

    chi_eth = chi_square(eth_counts.values(), expected_per_eth)
    chi_bt = chi_square(bt_counts.values(), expected_per_bt)
    chi_age = chi_square(age_counts.values(), expected_per_age)

    print(f"\nChi-square vs uniform distribution:")
    print(f"  Ethnicity: χ² = {chi_eth:.2f} (expected {expected_per_eth:.0f} per group)")
    print(f"  Body type: χ² = {chi_bt:.2f} (expected {expected_per_bt:.0f} per group)")
    print(f"  Age:       χ² = {chi_age:.2f} (expected {expected_per_age:.0f} per group)")

    # Per-group DFC breakdown
    print(f"\nMean DFC by demographic group (LLaVA):")
    print(f"  {'Group':<45} {'N':>4} {'Mean DFC':>10}")
    for key in sorted(group_dfc.keys()):
        vals = group_dfc[key]
        mean = sum(vals) / len(vals) if vals else 0
        print(f"  {key:<45} {len(vals):>4} {mean:>9.1%}")

    # ══════════════════════════════════════════════════════════════════
    # 3. SAVE FULL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    full_summary = {
        "scorer_agreement": {
            "matched_images": matched,
            "ethnicity_agreement": round(eth_agree, 3),
            "age_agreement": round(age_agree, 3),
            "gender_agreement": round(gen_agree, 3),
            "overall_agreement": round(overall_agree, 3),
            "pearson_correlation": round(pearson, 3) if pearson else None,
        },
        "side_by_side": {
            "clip": {"ethnicity": round(clip_eth_acc, 3), "age": round(clip_age_acc, 3), "gender": round(clip_gen_acc, 3)},
            "llava": {"ethnicity": round(llava_eth_acc, 3), "age": round(llava_age_acc, 3), "gender": round(llava_gen_acc, 3)},
        },
        "bias_distribution": {
            "by_ethnicity": eth_counts,
            "by_body_type": bt_counts,
            "by_age": age_counts,
            "chi_square_ethnicity": round(chi_eth, 3),
            "chi_square_body_type": round(chi_bt, 3),
            "chi_square_age": round(chi_age, 3),
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, "full_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Scorer agreement:  {agreement_csv}")
    print(f"  Full summary:      {summary_path}")
    print(f"\nYou now have everything needed for the paper!")


if __name__ == "__main__":
    main()
