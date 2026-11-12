"""
eval/bootstrap_ci.py — Bootstrap confidence intervals for the CPDC gain

Answers: "Is the fidelity improvement from CPDC (0.75 -> 0.797, or whatever
your actual numbers are) real, or could it just be noise from only having
3 seeds?"

Reads your EXISTING eval/results/cdvr_ablation.csv directly — no new
generation, no new scoring, just statistics on data you already have.

Where to put this file:
    eval/bootstrap_ci.py

How to run:
    cd <repo root>
    python eval/bootstrap_ci.py

    Optional flags:
    python eval/bootstrap_ci.py --csv eval/results/cdvr_ablation.csv \
                                 --n-boot 10000 --alpha 0.05

Output:
    eval/results/bootstrap_ci_summary.json
    Also prints a plain-text summary to stdout you can paste into the paper.

What it computes:
    1. Bootstrap CI on mean overall_fidelity for cdvr=True (CPDC) and
       cdvr=False (baseline) separately.
    2. Bootstrap CI on the PAIRED DIFFERENCE (CPDC - baseline) per profile
       -- this is the number that actually matters for the paper. If its
       95% CI excludes 0, the improvement is statistically real.
    3. Same treatment for iterations_used (efficiency claim), CPDC only
       vs a placeholder for legacy CDVR if you've already run
       run_cpdc_vs_cdvr.py and have eval/results/cpdc_vs_cdvr/legacy_ablation.csv.
"""

import argparse
import csv
import json
import os

import numpy as np


def load_rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def bootstrap_mean_ci(values, n_boot=10000, alpha=0.05, seed=0):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {
        "mean": round(float(values.mean()), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "n": n,
    }


def bootstrap_paired_diff_ci(paired_diffs, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap CI on the mean of per-profile (CPDC - baseline) differences.
    If ci_lower > 0, the improvement is significant at this alpha level."""
    result = bootstrap_mean_ci(paired_diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    result["significant"] = result["ci_lower"] > 0
    return result


def build_paired_diffs(rows):
    """Matches cdvr=True and cdvr=False rows by (image name minus seed
    suffix logic is already baked into 'image' + 'seed' + target_* cols),
    then returns list of (cpdc_fidelity - baseline_fidelity)."""
    by_key = {}
    for r in rows:
        key = (r["image"], r["seed"], r["target_ethnicity"], r["target_body_type"], r["target_age"])
        by_key.setdefault(key, {})[r["cdvr"].lower()] = r

    diffs = []
    iter_cpdc = []
    for key, pair in by_key.items():
        if "true" in pair and "false" in pair:
            f_cpdc = float(pair["true"]["overall_fidelity"])
            f_base = float(pair["false"]["overall_fidelity"])
            diffs.append(f_cpdc - f_base)
            iter_cpdc.append(int(pair["true"]["iterations_used"]))
    return diffs, iter_cpdc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="eval/results/cdvr_ablation.csv")
    parser.add_argument("--legacy-csv", default="eval/results/cpdc_vs_cdvr/legacy_ablation.csv",
                         help="Optional, only used if run_cpdc_vs_cdvr.py has already been run")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default="eval/results/bootstrap_ci_summary.json")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"{args.csv} not found — run eval/run_full_evaluation.py first."
        )
    rows = load_rows(args.csv)

    cpdc_fid = [float(r["overall_fidelity"]) for r in rows if r["cdvr"].lower() == "true"]
    base_fid = [float(r["overall_fidelity"]) for r in rows if r["cdvr"].lower() == "false"]

    diffs, iter_cpdc = build_paired_diffs(rows)

    summary = {
        "cpdc_fidelity": bootstrap_mean_ci(cpdc_fid, args.n_boot, args.alpha),
        "baseline_fidelity": bootstrap_mean_ci(base_fid, args.n_boot, args.alpha),
        "paired_diff_cpdc_minus_baseline": bootstrap_paired_diff_ci(diffs, args.n_boot, args.alpha),
        "cpdc_iterations_used": bootstrap_mean_ci(iter_cpdc, args.n_boot, args.alpha),
    }

    if os.path.exists(args.legacy_csv):
        legacy_rows = load_rows(args.legacy_csv)
        # NOTE: legacy_ablation.csv's "overall" column is on a 0-10 scale
        # (from eval/fidelity_scorer.py's score_fidelity()), while cpdc_fid
        # above (from cdvr_ablation.csv) is on a 0-1 scale. Divide by 10 so
        # both are directly comparable.
        legacy_fid = [float(r["overall"]) / 10.0 for r in legacy_rows if r.get("overall") not in (None, "")]
        legacy_iters = [int(r["iterations_used"]) for r in legacy_rows if r.get("iterations_used") not in (None, "")]
        if legacy_fid:
            summary["legacy_cdvr_fidelity"] = bootstrap_mean_ci(legacy_fid, args.n_boot, args.alpha)
        if legacy_iters:
            summary["legacy_cdvr_iterations_used"] = bootstrap_mean_ci(legacy_iters, args.n_boot, args.alpha)
        # Paired diff CPDC vs legacy, matched by index only if counts line up
        if len(legacy_fid) == len(cpdc_fid):
            cpdc_minus_legacy = [c - l for c, l in zip(cpdc_fid, legacy_fid)]
            summary["paired_diff_cpdc_minus_legacy"] = bootstrap_paired_diff_ci(
                cpdc_minus_legacy, args.n_boot, args.alpha
            )
    else:
        summary["note"] = ("legacy_ablation.csv not found — run eval/run_cpdc_vs_cdvr.py "
                            "first if you want CPDC-vs-legacy-CDVR bootstrap stats too. "
                            "This summary only covers CPDC vs no-correction.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {args.output}")

    d = summary["paired_diff_cpdc_minus_baseline"]
    verdict = "IS" if d["significant"] else "is NOT"
    print(f"\nCPDC fidelity gain over baseline {verdict} statistically "
          f"significant at alpha={args.alpha} "
          f"(mean diff={d['mean']}, 95% CI=[{d['ci_lower']}, {d['ci_upper']}])")


if __name__ == "__main__":
    main()