"""
Score a folder of generated images with the non-LLM scorer and report fidelity.

Targets are read from file names, e.g.
    South-Asian_slim_20s_seed42.png          (female, slim, 20s)
    South-Asian_30s_male_seed42.png          (male)
    African-American_plus-size_40s_seed7.png

    python -m eval.score_folder IMAGES_DIR --out outputs/rescore
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from eval.stats import wilson_ci
from scoring import evaluate_profile
from scoring.cache import CachedScorer
from scoring.registry import get_scorer

ETH = {"south-asian": "South Asian", "east-asian": "East Asian", "southeast-asian": "Southeast Asian",
       "african-american": "African American", "latin-american": "Latin American",
       "middle-eastern": "Middle Eastern", "white": "White"}


def profile_from_name(fname: str):
    stem = os.path.splitext(os.path.basename(fname))[0].lower()
    eth = next((v for k, v in ETH.items() if stem.startswith(k)), None)
    age = re.search(r"(?<![0-9])([2-6]0s)", stem)
    body = re.search(r"(slim|medium|plus-size)", stem)
    gender = "male" if re.search(r"(?<![a-z])male(?![a-z])", stem) and "female" not in stem else "female"
    if not (eth and age):
        return None
    return {"ethnicity": eth, "age": age.group(1), "gender": gender,
            "body_type": body.group(1) if body else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="outputs/rescore")
    ap.add_argument("--scorer", default="deepface")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    scorer = CachedScorer(get_scorer(a.scorer, cfg), os.path.join(a.out, "cache.json"))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(a.folder)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        prof = profile_from_name(f)
        if prof is None:
            print("skip (cannot parse target from name):", f)
            continue
        rep = evaluate_profile(prof, scorer.read(os.path.join(a.folder, f)), cfg)
        r = rep.reading
        row = {"image": f, **{f"target_{k}": v for k, v in prof.items()},
               "face_found": rep.face_found, "est_age": None if r.age is None else round(r.age, 1)}
        for ax in ("ethnicity", "gender", "age"):
            res = rep.axes[ax]
            row[f"{ax}_pred"] = res.predicted
            row[f"{ax}_conf"] = round(res.confidence, 3)
            row[f"{ax}_pass"] = res.passed
            row[f"{ax}_strict"] = res.strict
        rows.append(row)
        print(f"{f[:40]:40s} age={row['est_age']} eth={row['ethnicity_pred']}({row['ethnicity_pass']}) "
              f"gen={row['gender_pred']}({row['gender_pass']}) age_pass={row['age_pass']} strict={row['age_strict']}")
    with open(os.path.join(a.out, "scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    summ = {"n": n, "scorer": a.scorer}
    for ax in ("ethnicity", "gender", "age"):
        for kind in ("pass", "strict"):
            k = sum(1 for r in rows if r[f"{ax}_{kind}"])
            lo, hi = wilson_ci(k, n)
            summ[f"{ax}_{kind}"] = {"k": k, "rate": round(k / n, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    by = {}
    for r in rows:
        by.setdefault(r["target_age"], []).append(r)
    summ["age_by_target"] = {k: {"n": len(v), "mean_est": round(sum(x["est_age"] or 0 for x in v) / len(v), 1),
                                 "pass": sum(1 for x in v if x["age_pass"]),
                                 "strict": sum(1 for x in v if x["age_strict"])} for k, v in sorted(by.items())}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
