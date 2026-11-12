"""
Diagnose a finished run without generating anything (no API quota needed).

Re-reads every saved best image in OUT/images with the verifier and prints what
the classifier actually saw next to the target, so you can tell whether a
failed check is a bad image or a bad measurement.

    python eval/diagnose_run.py results/pilot > diagnosis.txt
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from scoring import evaluate_profile
from scoring.cache import CachedScorer
from scoring.registry import get_scorer


def main(out):
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    sc = CachedScorer(get_scorer(cfg["scoring"]["verifier"], cfg), os.path.join(out, "diag_cache.json"))
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"), newline="")))
    print("unit,arm,target,age_est,top3_race,eth_pass,gender_pass,age_pass")
    for r in rows:
        prof = {k: r[k] for k in ("ethnicity", "gender", "age", "body_type")}
        rd = sc.read(os.path.join(out, r["image"].replace("\\", os.sep)))
        rep = evaluate_profile(prof, rd, cfg)
        top3 = sorted(rd.race.items(), key=lambda kv: -kv[1])[:3]
        t3 = "; ".join(f"{k} {v:.0f}" for k, v in top3)
        ax = rep.axes
        print(f"{r['unit']},{r['arm']},{prof['ethnicity']}/{prof['gender']}/{prof['age']},"
              f"{'' if rd.age is None else round(rd.age)},\"{t3}\","
              f"{int(bool(ax['ethnicity'].passed))},{int(bool(ax['gender'].passed))},{int(bool(ax['age'].passed))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/pilot")
