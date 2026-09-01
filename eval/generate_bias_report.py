# eval/generate_bias_report.py
import sys, os, csv
sys.path.insert(0, '.')
from modules.bias_tracker import BiasTracker

def to_score(val):
    if str(val).lower() == 'true':
        return 10.0
    if str(val).lower() == 'false':
        return 0.0
    try:
        return float(val) * 10
    except:
        return 0.0

tracker = BiasTracker(log_path="eval/bias_log.json")

csv_path = "eval/results/clip_fidelity_scores.csv"
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        profile = {
            "ethnicity": row.get("target_ethnicity", ""),
            "body_type": row.get("target_body_type", ""),
            "age":       row.get("target_age", ""),
            "gender":    "Female",
        }
        fidelity = {
            "age_score":       to_score(row.get("age_match", 0)),
            "gender_score":    to_score(row.get("gender_match", 0)),
            "ethnicity_score": to_score(row.get("ethnicity_match", 0)),
            "overall":         to_score(row.get("overall_fidelity", 0)),
        }
        tracker.log(
            profile=profile,
            image_path=row.get("image", ""),
            fidelity_scores=fidelity,
            cdvr_iterations=0,
        )

tracker.save("eval/results/bias_report.json")
print("Done.")