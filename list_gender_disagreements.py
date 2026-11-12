"""List images where your perceived gender differs from the requested gender.
Run from the adfidelity folder:
  python list_gender_disagreements.py
Look at each image again and, if you decide you mislabelled it, fix that row in
results\\main\\rating\\ratings_person1.csv (column perceived_gender). Then rerun analyze_human_eval.py.
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ratings", default=r"results\main\rating\ratings_person1.csv")
ap.add_argument("--key", default=r"results\main\rating\rating_key_DO_NOT_SHARE.csv")
ap.add_argument("--map", default="rating_map.csv")
a = ap.parse_args()

rt = pd.read_csv(a.ratings)
key = pd.read_csv(a.key)
mp = pd.read_csv(a.map)
d = rt.merge(key, on="image_id").merge(mp, on="image_id")
bad = d[d.perceived_gender != d.gender][["image_id", "source", "gender", "perceived_gender", "ethnicity", "age", "notes"]]
bad = bad.rename(columns={"gender": "requested_gender"})
print(bad.to_string(index=False))
print("\n%d disagreement(s) out of %d rated images" % (len(bad), len(d)))
