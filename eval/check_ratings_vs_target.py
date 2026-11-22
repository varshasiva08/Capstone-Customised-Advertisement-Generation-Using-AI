"""Compare one rater's labels with the generation targets (read-only; changes nothing).

Run from the project folder:
  python check_ratings_vs_target.py results/extra1/rating person1
  python check_ratings_vs_target.py results/extra2/rating person1

Needs, inside the rating folder:
  rating_key_DO_NOT_SHARE.csv   (image_id, gender, ethnicity, age, body_type)
  ratings_<name>.csv
Age counts as matching when within 10 years of the target decade (as in the paper).
Do NOT edit ratings after looking at this: report the numbers as they are.
"""
import csv
import os
import re
import sys

folder, name = sys.argv[1], sys.argv[2]
key = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(folder, "rating_key_DO_NOT_SHARE.csv"), newline=""))}
rat = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(folder, f"ratings_{name}.csv"), newline="")) if r.get("image_id")}


def dec(s):
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else None


rows = []
for iid, r in rat.items():
    k = key.get(iid)
    if not k:
        continue
    ta, ra = dec(k["age"]), dec(r["perceived_age_decade"])
    rows.append({
        "id": iid, "target": k,
        "gender": r["perceived_gender"] == k["gender"],
        "eth": r["perceived_ethnicity"] == k["ethnicity"],
        "age": ta is not None and ra is not None and abs(ta - ra) <= 10,
        "body": r["perceived_body_type"] == k["body_type"],
        "r": r,
    })
n = len(rows)
print(f"{n} rated images ({len(key)} in key, {len(rat)} rated)\n")
print("axis        match   % of images")
for ax in ("gender", "eth", "age", "body"):
    c = sum(x[ax] for x in rows)
    print(f"{ax:<10} {c:>5}   {100 * c / max(n, 1):5.1f}")
allm = sum(x["gender"] and x["eth"] and x["age"] for x in rows)
print(f"all (gender+eth+age) {allm:>3}   {100 * allm / max(n, 1):5.1f}\n")

print("by target ethnicity: n, ethnicity match %")
by = {}
for x in rows:
    by.setdefault(x["target"]["ethnicity"], []).append(x["eth"])
for e, v in sorted(by.items()):
    print(f"  {e:<18} n={len(v):<3} {100 * sum(v) / len(v):5.1f}%")

multi = [x for x in rows if "multi" in (x["r"].get("notes") or "").lower()]
print(f"\nimages noted 'multi': {len(multi)}")
print("\nMismatches (image, target -> rated):")
for x in rows:
    t, r = x["target"], x["r"]
    bad = []
    if not x["gender"]:
        bad.append(f"gender {t['gender']}->{r['perceived_gender']}")
    if not x["eth"]:
        bad.append(f"eth {t['ethnicity']}->{r['perceived_ethnicity']}")
    if not x["age"]:
        bad.append(f"age {t['age']}->{r['perceived_age_decade']}")
    if bad:
        print(f"  {x['id']}: " + "; ".join(bad))
