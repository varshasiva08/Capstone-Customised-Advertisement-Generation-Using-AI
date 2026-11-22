"""Match each rated image (rating/images/rXXXX.png) to the unit / arm / attempt that produced it,
by comparing file hashes with the images listed in attempts.csv. Read-only except for the output file.

Run from the project folder:
  python make_image_map.py results/extra1
  python make_image_map.py results/extra2
Writes  results/extra1/image_map.csv  (small file; send it instead of the images).
"""
import csv
import hashlib
import os
import sys

out = sys.argv[1]


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


rated = {}
rdir = os.path.join(out, "rating", "images")
for fn in sorted(os.listdir(rdir)):
    if fn.lower().endswith(".png"):
        rated[sha(os.path.join(rdir, fn))] = fn[:-4]

rows, found = [], set()
for r in csv.DictReader(open(os.path.join(out, "attempts.csv"), newline="")):
    p = os.path.join(out, r["image"].replace("\\", os.sep).replace("/", os.sep))
    if not os.path.exists(p):
        continue
    iid = rated.get(sha(p))
    if iid:
        found.add(iid)
        rows.append({"image_id": iid, "unit": r["unit"], "arm": r["arm"], "attempt": r["attempt"]})

dst = os.path.join(out, "image_map.csv")
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, ["image_id", "unit", "arm", "attempt"])
    w.writeheader()
    w.writerows(rows)
missing = sorted(set(rated.values()) - found)
print(f"{len(found)} of {len(rated)} rated images matched -> {dst}")
if missing:
    print("NOT matched:", missing)
