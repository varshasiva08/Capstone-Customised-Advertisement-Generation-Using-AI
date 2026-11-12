"""
Build the BLIND rating set from a finished experiment folder.

    python eval/make_rating_set.py results/pilot2 --max 200

Collects every distinct generated image (all attempts, not just the best),
de-duplicates identical files, samples evenly across ethnicities, copies them to
OUT/rating/images under anonymous ids, and writes
    OUT/rating/rating_sheet_blind.csv      (raters never see targets)
    OUT/rating/rating_key_DO_NOT_SHARE.csv (targets; keep away from raters)
Then run the rating app:   streamlit run eval/rating_app.py -- OUT/rating
"""
import argparse
import csv
import hashlib
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.human import export_rating_sheet


def _sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    res = list(csv.DictReader(open(os.path.join(a.out, "results.csv"), newline="")))
    prof = {r["unit"]: {k: r[k] for k in ("ethnicity", "gender", "age", "body_type")} for r in res}
    paths = []                                    # (unit, relative image path)
    att_file = os.path.join(a.out, "attempts.csv")
    if os.path.exists(att_file):
        for r in csv.DictReader(open(att_file, newline="")):
            if r.get("image"):
                paths.append((r["unit"], r["image"]))
    if not paths:                                 # older runs: only the best images were kept
        paths = [(r["unit"], r["image"]) for r in res]

    seen, items = set(), []
    for unit, rel in paths:
        full = os.path.join(a.out, rel.replace("\\", os.sep))
        if not os.path.exists(full):
            continue
        h = _sha(full)
        if h in seen:
            continue
        seen.add(h)
        items.append((unit, full))

    # even sampling across ethnicities
    by_eth = {}
    for unit, full in items:
        by_eth.setdefault(prof[unit]["ethnicity"], []).append((unit, full))
    rnd = random.Random(a.seed)
    for v in by_eth.values():
        rnd.shuffle(v)
    chosen, i = [], 0
    while len(chosen) < a.max and any(by_eth.values()):
        for k in list(by_eth):
            if by_eth[k] and len(chosen) < a.max:
                chosen.append(by_eth[k].pop())
    rdir = os.path.join(a.out, "rating")
    shutil.rmtree(rdir, ignore_errors=True)
    os.makedirs(os.path.join(rdir, "images"))
    rows = []
    for n, (unit, full) in enumerate(chosen):
        iid = f"r{n:04d}"
        shutil.copy(full, os.path.join(rdir, "images", iid + ".png"))
        rows.append({"image_id": iid, "image_path": f"images/{iid}.png", "profile": prof[unit]})
    export_rating_sheet(rows, rdir, seed=a.seed)
    print(f"{len(rows)} images ready in {rdir}")
    print("Raters:  streamlit run eval/rating_app.py -- " + rdir.replace("\\", "/"))


if __name__ == "__main__":
    main()
