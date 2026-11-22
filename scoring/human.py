"""
Human rating workflow - the ground truth the automatic scorers are validated against.

1. export_rating_sheet(items, out_dir)   -> BLIND sheet (no target shown) + key
2. raters fill perceived_* columns in their own copy of the sheet
3. import_ratings([...csv]) + agreement(...)  -> kappa per attribute
4. human_fidelity(ratings, key)           -> pass rates against the targets

Raters must NOT see the target profile (that is exactly the leading-judge
problem the old LLaVA scorer had), hence the separate key file.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict

SHEET_COLUMNS = ["image_id", "image_path", "perceived_gender", "perceived_ethnicity",
                 "perceived_age_decade", "perceived_body_type", "face_visible", "notes"]


def export_rating_sheet(items, out_dir: str, seed: int = 0):
    """items: iterable of dicts {image_id, image_path, profile:{...}}. Rows are shuffled."""
    import random
    os.makedirs(out_dir, exist_ok=True)
    items = list(items)
    random.Random(seed).shuffle(items)
    sheet = os.path.join(out_dir, "rating_sheet_blind.csv")
    key = os.path.join(out_dir, "rating_key_DO_NOT_SHARE.csv")
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, SHEET_COLUMNS)
        w.writeheader()
        for it in items:
            w.writerow({"image_id": it["image_id"], "image_path": it["image_path"]})
    with open(key, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "gender", "ethnicity", "age", "body_type"])
        for it in items:
            p = it["profile"]
            w.writerow([it["image_id"], p.get("gender"), p.get("ethnicity"),
                        p.get("age"), p.get("body_type")])
    return sheet, key


def import_ratings(paths):
    """-> {rater_name: {image_id: row}} (rater = file stem)."""
    out = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        with open(p, newline="") as f:
            out[name] = {r["image_id"]: r for r in csv.DictReader(f) if r.get("image_id")}
    return out


def load_key(path):
    with open(path, newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def cohen_kappa(a, b):
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(ratings: dict, columns=("perceived_gender", "perceived_ethnicity",
                                      "perceived_age_decade", "perceived_body_type")):
    """Pairwise Cohen kappa between every two raters, per attribute."""
    names = sorted(ratings)
    res = defaultdict(dict)
    for col in columns:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                common = [k for k in ratings[names[i]] if k in ratings[names[j]]
                          and ratings[names[i]][k].get(col) and ratings[names[j]][k].get(col)]
                if len(common) >= 5:
                    res[col][f"{names[i]}~{names[j]}"] = round(cohen_kappa(
                        [ratings[names[i]][k][col] for k in common],
                        [ratings[names[j]][k][col] for k in common]), 3)
    return dict(res)


def human_fidelity(ratings: dict, key: dict):
    """Majority-vote human label per image vs target. Returns per-attribute match rate."""
    cols = {"gender": "perceived_gender", "ethnicity": "perceived_ethnicity",
            "age": "perceived_age_decade", "body_type": "perceived_body_type"}
    votes = defaultdict(lambda: defaultdict(list))
    for r in ratings.values():
        for iid, row in r.items():
            for attr, col in cols.items():
                if row.get(col):
                    votes[iid][attr].append(row[col].strip())
    stats = defaultdict(lambda: [0, 0])
    for iid, k in key.items():
        for attr in cols:
            v = votes[iid][attr]
            if not v or not k.get(attr):
                continue
            maj = Counter(v).most_common(1)[0][0]
            stats[attr][0] += int(maj.lower() == k[attr].lower())
            stats[attr][1] += 1
    return {a: {"match": m, "n": n, "rate": (m / n if n else None)} for a, (m, n) in stats.items()}
