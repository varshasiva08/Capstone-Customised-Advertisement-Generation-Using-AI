"""Agreement between two raters (Cohen's kappa) on gender, ethnicity and age.
Run from the adfidelity folder after the second rater has finished:
  python second_rater_kappa.py --r1 results\\main\\rating\\ratings_person1.csv --r2 ratings_person2.csv
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True)
ap.add_argument("--r2", required=True)
a = ap.parse_args()


def kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = (x == y).mean()
    labs = sorted(set(x) | set(y))
    pe = sum((x == l).mean() * (y == l).mean() for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def dec(s):
    return s.astype(str).str.extract(r"(\d+)")[0].astype(float)


m = pd.read_csv(a.r1).merge(pd.read_csv(a.r2), on="image_id", suffixes=("_1", "_2"))
print("images rated by both:", len(m))
rows = []
for name, c in [("gender", "perceived_gender"), ("ethnicity", "perceived_ethnicity")]:
    x, y = m[c + "_1"], m[c + "_2"]
    rows.append((name, "%.1f%%" % (100 * (x == y).mean()), "%.2f" % kappa(x, y)))
a1, a2 = dec(m.perceived_age_decade_1), dec(m.perceived_age_decade_2)
ok = a1.notna() & a2.notna()
rows.append(("age (exact decade)", "%.1f%%" % (100 * (a1[ok] == a2[ok]).mean()), "%.2f" % kappa(a1[ok], a2[ok])))
rows.append(("age (within 10 years)", "%.1f%%" % (100 * ((a1[ok] - a2[ok]).abs() <= 10).mean()), "n/a"))
print("\n| axis | raw agreement | Cohen's kappa |\n|---|---|---|")
for r in rows:
    print("| %s | %s | %s |" % r)
print("\nEthnicity disagreements by rater-1 label:")
dis = m[m.perceived_ethnicity_1 != m.perceived_ethnicity_2]
print(dis.groupby("perceived_ethnicity_1").size().to_string())
