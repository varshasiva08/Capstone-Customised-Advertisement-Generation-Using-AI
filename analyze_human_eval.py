r"""
analyze_human_eval.py - builds every result table for the human-validated evaluation.

Run from the adfidelity folder (venv active):

  python analyze_human_eval.py ^
      --results results\main\results.csv --attempts results\main\attempts.csv ^
      --ratings results\main\rating\ratings_person1.csv ^
      --key results\main\rating\rating_key_DO_NOT_SHARE.csv ^
      --map rating_map.csv --clip clip_scores.csv --out new_tables.md

Needs: pandas, numpy (scikit-learn is optional, only for the calibrator table).
Bootstrap: resamples whole PROFILES (profile_id), 5000 draws, fixed seed.
Age counts as a human match when the rater's decade is within 10 years of the target.
"""
import argparse, json, re
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/main/results.csv")
ap.add_argument("--attempts", default="results/main/attempts.csv")
ap.add_argument("--ratings", default="results/main/rating/ratings_person1.csv")
ap.add_argument("--key", default="results/main/rating/rating_key_DO_NOT_SHARE.csv")
ap.add_argument("--map", default="rating_map.csv")
ap.add_argument("--clip", default="clip_scores.csv")
ap.add_argument("--out", default="new_tables.md")
ap.add_argument("--boot", type=int, default=5000)
args = ap.parse_args()

ETHS = ["South Asian", "African American", "East Asian", "Latin American", "Middle Eastern", "White", "Southeast Asian"]
DFMAP = {"South Asian": "indian", "African American": "black", "East Asian": "asian",
         "Latin American": "latino hispanic", "Middle Eastern": "middle eastern", "White": "white"}
ARMS = ["vanilla", "resample", "cpdc"]
out = []


def md(df, index=False):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def auc(score, y):
    s, y = np.asarray(score, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# ---------------- load + join ----------------
res = pd.read_csv(args.results)
att = pd.read_csv(args.attempts)
rt = pd.read_csv(args.ratings)
key = pd.read_csv(args.key)
mp = pd.read_csv(args.map)
clip = pd.read_csv(args.clip)


def norm_img(p):
    p = str(p).replace("\\", "/")
    m = re.search(r"images/all/.*$", p)
    return m.group(0) if m else p


mp["image"] = mp.source.map(norm_img)
att["image"] = att.image.map(norm_img)
H = rt.merge(key, on="image_id").merge(mp[["image_id", "image"]], on="image_id")
H["t"] = H.age.str[:2].astype(int)
H["p"] = H.perceived_age_decade.str.extract(r"(\d+)")[0].astype(int)
H["h_e"] = (H.perceived_ethnicity == H.ethnicity).astype(int)
H["h_g"] = (H.perceived_gender == H.gender).astype(int)
H["h_a"] = ((H.p - H.t).abs() <= 10).astype(int)
H["h_a_exact"] = (H.p == H.t).astype(int)
H["h_eg"] = H.h_e * H.h_g
H["h_all"] = H.h_e * H.h_g * H.h_a

clip["image_id"] = clip.image_id
C = clip.merge(mp[["image_id", "image"]], on="image_id")
C["clip_top"] = C[[f"p_eth_{e}" for e in ETHS]].idxmax(axis=1).str.replace("p_eth_", "")
C["clip_g"] = np.where(C.p_gen_female > C.p_gen_male, "female", "male")

prof = res.drop_duplicates("unit")[["unit", "profile_id", "ethnicity", "gender", "age"]]
A = att.merge(prof, on="unit").merge(H[["image", "h_e", "h_g", "h_a", "h_a_exact", "h_eg", "h_all"]], on="image", how="left")
A = A.merge(C[["image", "clip_top", "clip_g"] + [f"p_eth_{e}" for e in ETHS]], on="image", how="left")
A["c_e"] = (A.clip_top == A.ethnicity).astype(int)
A["c_g"] = (A.clip_g == A.gender).astype(int)
A["c_eg"] = A.c_e * A.c_g
A["d_eg"] = A.ethnicity_pass * A.gender_pass
A["d_all"] = A.ethnicity_pass * A.gender_pass * A.age_pass
A["pt"] = [r[f"p_eth_{r.ethnicity}"] for _, r in A.iterrows()]
U = A.drop_duplicates("image").copy()  # the rated images

F = res.merge(A[["unit", "arm", "attempt", "h_e", "h_g", "h_eg", "h_all", "c_eg", "d_eg", "d_all"]]
              .rename(columns={"attempt": "best_attempt"}).drop_duplicates(["unit", "arm", "best_attempt"]),
              on=["unit", "arm", "best_attempt"], how="left")
assert F.h_all.notna().all(), "some final images are missing human labels"

rng = np.random.default_rng(0)
PIDS = sorted(F.profile_id.unique())
BCOLS = ["d_all", "h_all", "c_eg", "h_eg"]
SUM = {a: {c: np.array([F[(F.arm == a) & (F.profile_id == p)][c].sum() for p in PIDS]) for c in BCOLS} for a in ARMS}
CNT = {a: np.array([((F.arm == a) & (F.profile_id == p)).sum() for p in PIDS], float) for a in ARMS}
IDX = rng.integers(0, len(PIDS), size=(args.boot, len(PIDS)))  # same resamples for every contrast


def contrast(col, a1, a2):
    def m(a):
        return SUM[a][col][IDX].sum(1) / CNT[a][IDX].sum(1)
    d = m(a1) - m(a2)
    pt = SUM[a1][col].sum() / CNT[a1].sum() - SUM[a2][col].sum() / CNT[a2].sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{100 * pt:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


# ---------------- T1 setup ----------------
n_units = res.unit.nunique()
rep = res.drop_duplicates("unit").groupby("ethnicity").agg(profiles=("profile_id", "nunique"), units=("unit", "nunique"))
out.append("### Table A. Experimental setup\n")
out.append(f"- Units (profile x repeat): **{n_units}** from **{len(PIDS)}** distinct profiles; arms: {', '.join(ARMS)}; "
           f"up to 4 generator calls per unit; attempt 0 is shared by the three arms.")
out.append(f"- Images rated by one human rater: **{len(H)}** (all unique attempt images).")
out.append("- Units per target ethnicity:\n")
out.append(md(rep, index=True) + "\n")

# ---------------- T2 arms under DeepFace verifier ----------------
g = F.groupby("arm")
T = pd.DataFrame({
    "images/unit": g.calls.mean().round(2),
    "ethnicity": g.verifier_ethnicity_pass.mean().map(pct),
    "gender": g.verifier_gender_pass.mean().map(pct),
    "age": g.verifier_age_pass.mean().map(pct),
    "all axes": g.verifier_all_pass.mean().map(pct),
}).loc[ARMS]
out.append("### Table B. Arm comparison graded by the DeepFace verifier (optimistic: the verifier also chose the final image)\n")
out.append(md(T, index=True) + "\n")

# ---------------- T3 arms under three graders ----------------
T = pd.DataFrame({
    "images/unit": g.calls.mean().round(2),
    "DeepFace: all axes": g.d_all.mean().map(pct),
    "Human: all axes (age +/-10y)": g.h_all.mean().map(pct),
    "DeepFace: eth+gender": g.d_eg.mean().map(pct),
    "CLIP: eth+gender": g.c_eg.mean().map(pct),
    "Human: eth+gender": g.h_eg.mean().map(pct),
}).loc[ARMS]
out.append("### Table C. The same final images graded by three different graders (n = %d units per arm)\n" % (n_units))
out.append(md(T, index=True) + "\n")
rows = []
for a1, a2 in [("resample", "vanilla"), ("cpdc", "vanilla"), ("cpdc", "resample")]:
    rows.append({"contrast": f"{a1} - {a2}",
                 "DeepFace all axes": contrast("d_all", a1, a2),
                 "Human all axes": contrast("h_all", a1, a2),
                 "CLIP eth+gender": contrast("c_eg", a1, a2),
                 "Human eth+gender": contrast("h_eg", a1, a2)})
out.append("Differences in percentage points with 95% CI (bootstrap over profiles):\n")
out.append(md(pd.DataFrame(rows)) + "\n")
pv = F.pivot_table(index="unit", columns="arm", values="h_all")
d = pv["cpdc"] - pv["resample"]
out.append(f"Units where CPDC differs from resample on human all-axes: better in **{int((d > 0).sum())}**, worse in **{int((d < 0).sum())}**, tied in **{int((d == 0).sum())}** of {len(d)}.\n")

# ---------------- T4 verifier audit by target ethnicity ----------------
rows = []
for e, s in list(U.groupby("ethnicity")) + [("All", U)]:
    rows.append({"target": e, "n images": len(s),
                 "human: reads as target": pct(s.h_e.mean(), 0),
                 "DeepFace passes": pct(s.ethnicity_pass.mean(), 0),
                 "CLIP passes": pct(s.c_e.mean(), 0),
                 "DeepFace agrees with human": pct((s.ethnicity_pass == s.h_e).mean(), 0),
                 "CLIP agrees with human": pct((s.c_e == s.h_e).mean(), 0)})
out.append("### Table D. Ethnicity verifiers against the human label, by target group\n")
out.append(md(pd.DataFrame(rows)) + "\n")
out.append(f"Reference: answering \"matches\" for every image would agree with the human on {pct(U.h_e.mean())} of images. "
           f"Cohen's kappa with the human: DeepFace {kappa(U.ethnicity_pass, U.h_e):.2f}, CLIP {kappa(U.c_e, U.h_e):.2f}.\n")

# ---------------- T4b gender and age axes ----------------
rows = []
for gname, s in list(U.groupby("gender")) + [("all", U)]:
    rows.append({"axis": f"gender (target: {gname})", "n images": len(s),
                 "human: matches": pct(s.h_g.mean(), 0), "DeepFace passes": pct(s.gender_pass.mean(), 0),
                 "CLIP matches": pct(s.c_g.mean(), 0)})
for aname, s in list(U.groupby("age")) + [("all", U)]:
    rows.append({"axis": f"age (target: {aname})", "n images": len(s),
                 "human: within 10 years": pct(s.h_a.mean(), 0), "DeepFace passes": pct(s.age_pass.mean(), 0),
                 "CLIP matches": "n/a"})
out.append("### Table D2. Gender and age axes against the human label\n")
out.append(md(pd.DataFrame(rows).fillna("")) + "\n")
out.append(f"Exact-decade age agreement between the human and the target: {pct(U.h_a_exact.mean(), 0)} "
           f"(within one decade: {pct(U.h_a.mean(), 0)}). CLIP has no age output.\n")

# ---------------- T5 South Asian detail ----------------
sa = U[U.ethnicity == "South Asian"]
yes, no = sa[sa.h_e == 1], sa[sa.h_e == 0]
dfp = sa.race_top3.map(lambda s: next((x[1] for x in json.loads(s) if x[0] == "indian"), 0) / 100)
rows = [
    {"verifier": "DeepFace (lenient South Asian rule)",
     "passes of human-yes": f"{int(yes.ethnicity_pass.sum())}/{len(yes)}",
     "passes of human-no": f"{int(no.ethnicity_pass.sum())}/{len(no)}",
     "AUC of P(target)": f"{auc(dfp, sa.h_e):.2f}", "kappa": f"{kappa(sa.ethnicity_pass, sa.h_e):.2f}"},
    {"verifier": "CLIP ViT-L/14 zero-shot",
     "passes of human-yes": f"{int(yes.c_e.sum())}/{len(yes)}",
     "passes of human-no": f"{int(no.c_e.sum())}/{len(no)}",
     "AUC of P(target)": f"{auc(sa.pt, sa.h_e):.2f}", "kappa": f"{kappa(sa.c_e, sa.h_e):.2f}"},
]
out.append("### Table E. South Asian target images: can the verifiers tell good from bad? (n = %d; human-yes %d, human-no %d)\n" % (len(sa), len(yes), len(no)))
out.append(md(pd.DataFrame(rows)) + "\n")
byp = no.groupby("profile_id").size()
out.append("Human-no South Asian images by profile id: " + ", ".join(f"p{int(k):03d} ({v})" for k, v in byp.items()) +
           " (profiles are 30s/50s female/male South Asian).\n")

# ---------------- T6 combined verifier (leave one profile out) ----------------
try:
    from sklearn.linear_model import LogisticRegression
    X = pd.DataFrame({
        "clip_pt": U.pt.values,
        "clip_marg": [r.pt - max(r[f"p_eth_{e}"] for e in ETHS if e != r.ethnicity) for _, r in U.iterrows()],
        "df_pt": [next((x[1] for x in json.loads(r.race_top3) if x[0] == DFMAP[r.ethnicity]), 0) / 100 for _, r in U.iterrows()],
        "df_pass": U.ethnicity_pass.values}, index=U.index)
    y = U.h_e.values
    gpid = U.profile_id.values
    saflag = (U.ethnicity == "South Asian").values

    def loo(cols):
        pr = np.zeros(len(U))
        for p in np.unique(gpid):
            tr, te = gpid != p, gpid == p
            Xv = X[cols].values
            m = LogisticRegression(C=1.0, max_iter=2000).fit(Xv[tr], y[tr])
            pr[te] = m.predict_proba(Xv[te])[:, 1]
        return pr

    def row(name, pred, score):
        return {"verifier": name, "accuracy (all 145)": pct((pred == y).mean()),
                "accuracy (South Asian)": pct((pred[saflag] == y[saflag]).mean()),
                "AUC (all)": f"{auc(score, y):.2f}", "AUC (South Asian)": f"{auc(score[saflag], y[saflag]):.2f}"}
    rows = [row("always \"matches\"", np.ones(len(U), int), np.zeros(len(U))),
            row("DeepFace pass/fail", U.ethnicity_pass.values, U.ethnicity_pass.values.astype(float)),
            row("CLIP top class = target", U.c_e.values, X.clip_pt.values)]
    for nm, cols in [("logistic: CLIP P(target)", ["clip_pt"]),
                     ("logistic: CLIP + DeepFace", ["clip_pt", "clip_marg", "df_pt", "df_pass"])]:
        pr = loo(cols)
        rows.append(row(nm + " (leave-one-profile-out)", (pr >= 0.5).astype(int), pr))
    out.append("### Table F. Can a combined verifier beat the raw ones? (ethnicity axis, leave-one-profile-out)\n")
    out.append(md(pd.DataFrame(rows)) + "\n")
except ImportError:
    out.append("### Table F skipped (install scikit-learn to build it)\n")

# ---------------- T7 failure analysis and cost ----------------
rows = []
for a in ARMS:
    s = F[F.arm == a]
    rows.append({"arm": a, "images generated": int(s.calls.sum()),
                 "DeepFace successes": int(s.d_all.sum()), "images per DeepFace success": f"{s.calls.sum() / s.d_all.sum():.2f}",
                 "human successes": int(s.h_all.sum()), "images per human success": f"{s.calls.sum() / s.h_all.sum():.2f}"})
out.append("### Table G. Cost per success (all axes; final selected image)\n")
out.append(md(pd.DataFrame(rows)) + "\n")
a0 = A[A.attempt == 0].drop_duplicates(["unit"])
fail0 = set(a0[a0.d_all == 0].unit)
rows = []
for grp, mask in [("South Asian targets", F.ethnicity == "South Asian"), ("other targets", F.ethnicity != "South Asian")]:
    for a in ARMS:
        s = F[(F.arm == a) & mask & F.unit.isin(fail0)]
        if len(s):
            rows.append({"group": grp, "arm": a, "units failing at attempt 0 (DeepFace)": len(s),
                         "fixed by the end (DeepFace)": f"{int(s.d_all.sum())} ({pct(s.d_all.mean(), 0)})",
                         "judged OK by human at the end": f"{int(s.h_all.sum())} ({pct(s.h_all.mean(), 0)})",
                         "mean images": f"{s.calls.mean():.2f}"})
out.append("### Table H. Recovery of units whose first image failed the DeepFace verifier\n")
out.append(md(pd.DataFrame(rows)) + "\n")
ef = set(a0[a0.ethnicity_pass == 0].unit)
r = A[(A.attempt >= 1) & A.unit.isin(ef)].copy()


def wording(row):
    if row.arm == "resample":
        return "resample (no wording change)"
    ap_ = json.loads(row.applied) if isinstance(row.applied, str) else {}
    return "cpdc, ethnicity wording applied" if "ethnicity" in ap_ else "cpdc, other axis wording"


r["kind"] = r.apply(wording, axis=1)
rows = []
for k, s in r.groupby("kind"):
    rows.append({"retry type": k, "retry images": len(s),
                 "ethnicity passes (DeepFace)": f"{int(s.ethnicity_pass.sum())} ({pct(s.ethnicity_pass.mean(), 0)})",
                 "human says ethnicity matches": f"{int(s.h_e.sum())} ({pct(s.h_e.mean(), 0)})"})
out.append("### Table I. Does ethnicity wording help? Retry images in units whose attempt-0 image failed DeepFace's ethnicity check\n")
out.append(md(pd.DataFrame(rows)) + "\n")
ax = A[(A.arm == "cpdc") & (A.attempt >= 1)].copy()
ax["axis"] = ax.applied.map(lambda s: next(iter(json.loads(s).keys()), "none") if isinstance(s, str) else "none")
rows = []
for k, s in ax.groupby("axis"):
    rows.append({"correction axis (cpdc rounds)": k, "rounds": len(s),
                 "ethnicity pass": pct(s.ethnicity_pass.mean(), 0), "gender pass": pct(s.gender_pass.mean(), 0),
                 "age pass": pct(s.age_pass.mean(), 0)})
out.append("DeepFace pass rate of the image produced in each CPDC correction round, by the axis that was corrected:\n")
out.append(md(pd.DataFrame(rows)) + "\n")

open(args.out, "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
print("\nSaved", args.out)
