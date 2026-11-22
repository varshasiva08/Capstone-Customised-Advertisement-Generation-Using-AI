"""eval/analyze_final.py - every table and number for the paper, from the raw files.

Run from the project folder (adfidelity):
    python eval/analyze_final.py

Expected files
  results/main|extra1|extra2/results.csv, attempts.csv
  results/main|extra1|extra2/rating/ratings_person1.csv  and  ratings_person2.csv
  rating_map.csv (main)        results/extra1|extra2/image_map.csv  (make_image_map.py)
  rater3/ratings_person3.csv   rater3_key_PRIVATE.csv           (gender tie-break, optional)

Writes results/final/*.csv and results/final/tables.md. Nothing else is modified.
Units: a "profile" is the resampling unit (pairs of arms are resampled by profile).
Human "all axes" = gender + ethnicity match the target and apparent decade within 10 years.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.stats import mcnemar_exact, paired_bootstrap  # noqa: E402

BATCHES = ["main", "extra1", "extra2"]
OUT = os.path.join(ROOT, "results", "final")
os.makedirs(OUT, exist_ok=True)
ARMS = ["vanilla", "resample", "cpdc"]
DEF = dict(g="female", e="South Asian", a="20s", b="slim")  # first option of each rating-app question


def P(*p):
    return os.path.join(ROOT, *p)


def need(path):
    if not os.path.exists(path):
        sys.exit("Missing file: " + path)
    return path


def dec(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"(\d+)")[0], errors="coerce").values


def kappa(u, v):
    u, v = np.asarray(u), np.asarray(v)
    po = (u == v).mean()
    pe = sum((u == l).mean() * (v == l).mean() for l in set(u) | set(v))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kappa_ci(u, v, pids, B=3000, seed=1):
    u, v, pids = np.asarray(u), np.asarray(v), np.asarray(pids)
    ids = np.unique(pids)
    ix = {i: np.where(pids == i)[0] for i in ids}
    rng, out = np.random.default_rng(seed), []
    for _ in range(B):
        s = np.concatenate([ix[i] for i in rng.choice(ids, len(ids))])
        out.append(kappa(u[s], v[s]))
    return np.nanpercentile(out, [2.5, 97.5])


def fmt_ci(lo, hi):
    return "[%.2f, %.2f]" % (lo, hi)


# ----------------------------------------------------------------- load
res, att, amap, hum = [], [], [], []
for b in BATCHES:
    r = pd.read_csv(need(P("results", b, "results.csv")))
    r["batch"] = b
    res.append(r)
    a = pd.read_csv(need(P("results", b, "attempts.csv")))
    a["batch"] = b
    att.append(a)
    if b == "main":
        m = pd.read_csv(need(P("rating_map.csv")))
        m["base"] = m.source.str.replace("\\", "/").str.split("/").str[-1]
        a2 = a.copy()
        a2["base"] = a2.image.str.replace("\\", "/").str.split("/").str[-1]
        m = m.merge(a2[["unit", "arm", "attempt", "base"]], on="base")[["image_id", "unit", "arm", "attempt"]]
    else:
        m = pd.read_csv(need(P("results", b, "image_map.csv")))
    m["batch"] = b
    amap.append(m)
    h1 = pd.read_csv(need(P("results", b, "rating", "ratings_person1.csv")))
    h2 = pd.read_csv(need(P("results", b, "rating", "ratings_person2.csv")))
    h = h1.merge(h2, on="image_id", suffixes=("_1", "_2"))
    h["batch"] = b
    hum.append(h)
res, att, amap, hum = (pd.concat(x, ignore_index=True) for x in (res, att, amap, hum))
res["pid"] = res.batch + "_" + res.profile_id.astype(str)
tgt = res.drop_duplicates(["batch", "unit"])[["batch", "unit", "pid", "ethnicity", "gender", "age", "body_type"]]
amap = amap.merge(att[["batch", "unit", "arm", "attempt", "ethnicity_pass", "gender_pass", "age_pass"]],
                  on=["batch", "unit", "arm", "attempt"]).merge(tgt, on=["batch", "unit"])
amap["uid"] = amap.batch + "_" + amap.image_id
im = amap.drop_duplicates("uid").merge(hum, on=["batch", "image_id"])
print("rated images:", len(im), im.batch.value_counts().to_dict())

# ------------------------------------------------- gender tie-break (rater 3)
im["g3"] = np.nan
k3, r3 = P("rater3_key_PRIVATE.csv"), P("rater3", "ratings_person3.csv")
if os.path.exists(k3) and os.path.exists(r3):
    t = pd.read_csv(r3).merge(pd.read_csv(k3), left_on="image_id", right_on="new_id")
    t["uid"] = t.batch + "_" + t.orig_id
    im = im.drop(columns="g3").merge(t[["uid", "perceived_gender"]].rename(columns={"perceived_gender": "g3"}),
                                     on="uid", how="left")
im["g_res"] = im.perceived_gender_1
dis = im.perceived_gender_1 != im.perceived_gender_2
for i in im.index[dis]:
    v = [im.at[i, "perceived_gender_1"], im.at[i, "perceived_gender_2"], im.at[i, "g3"]]
    v = [x for x in v if isinstance(x, str)]
    im.at[i, "g_res"] = max(set(v), key=v.count) if len(v) == 3 else np.nan  # unresolved stays missing
gt = im[dis][["uid", "gender", "perceived_gender_1", "perceived_gender_2", "g3", "g_res"]]
gt.columns = ["image", "target", "rater1", "rater2", "rater3", "majority"]
gt.to_csv(os.path.join(OUT, "gender_resolution.csv"), index=False)

# ------------------------------------------------------------ match flags
im["e1"] = im.perceived_ethnicity_1 == im.ethnicity
im["e2"] = im.perceived_ethnicity_2 == im.ethnicity
im["a1"] = np.abs(dec(im.perceived_age_decade_1) - dec(im.age)) <= 10
im["a2"] = np.abs(dec(im.perceived_age_decade_2) - dec(im.age)) <= 10
im["g"] = im.g_res == im.gender  # NaN (unresolved) -> False
im["all1"], im["all2"] = im.e1 & im.g & im.a1, im.e2 & im.g & im.a2
im["both"] = im.all1 & im.all2
im["df_eth"] = im.ethnicity_pass == 1
for k, (gc, ec, ac, bc) in {"1": ("perceived_gender_1", "perceived_ethnicity_1", "perceived_age_decade_1", "perceived_body_type_1"),
                             "2": ("perceived_gender_2", "perceived_ethnicity_2", "perceived_age_decade_2", "perceived_body_type_2")}.items():
    n = (im[gc] == DEF["g"]).astype(int) + (im[ec] == DEF["e"]) + (im[ac] == DEF["a"]) + (im[bc] == DEF["b"])
    im["sus" + k] = n >= 3  # 3+ answers equal the old pre-selected defaults

# ------------------------------------------------------ final images / arms
fin = res.merge(amap.drop_duplicates(["batch", "unit", "arm", "attempt"])[["batch", "unit", "arm", "attempt", "uid"]],
                left_on=["batch", "unit", "arm", "best_attempt"], right_on=["batch", "unit", "arm", "attempt"])
fin = fin.merge(im[["uid", "all1", "all2", "both", "sus1", "sus2"]], on="uid")
fin["df_all"] = fin.deepface_all_pass.astype(bool)
NU, NP, NI = fin[["batch", "unit"]].drop_duplicates().shape[0], fin.pid.nunique(), len(im)
print("units", NU, "profiles", NP, "images", NI)

GR = {"DeepFace": "df_all", "Rater 1": "all1", "Rater 2": "all2", "Both agree": "both"}
rows = []
for n, c in GR.items():
    rows.append([n] + [round(100 * fin[fin.arm == a][c].mean(), 1) for a in ARMS])
t_arms = pd.DataFrame(rows, columns=["grader"] + ARMS)
t_arms.loc[len(t_arms)] = ["mean generator calls"] + [round(fin[fin.arm == a].calls.mean(), 2) for a in ARMS]
t_arms.to_csv(os.path.join(OUT, "arms.csv"), index=False)


def contrasts(col, data, label):
    p = data.groupby(["pid", "arm"])[col].mean().unstack()
    out = []
    for a, b in [("cpdc", "vanilla"), ("cpdc", "resample"), ("resample", "vanilla")]:
        q = p[[a, b]].dropna()
        r = paired_bootstrap(q[a].tolist(), q[b].tolist(), n_boot=10000, seed=0)
        out.append([label, f"{a} - {b}", round(100 * r["mean_diff"], 1), round(100 * r["ci_low"], 1), round(100 * r["ci_high"], 1),
                    r["n"], "yes" if r["excludes_zero"] else "no"])
    return out


t_con = []
for n, c in GR.items():
    t_con += contrasts(c, fin.assign(**{c: fin[c].astype(float)}), n)
t_con = pd.DataFrame(t_con, columns=["grader", "contrast", "diff_pp", "ci_low", "ci_high", "profiles", "excludes_zero"])
t_con.to_csv(os.path.join(OUT, "contrasts.csv"), index=False)

# unit-level discordance (McNemar exact) for each grader
disc = []
for n, c in GR.items():
    u = fin.pivot_table(index=["batch", "unit"], columns="arm", values=c, aggfunc="first").astype(bool)
    for b in ["resample", "vanilla"]:
        x, y = int((u.cpdc & ~u[b]).sum()), int((~u.cpdc & u[b]).sum())
        disc.append([n, f"cpdc vs {b}", x, y, round(float(mcnemar_exact(x, y)), 3)])
pd.DataFrame(disc, columns=["grader", "comparison", "cpdc_only", "other_only", "mcnemar_p"]).to_csv(
    os.path.join(OUT, "discordant_units.csv"), index=False)

# by batch
t_batch = fin.groupby(["batch", "arm"])[["df_all", "all1", "all2", "both"]].mean().mul(100).round(1).unstack("arm")
t_batch.to_csv(os.path.join(OUT, "by_batch.csv"))

# sensitivity: drop each rater's suspected default-click rows
sens = []
for n, c, s in [("Rater 1", "all1", "sus1"), ("Rater 2", "all2", "sus2")]:
    d = fin.copy()
    d[c] = d[c].astype(float).where(~d[s])
    sens += contrasts(c, d, n + f" (drop {int(fin[s].sum())} suspect final images)")
t_sens = pd.DataFrame(sens, columns=t_con.columns)
t_sens.to_csv(os.path.join(OUT, "sensitivity_default_rows.csv"), index=False)

# ------------------------------------------------------ rater agreement
ag = []
for n, u, v in [("gender (raw labels)", im.perceived_gender_1, im.perceived_gender_2),
                ("ethnicity (8 labels)", im.perceived_ethnicity_1, im.perceived_ethnicity_2),
                ("age, exact decade", dec(im.perceived_age_decade_1), dec(im.perceived_age_decade_2))]:
    u, v = np.asarray(u), np.asarray(v)
    ci = kappa_ci(u, v, im.pid.values)
    ag.append([n, round(100 * (u == v).mean(), 1), round(kappa(u, v), 2), fmt_ci(*ci)])
ag.append(["age, within 10 years", round(100 * (np.abs(dec(im.perceived_age_decade_1) - dec(im.perceived_age_decade_2)) <= 10).mean(), 1), "", ""])
t_ag = pd.DataFrame(ag, columns=["label", "agreement_%", "kappa", "kappa_95CI_over_profiles"])
t_ag.to_csv(os.path.join(OUT, "rater_agreement.csv"), index=False)

grp = []
for gname, d in list(im.groupby("ethnicity")) + [("ALL", im)]:
    grp.append([gname, len(d), d.pid.nunique(), round(100 * d.e1.mean(), 1), round(100 * d.e2.mean(), 1),
                round(100 * (d.e1 == d.e2).mean(), 1), round(kappa(d.e1, d.e2), 2)])
t_grp = pd.DataFrame(grp, columns=["target", "images", "profiles", "rater1_match_%", "rater2_match_%", "raters_agree_%", "kappa_match"])
t_grp.to_csv(os.path.join(OUT, "match_by_group.csv"), index=False)

# ------------------------------------------------- verifier vs humans
ver = []
for gname, d in list(im.groupby("ethnicity")) + [("ALL", im)]:
    row = [gname, len(d), round(100 * d.df_eth.mean(), 1)]
    for r in ["e1", "e2"]:
        row += [round(100 * d[r].mean(), 1), round(100 * (d[r] == d.df_eth).mean(), 1), round(kappa(d[r], d.df_eth), 2)]
    ver.append(row)
t_ver = pd.DataFrame(ver, columns=["target", "images", "DeepFace_pass_%", "r1_match_%", "r1_agree_%", "r1_kappa",
                                   "r2_match_%", "r2_agree_%", "r2_kappa"])
t_ver.to_csv(os.path.join(OUT, "verifier_vs_humans.csv"), index=False)
sa = im[im.ethnicity == "South Asian"]
sa_lines = []
for r, nm in [("e1", "rater 1"), ("e2", "rater 2")]:
    ci = kappa_ci(sa[r].astype(int), sa.df_eth.astype(int), sa.pid, seed=3)
    sa_lines.append(f"South Asian, DeepFace vs {nm}: kappa {kappa(sa[r], sa.df_eth):.2f} {fmt_ci(*ci)} over {sa.pid.nunique()} profiles; "
                    f"DeepFace passes {int(sa[sa[r]].df_eth.sum())}/{int(sa[r].sum())} human-accepted and "
                    f"{int(sa[~sa[r]].df_eth.sum())}/{int((~sa[r]).sum())} human-rejected images")
sa_lines.append("human-human kappa on 'matches target' overall: %.2f; DeepFace-human: %.2f (r1), %.2f (r2)"
                % (kappa(im.e1, im.e2), kappa(im.e1, im.df_eth), kappa(im.e2, im.df_eth)))

# ------------------------------------------------------------- markdown
def md(df, index=False):
    d = df.reset_index() if index else df
    d = d.copy()
    d.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in d.columns]
    L = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    L += ["| " + " | ".join(str(x) for x in r) + " |" for r in d.values]
    return "\n".join(L)


nmaj = int((gt.majority == gt.target).sum()) if len(gt) else 0
txt = [f"# Final analysis ({NU} units, {NP} profiles, {NI} rated images)\n",
       "## Rater agreement\n", md(t_ag), "\n## Target match and agreement by target group\n", md(t_grp),
       "\n## Gender tie-break\n",
       f"{len(gt)} images where raters 1 and 2 disagreed on gender; majority (rater 3) equals the target on {nmaj} of {len(gt)}.\n",
       md(gt), "\n## All-axis success on final images, % (units)\n", md(t_arms),
       "\n## Contrasts, percentage points, 95% CI over profiles\n", md(t_con),
       "\n## Unit-level discordance (exact McNemar)\n", md(pd.read_csv(os.path.join(OUT, 'discordant_units.csv'))),
       "\n## By batch, % all axes\n", md(t_batch, index=True),
       "\n## Sensitivity: dropping suspected default-click rows\n", md(t_sens),
       "\n## DeepFace ethnicity verdict vs each rater\n", md(t_ver), "\n" + "\n".join("* " + s for s in sa_lines) + "\n"]
open(os.path.join(OUT, "tables.md"), "w").write("\n".join(txt))
print("\n".join(txt))
print("\nWritten to", OUT)
