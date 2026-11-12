"""
Experiment runner: compare correction policies on a demographic grid.

    # real run (needs generator credentials in .env; DeepFace weights download once)
    python -m eval.run_experiment --out outputs/exp1 --repeats 3

    # logic smoke test with the SIMULATOR (not evidence - labelled DRY-RUN)
    python -m eval.run_experiment --out outputs/dry --dry-run --repeats 2 --max-profiles 6

Design
------
* Unit = (profile, repeat). Attempt 0 is generated ONCE per unit and shared by
  every arm, so arms are paired on the same first image even when the provider
  ignores seeds (Cloudflare does).
* Budget = `loop.max_calls` generator calls per unit; calls are reported per arm.
* The verifier (drives the loop) and the evaluators (grade the result) are
  configured separately (config.yaml -> scoring). The summary warns when they
  coincide.
* Statistics treat the PROFILE as the independent unit (repeats of one profile
  are averaged first), paired bootstrap for differences, exact McNemar for
  "all axes pass".
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from eval.stats import mcnemar_exact, paired_bootstrap, wilson_ci
from modules.correction_loop import POLICIES, run_loop
from modules.prompt_adapter import make_prompt_builder
from scoring import evaluate_profile
from scoring.cache import CachedScorer
from scoring.registry import get_scorer

DEFAULT_ARMS = ["vanilla", "best_of_n", "resample", "fixed_escalation", "cpdc"]
ABLATION_ARMS = ["cpdc_no_stop", "cpdc_no_direction", "cpdc_fixed_level", "cpdc_all_axes"]


def build_grid(cfg, ethnicities, genders, bodies, ages):
    return [{"ethnicity": e, "gender": g, "body_type": b, "age": a}
            for e, g, b, a in itertools.product(ethnicities, genders, bodies, ages)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(POLICIES))
    ap.add_argument("--ablations", action="store_true", help="add the cpdc single-component ablations")
    ap.add_argument("--ethnicities", nargs="+", default=["South Asian", "East Asian", "African American"])
    ap.add_argument("--genders", nargs="+", default=["female"])
    ap.add_argument("--bodies", nargs="+", default=["slim", "medium", "plus-size"])
    ap.add_argument("--ages", nargs="+", default=["20s", "30s", "40s", "50s"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-profiles", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--generator", default=None, help="cloudflare | hf | mock")
    ap.add_argument("--dry-run", action="store_true", help="use the simulator (no models, no API)")
    ap.add_argument("--evaluators", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open("config.yaml"))
    sc = cfg["scoring"]
    arms = a.arms + (ABLATION_ARMS if a.ablations else [])
    if "vanilla" not in arms:
        arms = ["vanilla"] + arms
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    max_calls = a.max_calls or cfg["loop"]["max_calls"]

    grid = build_grid(cfg, a.ethnicities, a.genders, a.bodies, a.ages)
    if a.max_profiles:
        grid = grid[: a.max_profiles]

    # ---- backends ---------------------------------------------------------
    world = None
    if a.dry_run:
        from tests.sim import SimWorld
        world = SimWorld(cfg)
        evaluators = {"sim_eval": None}
        verifier_name = "sim"
    else:
        verifier_name = sc["verifier"]
        verifier = CachedScorer(get_scorer(verifier_name, cfg), os.path.join(a.out, "verifier_cache.json"))
        evaluators = {n: (verifier if n == verifier_name else
                          CachedScorer(get_scorer(n, cfg), os.path.join(a.out, f"{n}_cache.json")))
                      for n in (a.evaluators or sc["evaluators"]) if n != "human"}
    independent = verifier_name not in evaluators

    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dry_run": a.dry_run, "arms": arms,
            "grid_size": len(grid), "repeats": a.repeats, "max_calls": max_calls,
            "verifier": verifier_name, "evaluators": list(evaluators) + (["human (separate)"] if not a.dry_run else []),
            "evaluator_independent_of_verifier": independent,
            "generator": "simulator" if a.dry_run else (a.generator or cfg["generation"]["provider"]),
            "note": "DRY-RUN: simulator output is NOT evidence about any real model." if a.dry_run else ""}
    if not independent:
        print("WARNING: the verifier is also used as an evaluator - results are not independent.")

    rows, att_rows = [], []
    # resume: reload finished units from an earlier (interrupted) run in the same --out
    def _load(name):
        fp = os.path.join(a.out, name)
        if not os.path.exists(fp) or a.dry_run:
            return []
        with open(fp, newline="") as f:
            out = list(csv.DictReader(f))
        for r in out:                                         # restore numeric types
            for k, v in list(r.items()):
                try:
                    r[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                except (ValueError, AttributeError):
                    pass
        return out
    rows, att_rows = _load("results.csv"), _load("attempts.csv")
    done_units = {r["unit"] for r in rows}
    if done_units:
        print(f"Resuming: {len(done_units)} unit(s) already finished, skipping them.")

    def dump(name, data):
        if not data:
            return
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(data)

    stopped = None
    t0 = time.time()
    for pi, profile in enumerate(grid):
        if stopped:
            break
        build = make_prompt_builder("White Blazer Suit", "Pure White", cfg)
        if a.dry_run:
            generate_fn, vfy = world.generate, world.make_verifier(profile)
        else:
            from generator.adapter import make_generate_fn
            raw_gen = make_generate_fn(cfg, a.generator, profile)
            generate_fn, vfy = _cached_generate(raw_gen, os.path.join(a.out, "gen_cache")), verifier
        for rep in range(a.repeats):
            uid = f"p{pi:03d}_r{rep}"
            if uid in done_units:
                continue
            base_seed = a.seed + 100003 * pi + 101 * rep
            p0 = build(profile)
            unit_rows, unit_att = [], []
            try:
                img0 = generate_fn(p0, base_seed)
            except RuntimeError as e:
                stopped = str(e)[:300]
                break
            for arm in arms:
                try:
                    res = run_loop(profile, policy=arm, build_prompt_fn=build, generate_fn=generate_fn,
                                   verifier=vfy, cfg=cfg, max_calls=max_calls, base_seed=base_seed,
                                   first_attempt=(img0, p0, base_seed))
                    best = res.best
                    path = os.path.join(a.out, "images", f"{uid}_{arm}_best.png")
                    best.image.save(path)
                    row = {"unit": uid, "profile_id": pi, "repeat": rep, **profile, "arm": arm,
                           "calls": res.calls, "best_attempt": res.best_index,
                           "stop_reason": res.stop_reason, "image": os.path.relpath(path, a.out)}
                    for ax in ("ethnicity", "gender", "age"):
                        r = best.report.axes[ax]
                        row[f"verifier_{ax}_pass"] = int(bool(r.passed))
                    row["verifier_all_pass"] = int(best.report.all_passed)
                    # independent evaluators grade the saved file
                    for en, ev in evaluators.items():
                        if ev is None:                                    # simulator: reuse verifier readout
                            rep_e = best.report
                        else:
                            rep_e = evaluate_profile(profile, ev.read(path), cfg)
                        for ax in ("ethnicity", "gender", "age"):
                            r = rep_e.axes[ax]
                            row[f"{en}_{ax}_pass"] = int(bool(r.passed))
                            row[f"{en}_{ax}_strict"] = int(bool(r.strict))
                        row[f"{en}_all_pass"] = int(rep_e.all_passed)
                        row[f"{en}_face"] = int(rep_e.face_found)
                    unit_rows.append(row)
                    for at in res.attempts:
                        os.makedirs(os.path.join(a.out, "images", "all"), exist_ok=True)
                        rel = (f"images/all/{uid}_a0.png" if at.reused
                               else f"images/all/{uid}_{arm}_a{at.index}.png")
                        if not os.path.exists(os.path.join(a.out, rel)):
                            at.image.save(os.path.join(a.out, rel))
                        unit_att.append({"unit": uid, "arm": arm, "attempt": at.index, "reused": int(at.reused),
                                         "image": rel,
                                         "applied": json.dumps({k: list(v) if isinstance(v, tuple) else v
                                                                for k, v in at.applied.items()}),
                                         "score": round(at.score, 4),
                                       "age_est": (round(at.report.reading.age, 1) if at.report.reading.age is not None else ""),
                                       "race_top3": json.dumps(sorted(((k, round(v, 1)) for k, v in at.report.reading.race.items()), key=lambda kv: -kv[1])[:3]),
                                         **{f"{x}_pass": int(bool(at.report.axes[x].passed)) for x in ("ethnicity", "gender", "age")}})
                except RuntimeError as e:                    # e.g. daily quota exhausted
                    stopped = str(e)[:300]
                    break
            if stopped:
                break
            rows.extend(unit_rows)
            att_rows.extend(unit_att)
            done_units.add(uid)
            dump("results.csv", rows)                      # save after every finished unit
            dump("attempts.csv", att_rows)
            print(f"[{pi + 1}/{len(grid)}] {uid} done ({time.time() - t0:.0f}s)")

    if stopped:
        print("\nSTOPPED EARLY (generator error):", stopped)
        print("Finished units are saved. Rerun the SAME command later to resume.")
    if not rows:
        return None
    meta["stopped_early"] = bool(stopped)
    summ = summarise(rows, arms, list(evaluators), meta)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print_summary(summ)
    return summ


def _cached_generate(raw_gen, cache_dir):
    """Disk cache keyed by (prompt, seed): if a run is interrupted in the middle of a
    profile, the images already paid for are reused on the next run instead of being
    generated again (the loop replays the same prompts/seeds until it reaches new work)."""
    import hashlib
    from PIL import Image
    os.makedirs(cache_dir, exist_ok=True)

    def gen(prompt, seed):
        key = hashlib.sha1(f"{prompt}|{seed}".encode()).hexdigest()
        fp = os.path.join(cache_dir, key + ".png")
        if os.path.exists(fp):
            return Image.open(fp).convert("RGB")
        img = raw_gen(prompt, seed)
        img.save(fp)
        return img
    return gen


def summarise(rows, arms, evaluators, meta):
    out = {"meta": meta, "per_arm": {}, "paired": {}}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        n = len(rs)
        entry = {"n_units": n, "mean_calls": sum(r["calls"] for r in rs) / n}
        for en in evaluators:
            for ax in ("ethnicity", "gender", "age", "all"):
                k = sum(r[f"{en}_{ax}_pass"] for r in rs)
                lo, hi = wilson_ci(k, n)
                entry[f"{en}_{ax}"] = {"k": k, "rate": k / n, "ci95": [lo, hi]}
        out["per_arm"][arm] = entry
    # paired comparisons (profile = independent unit)
    for en in evaluators:
        for base in ("vanilla", "resample", "best_of_n", "fixed_escalation"):
            if base not in arms or "cpdc" not in arms:
                continue
            for focus in ("cpdc",):
                by = {}
                for r in rows:
                    if r["arm"] in (focus, base):
                        by.setdefault((r["profile_id"], r["arm"]), []).append(r[f"{en}_all_pass"])
                pids = sorted({p for p, _ in by})
                a_ = [sum(by[(p, focus)]) / len(by[(p, focus)]) for p in pids]
                b_ = [sum(by[(p, base)]) / len(by[(p, base)]) for p in pids]
                unit_a = [r[f"{en}_all_pass"] for r in rows if r["arm"] == focus]
                unit_b = [r[f"{en}_all_pass"] for r in rows if r["arm"] == base]
                b_only = sum(1 for x, y in zip(unit_a, unit_b) if x and not y)
                c_only = sum(1 for x, y in zip(unit_a, unit_b) if y and not x)
                ca = sum(r["calls"] for r in rows if r["arm"] == focus) / len(unit_a)
                cb = sum(r["calls"] for r in rows if r["arm"] == base) / len(unit_b)
                out["paired"][f"{en}: {focus} vs {base}"] = {
                    "all_axes_pass_diff_by_profile": paired_bootstrap(a_, b_),
                    "mcnemar_p_units": mcnemar_exact(b_only, c_only),
                    "discordant_units": {f"{focus}_only": b_only, f"{base}_only": c_only},
                    "mean_calls": {focus: ca, base: cb}}
    return out


def print_summary(s):
    print("\n=== SUMMARY", "(DRY-RUN - simulator, not evidence)" if s["meta"]["dry_run"] else "", "===")
    evs = [k.split("_all")[0] for k in next(iter(s["per_arm"].values())) if k.endswith("_all")]
    for en in evs:
        print(f"\nEvaluator: {en}")
        print(f"{'arm':20s}{'calls':>7s}{'eth':>8s}{'gender':>8s}{'age':>8s}{'all axes':>10s}")
        for arm, e in s["per_arm"].items():
            print(f"{arm:20s}{e['mean_calls']:7.2f}" + "".join(f"{e[f'{en}_{x}']['rate']:8.1%}" for x in ("ethnicity", "gender", "age")) + f"{e[f'{en}_all']['rate']:10.1%}")
    for k, v in s["paired"].items():
        d = v["all_axes_pass_diff_by_profile"]
        print(f"{k}: diff={d['mean_diff']:+.3f} CI[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] "
              f"McNemar p={v['mcnemar_p_units']:.3f} calls {v['mean_calls']}")


if __name__ == "__main__":
    main()
