"""
Parser accuracy report on the labelled brief set (tests/parser_cases.py).

    python -m eval.parser_accuracy              # prints a report
    python -m eval.parser_accuracy --json out.json

Reports exact-profile accuracy plus per-field precision/recall. A "false
positive" is a field the parser set that the label says should be absent -
the failure mode of the old keyword parser ("black blazer" -> African American).

NOTE: the bundled cases were written alongside the parser, so this is a
regression suite, not an independent estimate. Re-run on briefs written blind.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from profiler.text_parser import parse_brief_detailed
from tests.parser_cases import ALL_CASES

FIELDS = ["ethnicity", "gender", "age", "body_type", "region"]


def evaluate(cases=ALL_CASES, cfg=None):
    stats = defaultdict(lambda: defaultdict(int))
    failures = []
    for kind, brief, exp, extras in cases:
        res = parse_brief_detailed(brief, cfg)
        got = res.profile()
        ok = got == exp
        stats[kind]["n"] += 1
        stats[kind]["exact"] += int(ok)
        for f in FIELDS:
            e, g = exp.get(f), got.get(f)
            s = stats[kind]
            if e is not None and g == e:
                s[f + "_tp"] += 1
            if e is not None and g is not None and g != e:
                s[f + "_wrong"] += 1
            if e is not None and g is None:
                s[f + "_missed"] += 1
            if e is None and g is not None:
                s[f + "_fp"] += 1
        notes = []
        if "warn" in extras:
            blob = " ".join(res.warnings + res.unsupported).lower()
            if extras["warn"].lower() not in blob:
                ok = False
                notes.append(f"missing warning containing {extras['warn']!r}")
        if "amb" in extras and extras["amb"] not in res.ambiguous:
            ok = False
            notes.append(f"expected ambiguity on {extras['amb']!r}")
        if not ok:
            failures.append({"kind": kind, "brief": brief, "expected": exp,
                             "got": got, "notes": notes})
    return stats, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    stats, failures = evaluate(cfg=cfg)
    total = defaultdict(int)
    for kind, s in stats.items():
        n = s["n"]
        exact = n - sum(1 for f in failures if f["kind"] == kind)
        print(f"{kind:10s} n={n:3d}  fully correct (fields + warnings): {exact}/{n} = {exact / n:.1%}")
        for k, v in s.items():
            total[k] += v
    print("\nper-field (all cases):")
    for f in FIELDS:
        tp, fp = total[f + "_tp"], total[f + "_fp"]
        wrong, miss = total[f + "_wrong"], total[f + "_missed"]
        print(f"  {f:10s} correct={tp:3d} wrong={wrong:2d} missed={miss:2d} false_positive={fp:2d}")
    print(f"\n{len(failures)} failing case(s)")
    for fl in failures:
        print(" -", fl["brief"][:70], "| exp", fl["expected"], "| got", fl["got"], fl["notes"])
    if a.json:
        json.dump({"stats": {k: dict(v) for k, v in stats.items()}, "failures": failures},
                  open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
