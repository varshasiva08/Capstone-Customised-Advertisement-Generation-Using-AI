"""
Turn a FaceReading + target profile into a FidelityReport.

Pure functions, no model imports - unit-testable with hand-made readings.

Pass criteria (all configurable in config.yaml -> scoring):
  gender     P(target gender) >= gender_pass_prob
  ethnicity  target's DeepFace class is the argmax AND P(target classes) >= ethnicity_pass_prob
  age        estimate within `age_tolerance_years` of the target decade
             (the estimator's own MAE is ~4-7 years; `strict` = exact decade)
  body_type  NOT automatically scored (no validated classifier). Rate it by
             humans (scoring/human.py).

`error` is a normalised 0..1 distance that the correction controller uses to
choose how strong a correction to apply; `direction` (age) says which way.
"""
from __future__ import annotations

from .types import AxisResult, FaceReading, FidelityReport

DEFAULTS = {
    "age_tolerance_years": 3,
    "age_error_scale_years": 15,
    "ethnicity_pass_prob": 0.25,
    "gender_pass_prob": 0.80,
    "race_map": {
        "South Asian": ["indian"], "East Asian": ["asian"], "Southeast Asian": ["asian"],
        "African American": ["black"], "Latin American": ["latino hispanic"],
        "Middle Eastern": ["middle eastern"], "White": ["white"],
    },
    "non_separable_ethnicities": ["East Asian", "Southeast Asian"],
    # Targets that may pass WITHOUT being the argmax if P(target) reaches this value.
    # Reason: DeepFace splits South Asian faces across indian / latino hispanic /
    # middle eastern, so "indian" is rarely the top class even on faces humans
    # call South Asian (pilot: top-1 in 2 of 12 images, 25-36% in 5 more).
    # Tuned on that small pilot; must be validated against human ratings and the
    # strict (argmax) rate is always reported alongside.
    "lenient_ethnicity_min_prob": {},
}


def _sc(cfg):
    s = dict(DEFAULTS)
    s.update((cfg or {}).get("scoring", cfg or {}) if isinstance(cfg, dict) else {})
    return s


def decade_bounds(age_label: str):
    lo = int(str(age_label).rstrip("s"))
    return lo, lo + 10


def _gender_axis(profile, r: FaceReading, sc) -> AxisResult:
    target = str(profile.get("gender", "female")).lower()
    key = "Woman" if target == "female" else "Man"
    total = sum(r.gender.values()) or 1.0
    p = r.gender.get(key, 0.0) / total
    pred = max(r.gender, key=r.gender.get) if r.gender else ""
    return AxisResult("gender", True, passed=p >= sc["gender_pass_prob"], strict=pred == key,
                      error=1.0 - p, target=target,
                      predicted={"Woman": "female", "Man": "male"}.get(pred, pred),
                      confidence=p)


def _ethnicity_axis(profile, r: FaceReading, sc) -> AxisResult:
    """Pass = the target's class is the argmax AND holds >= ethnicity_pass_prob.

    DeepFace's race head is diffuse for some groups (a correctly rendered South
    Asian or Black face often peaks at only 0.35-0.55), so the floor is low
    (chance for 6 classes is 0.17). The error used to size corrections is the
    RELATIVE deficit against the winning class, which is comparable across
    groups, rather than 1 - P(target).
    """
    target = profile.get("ethnicity", "")
    classes = sc["race_map"].get(target, [])
    total = sum(r.race.values()) or 1.0
    p = sum(r.race.get(c, 0.0) for c in classes) / total
    others = [v / total for k, v in r.race.items() if k not in classes]
    best_other = max(others) if others else 0.0
    pred = max(r.race, key=r.race.get) if r.race else ""
    is_argmax = pred in classes
    floor = sc["ethnicity_pass_prob"]
    if is_argmax:
        err = 0.0 if p >= floor else 0.5 * (floor - p) / floor
    else:
        err = (best_other - p) / best_other if best_other > 0 else 1.0
    note = ""
    if target in sc["non_separable_ethnicities"]:
        note = "DeepFace merges East and Southeast Asian into one class"
    strict_pass = bool(classes) and is_argmax and p >= floor
    lenient_min = (sc.get("lenient_ethnicity_min_prob") or {}).get(target)
    lenient_pass = bool(classes) and lenient_min is not None and p >= lenient_min
    if lenient_pass and not strict_pass:
        err = 0.0
        note = (note + "; " if note else "") + f"passed by lenient rule (P={p:.2f}, top class: {pred})"
    return AxisResult("ethnicity", True, passed=strict_pass or lenient_pass,
                      strict=is_argmax, error=max(0.0, min(1.0, err)), target=target,
                      predicted=pred, confidence=p, note=note)


def _age_axis(profile, r: FaceReading, sc) -> AxisResult:
    target = profile.get("age", "")
    lo, hi = decade_bounds(target)
    a = r.age if r.age is not None else (lo + hi) / 2
    if a < lo:
        off, direction = lo - a, -1
    elif a >= hi:
        off, direction = a - hi, +1
    else:
        off, direction = 0.0, 0
    tol = sc["age_tolerance_years"]
    err = min(1.0, off / sc["age_error_scale_years"])
    return AxisResult("age", True, passed=off <= tol, strict=off == 0, error=err,
                      direction=direction, target=target, predicted=f"{a:.0f}",
                      confidence=off,
                      note=("" if direction == 0 else
                            f"{off:.0f}y {'above' if direction > 0 else 'below'} the decade"))


def evaluate_profile(profile: dict, reading: FaceReading, cfg=None) -> FidelityReport:
    sc = _sc(cfg)
    if not reading.face_found:
        axes = {a: AxisResult(a, True, passed=False, strict=False, error=1.0,
                              target=str(profile.get(a, "")), note="no face found")
                for a in ("ethnicity", "gender", "age")}
    else:
        axes = {"ethnicity": _ethnicity_axis(profile, reading, sc),
                "gender": _gender_axis(profile, reading, sc),
                "age": _age_axis(profile, reading, sc)}
    axes["body_type"] = AxisResult("body_type", False, target=str(profile.get("body_type", "")),
                                   note="not automatically scored - use human rating")
    return FidelityReport(reading.face_found, axes, reading)


def selection_score(report: FidelityReport) -> float:
    """Scalar used to pick the best attempt: mean(1 - error) over scored axes,
    with an extra bonus per passed axis so a pass always beats a near-miss."""
    ax = report.scored_axes
    if not ax or not report.face_found:
        return -1.0
    return sum((1.0 - a.error) + (1.0 if a.passed else 0.0) for a in ax) / len(ax)
