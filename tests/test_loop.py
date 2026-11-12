import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from modules.correction_loop import run_loop
from modules.cpdc import AxisTracker, CorrectionController, select_level
from modules.prompt_adapter import make_prompt_builder
from scoring import evaluate_profile
from scoring.types import FaceReading
from tests.sim import SimWorld

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))
# The simulator tests below were written for the strict (argmax) ethnicity rule.
SIM_CFG = None  # set after CFG is loaded
PROFILE = {"ethnicity": "South Asian", "gender": "female", "age": "50s", "body_type": "medium"}
BUILD = make_prompt_builder("White Blazer Suit", "Pure White", CFG)


def reading(age=55, eth="indian", p=90.0, gender="Woman"):
    other = "asian" if eth != "asian" else "black"
    return FaceReading(True, 0.99, (0, 0, 1, 1), age, {gender: 99.0, "Man" if gender == "Woman" else "Woman": 1.0},
                       {eth: p, other: 100 - p}, "t", 1)


# --- scoring ---------------------------------------------------------------
def test_passing_image_passes_all_scored_axes():
    rep = evaluate_profile(PROFILE, reading(), CFG)
    assert rep.all_passed and not rep.axes["body_type"].scored


def test_age_direction_and_tolerance():
    too_young = evaluate_profile(PROFILE, reading(age=38), CFG).axes["age"]
    assert not too_young.passed and too_young.direction == -1 and too_young.error > 0.5
    near = evaluate_profile(PROFILE, reading(age=48), CFG).axes["age"]
    assert near.passed and not near.strict                       # within 3y tolerance, not exact
    too_old = evaluate_profile({**PROFILE, "age": "20s"}, reading(age=45), CFG).axes["age"]
    assert not too_old.passed and too_old.direction == +1


def test_no_face_fails_everything():
    rep = evaluate_profile(PROFILE, FaceReading(False), CFG)
    assert not rep.all_passed and all(not a.passed for a in rep.scored_axes)


def test_wrong_ethnicity_fails():
    rep = evaluate_profile(PROFILE, reading(eth="black", p=90), CFG)
    assert not rep.axes["ethnicity"].passed


# --- controller --------------------------------------------------------------
def test_level_is_proportional_and_clipped():
    assert [select_level(e) for e in (0, 0.01, 0.25, 0.26, 0.5, 0.75, 1.0)] == [0, 1, 1, 2, 2, 3, 4]


def test_no_correction_when_passed():
    c = CorrectionController.from_config(CFG)
    plan = c.plan(evaluate_profile(PROFILE, reading(), CFG), 1)
    assert plan.reason == "passed" and not plan.levels


def test_age_direction_drives_token():
    c = CorrectionController.from_config(CFG)
    plan = c.plan(evaluate_profile(PROFILE, reading(age=33), CFG), 1)   # 50s target, looks 33
    assert plan.levels["age"][0] == "older"
    c2 = CorrectionController.from_config(CFG)
    plan2 = c2.plan(evaluate_profile({**PROFILE, "age": "20s"}, reading(age=47), CFG), 1)
    assert plan2.levels["age"][0] == "younger"


def test_without_direction_awareness_always_older():
    c = CorrectionController.from_config(CFG, direction_aware=False)
    plan = c.plan(evaluate_profile({**PROFILE, "age": "20s"}, reading(age=47), CFG), 1)
    assert plan.levels["age"][0] == "older"           # the original (wrong) behaviour


def test_stall_stops_axis():
    t = AxisTracker("age", epsilon=0.03, stall_limit=2)
    assert [t.update(e) for e in (0.9, 0.89, 0.9)] == [False, False, True]
    c = CorrectionController.from_config(CFG)
    reps = [evaluate_profile(PROFILE, reading(age=30), CFG) for _ in range(4)]
    kinds = [c.plan(r, i + 1).reason for i, r in enumerate(reps)]
    assert kinds[:2] == ["correct", "correct"] and kinds[2] == "stalled"


def test_proportional_flag_off_uses_attempt_number():
    c = CorrectionController.from_config(CFG, proportional=False)
    plan = c.plan(evaluate_profile(PROFILE, reading(age=30), CFG), 2)
    assert plan.levels["age"][1] == 2


# --- prompts -----------------------------------------------------------------
def test_correction_tokens_target_aware_in_prompt():
    p = BUILD(PROFILE, correction_levels={"ethnicity": 2, "age": ("older", 3)})
    assert "clearly South Asian woman" in p and "visible age lines" in p
    assert "melanin" not in p and "silver" not in p            # old colourist/one-way tokens gone
    m = BUILD({**PROFILE, "gender": "male"}, correction_levels={"gender": 2})
    assert "clearly a man" in m and "man in his 50s" in m


# --- loop policies -------------------------------------------------------------
SIM_CFG = {**CFG, "scoring": {**CFG["scoring"], "lenient_ethnicity_min_prob": {}}}


def _run(policy, world=None, **kw):
    world = world or SimWorld(CFG)
    v = world.make_verifier(PROFILE)
    r = run_loop(PROFILE, policy=policy, build_prompt_fn=BUILD, generate_fn=world.generate,
                 verifier=v, cfg=SIM_CFG, max_calls=4, base_seed=11, **kw)
    return r, world


def test_vanilla_is_one_call():
    r, w = _run("vanilla")
    assert r.calls == 1 == w.calls and r.stop_reason == "single_call"


def test_best_of_n_spends_full_budget_and_same_prompt():
    r, w = _run("best_of_n")
    assert r.calls == 4 and len({a.prompt for a in r.attempts}) == 1


def test_resample_stops_on_pass_and_never_changes_prompt():
    r, _ = _run("resample", SimWorld(CFG, age_bias=0.0, noise=1.0, eth_base=(0.9, 0.95)))
    assert r.calls == 1 and r.stop_reason == "passed"
    r2, _ = _run("resample")
    assert len({a.prompt for a in r2.attempts}) == 1


def test_cpdc_changes_prompt_in_right_direction_and_respects_budget():
    r, w = _run("cpdc")
    assert r.calls <= 4 and w.calls == r.calls
    corrected = [a for a in r.attempts[1:] if a.applied]
    assert corrected and all(a.applied.get("age", ("older",))[0] == "older" for a in corrected)


def test_cpdc_beats_resample_when_correction_works_and_not_when_it_does_not():
    """Control logic check on the simulator: if prompts have no effect, CPDC
    must not look better than plain resampling (paired over many profiles)."""
    wins = {"works": 0, "fails": 0}
    for label, works in (("works", True), ("fails", False)):
        for s in range(40):
            res = {}
            for pol in ("resample", "cpdc"):
                w = SimWorld(CFG, correction_works=works)
                v = w.make_verifier(PROFILE)
                r = run_loop(PROFILE, policy=pol, build_prompt_fn=BUILD, generate_fn=w.generate,
                             verifier=v, cfg=SIM_CFG, max_calls=4, base_seed=s * 17)
                res[pol] = r.best.report.fraction_passed
            wins[label] += res["cpdc"] > res["resample"]
    assert wins["works"] > wins["fails"]


def test_shared_first_attempt_is_reused_not_regenerated():
    w = SimWorld(CFG)
    img = w.generate("first", 5)
    r = run_loop(PROFILE, policy="cpdc", build_prompt_fn=BUILD, generate_fn=w.generate,
                 verifier=w.make_verifier(PROFILE), cfg=CFG, max_calls=3, base_seed=1,
                 first_attempt=(img, "first", 5))
    assert r.attempts[0].reused and r.attempts[0].prompt == "first"
    assert w.calls == 1 + (r.calls - 1)


def test_best_attempt_selected_by_verifier_score():
    r, _ = _run("best_of_n")
    assert r.best.score == max(a.score for a in r.attempts)


def test_unknown_policy_rejected():
    import pytest
    with pytest.raises(ValueError):
        _run("nonsense")


# --- one axis per round / level cap -----------------------------------------------
def test_one_axis_per_round_by_default_and_all_axes_ablation():
    bad = reading(age=30, gender="Man")                        # age + gender both wrong
    rep = evaluate_profile(PROFILE, bad, CFG)
    assert len(rep.failing()) >= 2
    one = CorrectionController.from_config(CFG).plan(rep, 1)
    assert len(one.levels) == 1
    many = CorrectionController.from_config(CFG, max_axes=None).plan(rep, 1)
    assert len(many.levels) >= 2


def test_level_cap_limits_strength():
    rep = evaluate_profile(PROFILE, reading(age=20), CFG)
    plan = CorrectionController.from_config(CFG, level_cap=2).plan(rep, 1)
    assert max((v[1] if isinstance(v, tuple) else v) for v in plan.levels.values()) <= 2


def test_lenient_south_asian_rule():
    # indian is 2nd at 33%, latino hispanic on top: strict fails, lenient passes
    r = FaceReading(True, 0.99, (0, 0, 1, 1), 55.0, {"Woman": 99.0, "Man": 1.0},
                    {"latino hispanic": 46.0, "indian": 33.0, "asian": 21.0}, "t", 1)
    assert evaluate_profile(PROFILE, r, CFG).axes["ethnicity"].passed
    strict_cfg = {"scoring": {**CFG["scoring"], "lenient_ethnicity_min_prob": {}}}
    assert not evaluate_profile(PROFILE, r, strict_cfg).axes["ethnicity"].passed
    low = FaceReading(True, 0.99, (0, 0, 1, 1), 55.0, {"Woman": 99.0, "Man": 1.0},
                      {"latino hispanic": 48.0, "middle eastern": 37.0, "indian": 15.0}, "t", 1)
    assert not evaluate_profile(PROFILE, low, CFG).axes["ethnicity"].passed
