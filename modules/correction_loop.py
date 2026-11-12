"""
Closed-loop generation with interchangeable policies (the experiment arms).

    vanilla           1 call, no verification loop (the baseline every paper needs)
    best_of_n         N calls, SAME prompt, new seed each time; keep the best.
                      Spends the whole budget regardless of passing.
    resample          same prompt, new seed, stop at the first pass. This is the
                      control that isolates the effect of changing the PROMPT:
                      it gets every benefit of re-rolling the dice and none of
                      the correction.
    fixed_escalation  the original CDVR idea with a real pass threshold:
                      legacy target-agnostic tokens, mild -> strong by attempt.
    cpdc              error-proportional, direction-aware, early-stopping
                      correction (modules/cpdc.py).
    cpdc_no_stop / cpdc_no_direction / cpdc_fixed_level / cpdc_all_axes
                      single-component ablations of cpdc.

Every policy:
  * counts one `call` per generator invocation (the budget is `max_calls`),
  * may reuse a pre-made first attempt (`first_attempt`) so all arms are PAIRED
    on the same attempt-0 image even when the generator ignores seeds,
  * returns the BEST attempt by the verifier's selection score. Selection uses
    the verifier only - never the evaluator that will grade the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from modules.cpdc import CorrectionController
from scoring import evaluate_profile, selection_score

POLICIES = ("vanilla", "best_of_n", "resample", "fixed_escalation", "cpdc",
            "cpdc_no_stop", "cpdc_no_direction", "cpdc_fixed_level", "cpdc_all_axes")


@dataclass
class Attempt:
    index: int
    seed: int
    prompt: str
    image: object                    # PIL image
    report: object                   # FidelityReport (verifier)
    applied: dict = field(default_factory=dict)   # levels used to build THIS attempt's prompt
    reused: bool = False             # True when supplied as a shared first attempt

    @property
    def score(self) -> float:
        return selection_score(self.report)


@dataclass
class LoopResult:
    policy: str
    attempts: list
    best_index: int
    calls: int
    stop_reason: str

    @property
    def best(self) -> Attempt:
        return self.attempts[self.best_index]

    def to_dict(self):
        return {"policy": self.policy, "calls": self.calls, "best_index": self.best_index,
                "stop_reason": self.stop_reason,
                "attempts": [{"index": a.index, "seed": a.seed, "prompt": a.prompt,
                              "applied": {k: (list(v) if isinstance(v, tuple) else v)
                                          for k, v in a.applied.items()},
                              "score": a.score, "report": a.report.to_dict()}
                             for a in self.attempts]}


def _best(attempts):
    return max(range(len(attempts)), key=lambda i: (attempts[i].score, -i))


def run_loop(profile: dict, *, policy: str, build_prompt_fn: Callable,
             generate_fn: Callable, verifier, cfg: dict, max_calls: int = None,
             base_seed: int = 0, first_attempt: Optional[tuple] = None,
             on_attempt: Optional[Callable] = None) -> LoopResult:
    """
    build_prompt_fn(profile, correction_levels=None, legacy_keys=None, iteration=0) -> str
    generate_fn(prompt, seed) -> PIL.Image
    verifier.read(image) -> FaceReading
    first_attempt: optional (image, prompt, seed) reused as attempt 0.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
    max_calls = max_calls or cfg.get("loop", {}).get("max_calls", 4)
    if policy == "vanilla":
        max_calls = 1

    attempts, calls = [], 0
    ctrl = None
    if policy.startswith("cpdc"):
        flags = {"cpdc_no_stop": {"early_stop": False},
                 "cpdc_no_direction": {"direction_aware": False},
                 "cpdc_fixed_level": {"proportional": False},
                 "cpdc_all_axes": {"max_axes": None, "level_cap": cfg.get("loop", {}).get("num_levels", 4)}}.get(policy, {})
        ctrl = CorrectionController.from_config(cfg, **flags)

    levels, legacy_keys, stop_reason = None, None, "budget"

    for k in range(max_calls):
        seed = base_seed + 1000 * k
        if k == 0 and first_attempt is not None:
            image, prompt, seed = first_attempt
            reused = True
        else:
            prompt = build_prompt_fn(profile, correction_levels=levels,
                                     legacy_keys=legacy_keys, iteration=k)
            image = generate_fn(prompt, seed)
            reused = False
        calls += 1
        report = evaluate_profile(profile, verifier.read(image), cfg)
        att = Attempt(k, seed, prompt, image, report,
                      applied=dict(levels or {}) or ({"legacy": list(legacy_keys)} if legacy_keys else {}),
                      reused=reused)
        attempts.append(att)
        if on_attempt:
            on_attempt(att)

        if policy == "vanilla":
            stop_reason = "single_call"
            break
        if policy == "best_of_n":
            continue                                   # always spend the full budget
        if report.all_passed:
            stop_reason = "passed"
            break
        if k == max_calls - 1:
            break

        if policy == "resample":
            continue
        if policy == "fixed_escalation":
            failing = {a.axis for a in report.failing()}
            keys = []
            if "ethnicity" in failing:
                keys.append("STF")
            if "age" in failing:
                keys.append("AF")
            # body type has no verifier => never part of the verified loop
            legacy_keys = keys or None
            levels = None
            if not keys:                               # gender-only/no-face failure: just resample
                legacy_keys = None
            continue
        # cpdc family
        plan = ctrl.plan(report, attempt_number=k + 1)
        if plan.stop:
            stop_reason = plan.reason
            break
        levels, legacy_keys = (plan.levels or None), None

    return LoopResult(policy, attempts, _best(attempts), calls, stop_reason)
