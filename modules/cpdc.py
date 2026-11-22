"""
CPDC - Confidence-Proportional Demographic Correction (v2).

What changed from the first version, and why
--------------------------------------------
The original controller had three defects that inflated the reported gains:

1. No pass threshold. It only stopped on an axis when the scorer returned a
   perfect confidence, so almost every image was "corrected" (the earlier
   "69% needed correction / 22 of 36 used all iterations" were artefacts).
   -> A failing axis is now decided by the verifier (`FidelityReport`), and a
      passing image is left alone.
2. One-directional corrections. The skin-tone and age tokens always pushed
   "darker" and "older" whatever the target (a 20s East Asian target got
   "deep melanin-rich complexion" and "silver streaks in hair").
   -> Ethnicity/gender tokens are built from the TARGET profile, and age has an
      explicit direction (`younger` / `older`) taken from the signed estimate.
3. Error signal came from a VLM's self-reported score. -> Errors now come from
   classifier probabilities and an age estimate (scoring/fidelity.py).

Algorithm (per generation attempt k >= 1)
-----------------------------------------
  for every failing, automatically-scored axis a:
      e_a   = normalised error in [0, 1]                     (scoring)
      L_a   = clip(ceil(e_a * K), 1, K)                      (graduated level)
      stall = [prev_e_a - e_a < epsilon] for `stall_limit` consecutive rounds
      if stall: stop correcting a (diminishing returns)
  If no axis is left to correct and the image still fails, stop.

Flags let the ablation switch components off individually:
  proportional=False   -> level = attempt number (fixed escalation, not error-driven)
  direction_aware=False-> age always "older" (the original behaviour)
  early_stop=False     -> never stop an axis for stalling

No dependency on FLUX, DeepFace or any model: pure functions + dataclasses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

CORRECTABLE_AXES = ("ethnicity", "gender", "age")      # body_type has no verifier


def compute_error(confidence_target: float) -> float:
    """Error = 1 - P(target), clamped to [0, 1]."""
    return 1.0 - max(0.0, min(1.0, float(confidence_target)))


def select_level(error: float, num_levels: int = 4) -> int:
    """level = clip(ceil(error * K), 1, K); 0 only when error <= 0."""
    if error <= 0:
        return 0
    return max(1, min(num_levels, math.ceil(error * num_levels)))


@dataclass
class AxisTracker:
    """Error trajectory of one axis across attempts + stall detection."""
    axis: str
    epsilon: float = 0.03
    stall_limit: int = 2
    history: list = field(default_factory=list)
    stall_count: int = 0

    def update(self, error: float) -> bool:
        """Record an error reading. True => stop correcting this axis."""
        if self.history:
            improvement = self.history[-1] - error
            self.stall_count = self.stall_count + 1 if improvement < self.epsilon else 0
        self.history.append(error)
        return self.stall_count >= self.stall_limit

    @property
    def stopped(self) -> bool:
        return self.stall_count >= self.stall_limit


@dataclass
class Plan:
    """What to change for the next attempt."""
    levels: dict = field(default_factory=dict)   # axis -> level | ("younger"/"older", level)
    reason: str = ""                             # "correct" | "no_face" | "passed" | "stalled"
    stalled_axes: list = field(default_factory=list)

    @property
    def stop(self) -> bool:
        return self.reason in ("passed", "stalled")


class CorrectionController:
    def __init__(self, num_levels: int = 4, epsilon: float = 0.03, stall_limit: int = 2,
                 proportional: bool = True, direction_aware: bool = True,
                 early_stop: bool = True, max_axes: Optional[int] = 1,
                 level_cap: Optional[int] = None):
        # max_axes: correct at most this many failing axes per round (worst first).
        #   Pilot data showed that correcting several axes at once, or using the
        #   strongest wording, often damaged axes that were already fine.
        # level_cap: strongest wording allowed (None -> num_levels).
        self.max_axes = max_axes
        self.level_cap = min(level_cap, num_levels) if level_cap else num_levels
        self.K, self.epsilon, self.stall_limit = num_levels, epsilon, stall_limit
        self.proportional, self.direction_aware, self.early_stop = (
            proportional, direction_aware, early_stop)
        self.trackers = {a: AxisTracker(a, epsilon, stall_limit) for a in CORRECTABLE_AXES}

    @classmethod
    def from_config(cls, cfg: dict, **flags):
        lp = (cfg or {}).get("loop", {})
        kw = dict(max_axes=lp.get("max_axes_per_round", 1), level_cap=lp.get("level_cap", 3))
        kw.update(flags)
        return cls(lp.get("num_levels", 4), lp.get("epsilon", 0.03),
                   lp.get("stall_limit", 2), **kw)

    def plan(self, report, attempt_number: int) -> Plan:
        """`report` = FidelityReport of the attempt just made; attempt_number = the
        1-based index of the attempt that will be generated NEXT."""
        if report.all_passed:
            return Plan(reason="passed")
        if not report.face_found:
            return Plan(reason="no_face")        # resample; nothing attribute-specific to fix

        levels, stalled, cand = {}, [], []
        for ax in report.failing():
            if ax.axis not in CORRECTABLE_AXES:
                continue
            tr = self.trackers[ax.axis]
            stop = tr.update(ax.error)
            if stop and self.early_stop:
                stalled.append(ax.axis)
                continue
            cand.append(ax)
        cand.sort(key=lambda a: -a.error)                   # worst axis first
        if self.max_axes:
            cand = cand[:self.max_axes]
        for ax in cand:
            level = (select_level(ax.error, self.K) if self.proportional
                     else min(self.K, attempt_number))
            level = max(1, min(level, self.level_cap))
            if ax.axis == "age":
                direction = ("younger" if ax.direction > 0 else "older") \
                    if self.direction_aware else "older"
                levels["age"] = (direction, level)
            else:
                levels[ax.axis] = level
        if not levels:
            return Plan(reason="stalled", stalled_axes=stalled)
        return Plan(levels=levels, reason="correct", stalled_axes=stalled)
