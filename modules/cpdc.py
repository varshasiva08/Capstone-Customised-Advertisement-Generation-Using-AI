"""
modules/cpdc.py — Confidence-Proportional Demographic Correction (CPDC)

Replaces CDVR's binary pass/fail correction trigger with:
  1. A continuous error signal per axis, drawn from whichever scorer is
     empirically reliable on that axis (see docstrings below — this choice
     is justified by the CLIP-vs-LLaVA agreement analysis in
     EVALUATION_RESULTS.md, Table 4).
  2. A graduated correction level (not just mild/strong) proportional to
     that error.
  3. A diminishing-returns early-stop rule that halts correction on an axis
     once further attempts stop improving confidence, instead of always
     burning the full iteration budget.

This module has NO dependency on FLUX, LLaVA, Ollama, or the HF API — it
operates purely on confidence scores that the existing scorers already
produce. That means it can be unit-tested standalone (see bottom of file)
before ever being wired into the live generation loop.

Drop-in point: profiler/prompt_builder.py currently takes `correction_keys`
(a flat list) and `iteration` (used only to pick "mild" vs "strong").
CPDC replaces that iteration-based severity pick with `select_level()`
below, and the CDVR loop's flat threshold check
(`if ethnicity_score < 7: correction_keys.append("STF")`) is replaced with
`compute_error()` + `select_level()` + `should_stop()`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Number of graduated correction tokens available per axis. Must match the
# number of "level_N" entries you add under `corrections:` in config.yaml.
NUM_LEVELS = {
    "STF": 4,   # skin tone / ethnicity fix
    "AF":  4,   # age fix
    # BTF intentionally excluded — no trained confidence classifier exists
    # for body type, so it stays on the original binary rule.
}

# Minimum confidence improvement between rounds for a correction to be
# considered "working". Below this for CONSECUTIVE_STALL_LIMIT rounds in a
# row, CPDC stops trying rather than burning the remaining iteration budget.
EPSILON = 0.03
CONSECUTIVE_STALL_LIMIT = 2


# ---------------------------------------------------------------------------
# Error signal
# ---------------------------------------------------------------------------

def compute_error(confidence_target_class: float) -> float:
    c = confidence_target_class
    if c > 1.0:
        # LLaVA occasionally answers the confidence field on the same
        # 0-10 scale as the score field despite the prompt asking for
        # 0.0-1.0 - defensively normalize rather than silently misreading
        # e.g. "9.0" as near-total error.
        c = c / 10.0
    c = max(0.0, min(1.0, c))
    return 1.0 - c


# ---------------------------------------------------------------------------
# Graduated correction level
# ---------------------------------------------------------------------------

def select_level(axis: str, error: float) -> int:
    """
    Map a continuous error to a discrete correction level for `axis`.

    level = clip(ceil(error * K), 0, K), where K = NUM_LEVELS[axis].

    error=0.0        -> level 0 (no correction needed)
    error in (0,.25]  -> level 1 (mild, K=4 case)
    error in (.25,.5] -> level 2
    error in (.5,.75] -> level 3
    error in (.75,1]  -> level 4 (max correction on first attempt)

    Args:
        axis:  "STF" or "AF" (must be a key in NUM_LEVELS).
        error: output of compute_error().

    Returns:
        Integer level, 0..NUM_LEVELS[axis]. 0 means "don't correct".
    """
    if axis not in NUM_LEVELS:
        raise ValueError(f"No graduated ladder defined for axis '{axis}'. "
                          f"Known axes: {list(NUM_LEVELS)}")
    k = NUM_LEVELS[axis]
    if error <= 0:
        return 0
    import math
    level = math.ceil(error * k)
    return max(0, min(k, level))


def level_to_config_key(level: int) -> str:
    """Maps an integer level to the config.yaml key, e.g. 3 -> 'level_3'."""
    return f"level_{level}"


# ---------------------------------------------------------------------------
# Diminishing-returns early stop
# ---------------------------------------------------------------------------

@dataclass
class AxisTracker:
    """
    Tracks the error trajectory for one axis across CDVR iterations, and
    decides whether further correction attempts are worth spending a
    generation call on.

    Usage inside the CDVR loop:

        tracker = AxisTracker(axis="AF")
        for iteration in range(1, max_iterations + 1):
            ... generate image with correction level tracker.current_level() ...
            confidence = <scorer output for target class>
            stop = tracker.update(confidence)
            if stop:
                break   # correction-resistant; don't spend another call
    """
    axis: str
    history: list[float] = field(default_factory=list)   # error history
    stall_count: int = 0

    def update(self, confidence_target_class: float) -> bool:
        """
        Record a new confidence reading and decide whether to stop.

        Returns:
            True  -> stop correcting this axis (diminishing returns hit).
            False -> keep going.
        """
        error = compute_error(confidence_target_class)

        if self.history:
            delta = self.history[-1] - error   # positive = improving
            if delta < EPSILON:
                self.stall_count += 1
            else:
                self.stall_count = 0

        self.history.append(error)

        return self.stall_count >= CONSECUTIVE_STALL_LIMIT

    def current_level(self) -> int:
        """Correction level to apply on the NEXT attempt, based on the
        most recent error reading (or a cold start of 'unknown -> max
        caution', i.e. level determined once the first reading arrives)."""
        if not self.history:
            return 0
        return select_level(self.axis, self.history[-1])

    def wasted(self) -> bool:
        """True if this axis never improved enough to pass but we stopped
        anyway (diminishing returns) — the case the 88%-waste finding is
        about. Useful for the efficiency comparison table."""
        return self.stall_count >= CONSECUTIVE_STALL_LIMIT and (
            not self.history or self.history[-1] > 0.3   # still clearly failing
        )


# ---------------------------------------------------------------------------
# Standalone self-test — no external dependencies, run with:
#     python modules/cpdc.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== select_level sanity check (K=4) ===")
    for err in [0.0, 0.1, 0.26, 0.5, 0.51, 0.8, 1.0]:
        print(f"  error={err:.2f} -> level {select_level('AF', err)}")

    print("\n=== Simulated trajectory: a profile that stalls "
          "(mirrors the real South-Asian_slim_20s_seed42 case) ===")
    # Real data: age_conf went 0.376 -> ... -> 0.407 over 3 rounds, never
    # passing. Simulate CPDC watching that same trajectory.
    tracker = AxisTracker(axis="AF")
    for i, conf in enumerate([0.376, 0.390, 0.407], start=1):
        stop = tracker.update(conf)
        print(f"  iteration {i}: confidence={conf:.3f}  "
              f"next_level={tracker.current_level()}  stop={stop}")
    print(f"  wasted={tracker.wasted()}  "
          f"(blind escalation would have burned a 3rd generation call here; "
          f"CPDC stops after iteration 2)")

    print("\n=== Simulated trajectory: a profile that DOES converge ===")
    tracker2 = AxisTracker(axis="AF")
    for i, conf in enumerate([0.32, 0.55, 0.78], start=1):
        stop = tracker2.update(conf)
        print(f"  iteration {i}: confidence={conf:.3f}  "
              f"next_level={tracker2.current_level()}  stop={stop}")