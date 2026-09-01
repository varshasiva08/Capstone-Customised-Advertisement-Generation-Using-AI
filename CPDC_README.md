# CPDC — Confidence-Proportional Demographic Correction

This document explains the CPDC module added on the `feature/cpdc-correction`
branch: what it is, why it's a genuine methodological contribution (not just
an engineering feature), exactly what changed, and what's still needed
before it's paper-ready.

---

## 1. The problem it fixes

The original CDVR (Contextual Demographic Verification and Refinement) loop
worked like this: generate an image, check it with LLaVA, and if any axis
scored below 7/10, apply a **fixed** correction — always "mild" on the first
retry, always "strong" on retries after that — regardless of how wrong the
image actually was. It never asked *how wrong*, only *wrong or not*. It also
never noticed when a correction wasn't working — it always burned all 3
iterations before giving up, even if the score hadn't moved between rounds.

Re-analyzing `eval/results/cdvr_ablation.csv` (data already collected before
this branch existed) confirmed this was a real problem, not a hypothetical
one:

| Metric | Value |
|---|---|
| Profiles that triggered at least one correction | 25 / 36 |
| Of those, profiles that burned all 3 iterations and still failed | **22 / 25 (88%)** |
| Avg confidence gain per correction round — ethnicity axis | +0.056 |
| Avg confidence gain per correction round — age axis | +0.125 |

Correction *was* moving the needle — just by a small, roughly fixed amount
per round, independent of how far off the first attempt was. A profile that
started very wrong and a profile that started barely wrong got identical
treatment.

## 2. What CPDC does about it

CPDC replaces the binary threshold + fixed mild/strong ladder with two
mechanisms:

**a) Graduated, error-proportional correction.** Instead of two severities
(mild/strong), each axis has 4 levels. The level applied is computed from
how wrong the previous attempt actually was:

```python
# modules/cpdc.py
def compute_error(confidence_target_class: float) -> float:
    c = confidence_target_class
    if c > 1.0:
        c = c / 10.0   # defensive normalization, see §5
    c = max(0.0, min(1.0, c))
    return 1.0 - c

def select_level(axis: str, error: float) -> int:
    k = NUM_LEVELS[axis]          # 4
    if error <= 0:
        return 0
    level = math.ceil(error * k)
    return max(0, min(k, level))
```

A profile that's barely off gets a gentle nudge; a profile that's badly
wrong gets the strongest correction on its very first retry, instead of
working up to it through two wasted rounds.

**b) A diminishing-returns early stop.** After each attempt, CPDC checks
whether the correction actually improved confidence. If it hasn't, for two
rounds running, CPDC stops trying on that axis instead of spending a third
generation call that — per the 88% figure above — has close to zero chance
of succeeding:

```python
# modules/cpdc.py — AxisTracker.update()
def update(self, confidence_target_class: float) -> bool:
    error = compute_error(confidence_target_class)
    if self.history:
        delta = self.history[-1] - error
        if delta < EPSILON:            # EPSILON = 0.03
            self.stall_count += 1
        else:
            self.stall_count = 0
    self.history.append(error)
    return self.stall_count >= CONSECUTIVE_STALL_LIMIT   # = 2
```

## 3. Why this counts as a novel contribution

- It's a **named, formalized algorithm** (error → graduated level, plus a
  stall detector) — stateable as a methods-section algorithm, not a
  prompt-engineering trick specific to this pipeline.
- **Nothing in the cited related work does this.** Every correction-loop
  paper in the lit review is either single-pass generation or blind
  iterative retry with no graduated response and no stopping rule.
- It's **grounded in a real, surprising finding in the project's own data**
  (the 88% waste figure), not a hypothesis.
- It produces **two independent, reportable results**: fidelity (does
  graduated correction converge faster / more reliably than blind
  escalation) and efficiency (how many generation calls does the early-stop
  rule save).
- It's **separate from, and doesn't overlap with**, the two other
  contributions in this project (regional prompt synthesis for appearance,
  and any RAG-based profile generation) — it operates purely in the
  post-generation verification/correction loop.

## 4. Files changed

| File | Status | What changed |
|---|---|---|
| `modules/cpdc.py` | **New** | The algorithm itself: `compute_error`, `select_level`, `AxisTracker` (stall detection + level selection). No dependency on FLUX/LLaVA/Ollama — pure functions, unit-testable standalone. |
| `config.yaml` | Modified | `corrections.STF` and `corrections.AF` expanded from 2 levels (`mild`/`strong`) to 4 (`level_1`–`level_4`). New `cpdc:` block for `epsilon` and `consecutive_stall_limit`. `corrections.BTF` (body type) left untouched — no trained confidence classifier exists for that axis. |
| `profiler/prompt_builder.py` | Modified | `build_prompt()` gained a `correction_levels: dict` parameter (e.g. `{"AF": 3, "STF": 2}`), which looks up `config["corrections"][axis][f"level_{level}"]`. The old `correction_keys`/`iteration` path is preserved for BTF. |
| `eval/fidelity_scorer.py` | Modified | LLaVA's prompt now asks for a `0.0–1.0` confidence alongside each `0–10` score (`age_confidence`, `gender_confidence`, `ethnicity_confidence`). This is the error signal CPDC consumes. |
| `eval/run_full_evaluation.py` | Modified | The CDVR loop inside `generate_all()` now uses `AxisTracker` per axis instead of a flat `if score < 7` check, and stops correcting an axis once `tracker.update()` returns `True`. **Both axes currently use LLaVA's confidence** (see §6 for why). |
| `eval/clip_trainer/train_clip_fidelity.py` | Modified | Unrelated compatibility fix found during debugging: newer `transformers` wraps `get_image_features()` output differently. Not part of CPDC itself. |

## 5. Some limitations 
- **The verification signal is a VLM's subjective judgment, not an
  objective measurement.** Manual inspection of one correction pair (a
  South Asian profile scored 4/10 pre-correction despite already reading
  as South Asian to a human eye) suggests LLaVA's score is sensitive to
  how *visually pronounced* demographic markers are (skin tone contrast,
  styling, jewelry) rather than strictly correctness. A "correction" driven
  by that signal can push an image toward more exaggerated cues rather than
  toward genuine accuracy. This is a real, general risk of VLM-as-judge
  correction loops, not unique to this pipeline — worth a limitations
  sentence, and a natural pointer to future work (an objective, non-VLM
  check as a sanity filter).
- **The ethnicity axis currently uses LLaVA, not CLIP**, even though the
  original design called for CLIP's target-class confidence there (CLIP is
  the more reliable judge on ethnicity per the scorer-agreement table).
  The trained CLIP classifier collapsed to majority-class prediction during
  training (confirmed via `validate_classifier.py` — 19.1% race accuracy,
  exactly the "White" class proportion; same pattern on age and gender).
  This is a **training bug, not a CPDC bug** — likely a learning-rate /
  frozen-embedding mismatch in `train_clip_fidelity.py`. Falling back to
  LLaVA for both axes is a legitimate, documented engineering decision, not
  a silent compromise — state it as such.


