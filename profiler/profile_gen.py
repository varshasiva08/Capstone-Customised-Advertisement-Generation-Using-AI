"""
Deterministic profile suggestions (replaces the phi3:mini "generate 6 diverse
profiles" call).

Given a brief, fix whatever the parser already found and fill the remaining
attributes with a COVERAGE-MAXIMISING sample: each new profile is the candidate
that adds the most not-yet-seen attribute levels (ties broken by a seeded RNG),
so a set of six covers as many ethnicities / ages / body types as possible.
Same brief + seed -> same suggestions. No model, no server.
"""
from __future__ import annotations

import itertools
import random
from collections import Counter

import yaml

from profiler.text_parser import parse_brief_detailed

FREE_FIELDS = ["ethnicity", "gender", "body_type", "age"]


def _load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def generate_profiles(brand_brief: str, n: int = 6, cfg: dict = None, seed: int = 0,
                      fixed: dict = None, **_ignored) -> list:
    """
    Args:
        brand_brief: brief text (parsed for any attributes it already states).
        n:           number of suggestions.
        fixed:       extra attributes to hold constant (e.g. from the UI).
        **_ignored:  model=/host=/rag_context= from the old LLM signature.
    Returns: list of profile dicts (ethnicity, gender, body_type, age).
    """
    cfg = cfg or _load_cfg()
    opts = cfg["options"]
    parsed = parse_brief_detailed(brand_brief, cfg).profile()
    held = {**parsed, **(fixed or {})}
    held.setdefault("gender", cfg.get("defaults", {}).get("gender", "female"))

    free = [f for f in FREE_FIELDS if f not in held]
    space = [dict(zip(free, combo)) for combo in itertools.product(*(opts[f] for f in free))] or [{}]
    rng = random.Random(f"{seed}|{brand_brief}")
    rng.shuffle(space)

    chosen, seen = [], {f: Counter() for f in free}
    while space and len(chosen) < n:
        # candidate that adds the most unseen levels; prefer the least-used levels
        best = min(space, key=lambda c: sum(seen[f][c[f]] for f in free))
        space.remove(best)
        for f in free:
            seen[f][best[f]] += 1
        chosen.append({**held, **best})
    # fill remaining slots (when the space is smaller than n) by repeating the cycle
    while len(chosen) < n and chosen:
        chosen.append(dict(chosen[len(chosen) % max(1, len(chosen))]))
    return chosen


if __name__ == "__main__":
    for p in generate_profiles("sportswear brand targeting diverse women"):
        print(p)
