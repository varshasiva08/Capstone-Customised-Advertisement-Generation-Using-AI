"""Small, dependency-light statistics helpers used by the experiment runner."""
from __future__ import annotations

import math
import random
from typing import Sequence


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                     seed: int = 0, alpha: float = 0.05):
    """Mean of (a-b) with a percentile CI, resampling PAIRS (the unit is the profile)."""
    assert len(a) == len(b) and len(a) > 0
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean_diff": sum(d) / n, "ci_low": lo, "ci_high": hi, "n": n,
            "excludes_zero": lo > 0 or hi < 0}


def mcnemar_exact(b_only: int, c_only: int):
    """Exact two-sided McNemar on discordant pairs (b_only: A pass & B fail; c_only: reverse)."""
    n = b_only + c_only
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(b_only, c_only), n, 0.5).pvalue)


def cluster_note(n_units: int, n_images: int) -> str:
    return (f"{n_images} images from {n_units} independent profiles: resample profiles, "
            "not images, or CIs will be too narrow.")
