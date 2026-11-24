"""Confidence intervals for the per-cell confabulation rates (a proportion).

We report both:
- ``wilson_ci``: the analytic Wilson score interval — the right default for a
  Bernoulli proportion at small n (well-behaved at 0/1, unlike normal-approx).
- ``bootstrap_ci``: percentile bootstrap over the per-instance bits — generalises
  to derived quantities (rate differences, decoy-share) and is what we cite when
  comparing two cells.
"""

from __future__ import annotations

import math
import random


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (95% at z=1.96)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(bits: list[bool] | list[int], n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of Bernoulli ``bits``.

    ``seed`` makes it deterministic/reproducible (important for a benchmark).
    """
    n = len(bits)
    if n == 0:
        return (0.0, 1.0)
    vals = [1 if b else 0 for b in bits]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def rate_ci(bits: list[bool] | list[int], method: str = "wilson") -> dict:
    """Point rate + CI for a list of per-instance bits, as a JSON-ready dict."""
    n = len(bits)
    k = sum(1 for b in bits if b)
    lo, hi = (wilson_ci(k, n) if method == "wilson" else bootstrap_ci(bits))
    return {"rate": round(k / n, 4) if n else None, "n": n, "k": k,
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "ci_method": method}
