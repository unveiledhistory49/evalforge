"""Small-sample statistics from scratch: paired bootstrap CIs, exact McNemar,
win/loss/tie. All randomness flows through an explicit seeded RNG."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lo: float
    hi: float
    confidence: float
    resamples: int


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("no values")
    if not 0 <= q <= 1:
        raise ValueError(f"q must be in [0,1], got {q}")
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_mean_ci(
    rng: random.Random, values: list[float], *, resamples: int = 2000, confidence: float = 0.95
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the mean. Deterministic given rng."""
    if not values:
        raise ValueError("no values")
    if resamples < 100:
        raise ValueError("resamples must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    n = len(values)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(resamples)]
    means.sort()
    alpha = 1 - confidence
    return ConfidenceInterval(
        estimate=sum(values) / n,
        lo=_quantile(means, alpha / 2),
        hi=_quantile(means, 1 - alpha / 2),
        confidence=confidence,
        resamples=resamples,
    )


def paired_diff_ci(
    rng: random.Random,
    scores_a: list[float],
    scores_b: list[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Bootstrap CI for mean(a - b) on paired per-example scores."""
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("paired scores must be non-empty and equal length")
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    return bootstrap_mean_ci(rng, diffs, resamples=resamples, confidence=confidence)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant pairs (b = A-only
    correct, c = B-only correct) via the binomial tail. b+c == 0 → 1.0."""
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, float(2 * tail) / float(2**n))


def win_loss_tie(correct_a: list[bool], correct_b: list[bool]) -> dict[str, int]:
    if len(correct_a) != len(correct_b):
        raise ValueError("length mismatch")
    out = {"wins": 0, "losses": 0, "ties": 0}
    for a, b in zip(correct_a, correct_b, strict=True):
        if a and not b:
            out["wins"] += 1
        elif b and not a:
            out["losses"] += 1
        else:
            out["ties"] += 1
    return out
