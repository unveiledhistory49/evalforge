"""Metrics, implemented from scratch (numpy only). Every metric is a pure
function of (expected, predicted) pairs with independently testable math."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class MetricError(ValueError):
    pass


def _labels(expected: Sequence[Any], predicted: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    if len(expected) != len(predicted):
        raise MetricError(f"length mismatch: {len(expected)} vs {len(predicted)}")
    if not expected:
        raise MetricError("no examples")
    return list(expected), list(predicted)


def accuracy(expected: Sequence[Any], predicted: Sequence[Any]) -> float:
    e, p = _labels(expected, predicted)
    return float(np.mean([1.0 if a == b else 0.0 for a, b in zip(e, p, strict=True)]))


def confusion_binary(expected: Sequence[int], predicted: Sequence[int]) -> dict[str, int]:
    e, p = _labels(expected, predicted)
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for a, b in zip(e, p, strict=True):
        if a not in (0, 1) or b not in (0, 1):
            raise MetricError("confusion_binary needs 0/1 labels")
        if a == 1 and b == 1:
            out["tp"] += 1
        elif a == 0 and b == 1:
            out["fp"] += 1
        elif a == 0 and b == 0:
            out["tn"] += 1
        else:
            out["fn"] += 1
    return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def f1_binary(expected: Sequence[int], predicted: Sequence[int]) -> float:
    c = confusion_binary(expected, predicted)
    return _prf(c["tp"], c["fp"], c["fn"])[2]


def precision_binary(expected: Sequence[int], predicted: Sequence[int]) -> float:
    c = confusion_binary(expected, predicted)
    return _prf(c["tp"], c["fp"], c["fn"])[0]


def recall_binary(expected: Sequence[int], predicted: Sequence[int]) -> float:
    c = confusion_binary(expected, predicted)
    return _prf(c["tp"], c["fp"], c["fn"])[1]


def macro_f1(expected: Sequence[Any], predicted: Sequence[Any]) -> float:
    e, p = _labels(expected, predicted)
    labels = sorted(set(e) | set(p), key=repr)
    if not labels:
        raise MetricError("no labels")
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(e, p, strict=True) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(e, p, strict=True) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(e, p, strict=True) if a == lab and b != lab)
        f1s.append(_prf(tp, fp, fn)[2])
    return float(sum(f1s) / len(f1s))


def mae(expected: Sequence[float], predicted: Sequence[float]) -> float:
    e, p = _labels(expected, predicted)
    return float(np.mean([abs(float(a) - float(b)) for a, b in zip(e, p, strict=True)]))


def rmse(expected: Sequence[float], predicted: Sequence[float]) -> float:
    e, p = _labels(expected, predicted)
    sq = [(float(a) - float(b)) ** 2 for a, b in zip(e, p, strict=True)]
    return float(math.sqrt(np.mean(sq)))


def exact_match(expected: Sequence[str], predicted: Sequence[str]) -> float:
    e, p = _labels(expected, predicted)
    hits = [1.0 if str(a).strip() == str(b).strip() else 0.0 for a, b in zip(e, p, strict=True)]
    return float(np.mean(hits))


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    """SQuAD-style token F1 averaged over examples."""
    e, p = _labels(expected, predicted)
    scores = []
    for a, b in zip(e, p, strict=True):
        at, bt = _normalize_text(str(a)).split(), _normalize_text(str(b)).split()
        if not at and not bt:
            scores.append(1.0)
            continue
        if not at or not bt:
            scores.append(0.0)
            continue
        # multiset-correct overlap so duplicate tokens count properly
        ca, cb = Counter(at), Counter(bt)
        common = sum((ca & cb).values())
        if common == 0:
            scores.append(0.0)
            continue
        prec = common / len(bt)
        rec = common / len(at)
        scores.append(2 * prec * rec / (prec + rec))
    return float(sum(scores) / len(scores))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al.): 1 - C(n-c,k)/C(n,k)."""
    if not 0 <= c <= n or k <= 0:
        raise MetricError(f"invalid pass@k args n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


METRICS: dict[str, Callable[[Sequence[Any], Sequence[Any]], float]] = {
    "accuracy": accuracy,
    "f1_binary": f1_binary,
    "precision_binary": precision_binary,
    "recall_binary": recall_binary,
    "macro_f1": macro_f1,
    "mae": mae,
    "rmse": rmse,
    "exact_match": exact_match,
    "token_f1": token_f1,
}


def compute_metrics(
    names: Sequence[str], expected: Sequence[Any], predicted: Sequence[Any]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        fn = METRICS.get(name)
        if fn is None:
            raise MetricError(f"unknown metric {name!r}; available: {sorted(METRICS)}")
        out[name] = float(fn(expected, predicted))
    return out
