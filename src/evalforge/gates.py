"""CI quality gates: fail loudly when a metric drops below a floor or
regresses against a baseline. Exit codes are the interface (0 pass, 1 fail)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GateError(ValueError):
    pass


@dataclass(frozen=True)
class GateCheck:
    metric: str
    rule: str
    value: float | None
    threshold: float | None
    passed: bool
    detail: str


def check_gate(
    metrics: dict[str, float],
    rules: dict[str, dict[str, Any]],
    baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Rules per metric: {"min": x} and/or {"max_drop": d} (needs baseline).
    Unknown metrics in rules fail the gate — a gate on a metric you didn't
    compute is a misconfiguration, not a pass."""
    checks: list[GateCheck] = []
    for metric, rule in rules.items():
        value = metrics.get(metric)
        if value is None:
            checks.append(GateCheck(metric, "present", None, None, False, "metric not computed"))
            continue
        if "min" in rule:
            floor = float(rule["min"])
            checks.append(
                GateCheck(
                    metric,
                    f">= {floor}",
                    value,
                    floor,
                    value >= floor,
                    f"{metric}={value:.4f} vs floor {floor}",
                )
            )
        if "max" in rule:
            ceil = float(rule["max"])
            checks.append(
                GateCheck(
                    metric,
                    f"<= {ceil}",
                    value,
                    ceil,
                    value <= ceil,
                    f"{metric}={value:.4f} vs ceiling {ceil}",
                )
            )
        if "max_drop" in rule:
            if baseline is None or metric not in baseline:
                raise GateError(f"max_drop on {metric!r} needs a baseline value")
            drop = float(baseline[metric]) - value
            allowed = float(rule["max_drop"])
            checks.append(
                GateCheck(
                    metric,
                    f"drop <= {allowed}",
                    value,
                    allowed,
                    drop <= allowed,
                    f"{metric} dropped {drop:.4f} from {baseline[metric]:.4f}",
                )
            )
    passed = all(c.passed for c in checks)
    return {"passed": passed, "checks": [c.__dict__ for c in checks]}
