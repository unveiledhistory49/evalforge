"""Model comparison: paired bootstrap CIs, exact McNemar, win/loss/tie.
Comparisons align runs by example id — runs over different examples cannot
be compared and fail loudly instead of silently misaligning."""

from __future__ import annotations

import json
from typing import Any, cast

from sqlalchemy.orm import Session

from evalforge.runs import get_run
from evalforge.seeds import seeded_rng, validate_seed
from evalforge.stats import mcnemar_exact, paired_diff_ci, win_loss_tie


class CompareError(Exception):
    pass


def _artifact(run: Any) -> dict[str, Any]:
    with open(run.artifact_path, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def compare_runs(
    db: Session,
    run_a_id: str,
    run_b_id: str,
    metric: str,
    *,
    seed: int = 0,
    resamples: int = 2000,
) -> dict[str, Any]:
    validate_seed(seed)
    ra, rb = get_run(db, run_a_id), get_run(db, run_b_id)
    if ra.dataset_name != rb.dataset_name or ra.dataset_version != rb.dataset_version:
        raise CompareError(
            f"dataset mismatch: {ra.dataset_name}v{ra.dataset_version} vs "
            f"{rb.dataset_name}v{rb.dataset_version}"
        )
    aa, ab = _artifact(ra), _artifact(rb)
    if metric not in aa["aggregates"] or metric not in ab["aggregates"]:
        raise CompareError(f"metric {metric!r} not in both runs")
    out_a = {o["id"]: o["output"] for o in aa["outputs"]}
    out_b = {o["id"]: o["output"] for o in ab["outputs"]}
    exp_a = {o["id"]: o.get("expected") for o in aa["outputs"]}
    common = sorted(set(out_a) & set(out_b))
    if not common:
        raise CompareError("runs share no example ids")
    scores_a: list[float] = []
    scores_b: list[float] = []
    for i in common:
        ea = exp_a.get(i)
        if ea is None:
            raise CompareError(f"artifact of run {ra.id} lacks expected labels")
        if (
            isinstance(ea, (int, float))
            and isinstance(out_a[i], (int, float))
            and isinstance(out_b[i], (int, float))
        ):
            scores_a.append(-abs(float(out_a[i]) - float(ea)))
            scores_b.append(-abs(float(out_b[i]) - float(ea)))
        else:
            scores_a.append(1.0 if out_a[i] == ea else 0.0)
            scores_b.append(1.0 if out_b[i] == ea else 0.0)
    rng = seeded_rng(seed, f"compare:{run_a_id[:8]}:{run_b_id[:8]}:{metric}")
    ci = paired_diff_ci(rng, scores_b, scores_a, resamples=resamples)
    # Correctness is exact output match — independent of the score scale used
    # for the CI (e.g. negative errors for numeric outputs), so WLT/McNemar
    # stay meaningful for any metric.
    correct_a = [out_a[i] == ea for i, ea in ((i, exp_a.get(i)) for i in common)]
    correct_b = [out_b[i] == exp_a.get(i) for i in common]
    wlt = win_loss_tie(correct_a, correct_b)
    return {
        "metric": metric,
        "run_a": {"id": ra.id, "value": aa["aggregates"][metric]},
        "run_b": {"id": rb.id, "value": ab["aggregates"][metric]},
        "diff_b_minus_a": ci.estimate,
        "ci_95": {"lo": ci.lo, "hi": ci.hi},
        "mcnemar_p": mcnemar_exact(wlt["wins"], wlt["losses"]),
        **wlt,
        "n_common": len(common),
    }
