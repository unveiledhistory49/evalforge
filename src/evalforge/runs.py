"""Evaluation runs. A run is fully described by (model_ref, dataset, seed,
metrics, params) and its digest binds the exact per-example outputs —
re-running with the same inputs must reproduce the digest bit-for-bit."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import subprocess
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.datasets import load_examples
from evalforge.db import store_path
from evalforge.metrics import compute_metrics
from evalforge.models import Run
from evalforge.seeds import SeedBundle, validate_seed

ModelFn = Callable[[Any, random.Random], Any]


class RunError(Exception):
    pass


def load_model(ref: str) -> ModelFn:
    """Load a model callable from 'module:attr' import path."""
    if ":" not in ref:
        raise RunError(f"model ref must be 'module:attr', got {ref!r}")
    mod_name, attr = ref.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise RunError(f"cannot import model module {mod_name!r}: {e}") from e
    try:
        fn = getattr(mod, attr)
    except AttributeError as e:
        raise RunError(f"module {mod_name!r} has no attribute {attr!r}") from e
    if not callable(fn):
        raise RunError(f"model {ref!r} is not callable")
    return fn  # type: ignore[no-any-return]


class SubprocessModel:
    """Language-agnostic model: JSON {id, input} on stdin, {output} on stdout."""

    def __init__(self, command: list[str], timeout_s: float = 30.0) -> None:
        if not command:
            raise RunError("subprocess model needs a command")
        self.command = command
        self.timeout_s = timeout_s

    def __call__(self, example_input: Any, _rng: random.Random) -> Any:
        try:
            # Command comes from the operator's own CLI flag, never from
            # evaluated data; list form, no shell.
            proc = subprocess.run(  # noqa: S603
                self.command,
                input=json.dumps({"input": example_input}),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise RunError(f"model timed out after {self.timeout_s}s") from e
        if proc.returncode != 0:
            raise RunError(f"model exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        try:
            return json.loads(proc.stdout)["output"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise RunError(f"model returned invalid JSON: {proc.stdout.strip()[:200]}") from e


def digest_results(outputs: list[tuple[str, Any]], config: dict[str, Any]) -> str:
    """Canonical digest: per-example outputs sorted by id plus the full config."""
    canonical = json.dumps(
        {"config": config, "outputs": sorted(outputs, key=lambda t: t[0])},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _config_of(
    model_ref: str,
    dataset_name: str,
    version: int,
    sha: str,
    seed: int,
    metrics: list[str],
    limit: int | None,
    params: dict[str, Any],
    shuffle: bool,
) -> dict[str, Any]:
    return {
        "model_ref": model_ref,
        "dataset": {"name": dataset_name, "version": version, "sha256": sha},
        "seed": seed,
        "metrics": sorted(metrics),
        "limit": limit,
        "params": params,
        "shuffle": shuffle,
    }


def run_evaluation(
    db: Session,
    *,
    model: ModelFn,
    model_ref: str,
    dataset_name: str,
    dataset_version: int | None,
    seed: int,
    metrics: list[str],
    limit: int | None = None,
    params: dict[str, Any] | None = None,
    shuffle: bool = False,
    run_name: str = "",
) -> dict[str, Any]:
    validate_seed(seed)
    if not metrics:
        raise RunError("at least one metric is required")
    params = dict(params or {})
    bundle = SeedBundle(seed)
    provenance = bundle.apply()
    examples, ds = load_examples(
        db, dataset_name, dataset_version, seed=seed, shuffle=shuffle, limit=limit
    )
    model_rng = bundle.rng("model")
    config = _config_of(
        model_ref, ds.name, ds.version, ds.sha256, seed, metrics, limit, params, shuffle
    )
    outputs: list[tuple[str, Any]] = []
    started = time.time()
    for ex in examples:
        outputs.append((ex.id, model(ex.input, model_rng)))
    duration_ms = int((time.time() - started) * 1000)
    digest = digest_results(outputs, config)
    expected = [ex.expected for ex in examples]
    predicted = [o for _, o in outputs]
    aggregates = compute_metrics(metrics, expected, predicted)
    rows = [
        {"id": ex.id, "expected": ex.expected, "output": o}
        for ex, (_, o) in zip(examples, outputs, strict=True)
    ]
    return _persist(
        db, config, len(examples), digest, rows, aggregates, duration_ms, dict(provenance), run_name
    )


def _persist(
    db: Session,
    config: dict[str, Any],
    n: int,
    digest: str,
    rows: list[dict[str, Any]],
    aggregates: dict[str, float],
    duration_ms: int,
    provenance: dict[str, str],
    run_name: str,
) -> dict[str, Any]:
    run = Run(
        name=run_name,
        model_ref=str(config["model_ref"]),
        dataset_name=str(config["dataset"]["name"]),
        dataset_version=int(config["dataset"]["version"]),
        dataset_sha256=str(config["dataset"]["sha256"]),
        seed=int(config["seed"]),
        metrics=json.dumps(sorted(config["metrics"])),
        params=json.dumps(config["params"], sort_keys=True),
        n_examples=n,
        digest=digest,
        artifact_path="",
        status="complete",
        duration_ms=duration_ms,
    )
    db.add(run)
    db.flush()
    artifact = os.path.join(store_path(), "runs", f"{run.id}.json")
    os.makedirs(os.path.dirname(artifact), exist_ok=True)
    with open(artifact, "w", encoding="utf-8") as f:
        json.dump(
            {"config": config, "provenance": provenance, "aggregates": aggregates, "outputs": rows},
            f,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    run.artifact_path = artifact
    db.flush()
    return run_report(db, run.id)


def get_run(db: Session, run_id: str) -> Run:
    run = db.execute(select(Run).where(Run.id == run_id)).scalar_one_or_none()
    if run is None:
        raise RunError(f"run {run_id} not found")
    return run


def list_runs(db: Session, limit: int = 50) -> list[Run]:
    return list(
        db.execute(select(Run).order_by(Run.created_at.desc()).limit(limit)).scalars().all()
    )


def run_report(db: Session, run_id: str) -> dict[str, Any]:
    run = get_run(db, run_id)
    with open(run.artifact_path, encoding="utf-8") as f:
        artifact = json.load(f)
    return {
        "id": run.id,
        "name": run.name,
        "model_ref": run.model_ref,
        "dataset": {
            "name": run.dataset_name,
            "version": run.dataset_version,
            "sha256": run.dataset_sha256,
        },
        "seed": run.seed,
        "metrics": json.loads(run.metrics),
        "params": json.loads(run.params),
        "n_examples": run.n_examples,
        "digest": run.digest,
        "status": run.status,
        "duration_ms": run.duration_ms,
        "aggregates": artifact["aggregates"],
        "provenance": artifact["provenance"],
    }


def reproduce_run(db: Session, run_id: str, model: ModelFn | None = None) -> dict[str, Any]:
    """Re-execute a run's exact config and compare digests. Any mismatch —
    data edit, code change, seed drift — fails loudly."""
    run = get_run(db, run_id)
    fn = model if model is not None else load_model(run.model_ref)
    fresh = run_evaluation(
        db,
        model=fn,
        model_ref=run.model_ref,
        dataset_name=run.dataset_name,
        dataset_version=run.dataset_version,
        seed=run.seed,
        metrics=json.loads(run.metrics),
        limit=None,
        params=json.loads(run.params),
        shuffle=_shuffle_of(db, run_id),
        run_name=f"reproduce:{run_id[:8]}",
    )
    match = fresh["digest"] == run.digest
    return {
        "match": match,
        "expected_digest": run.digest,
        "actual_digest": fresh["digest"],
        "reproduction_run_id": fresh["id"],
    }


def _shuffle_of(db: Session, run_id: str) -> bool:
    with open(get_run(db, run_id).artifact_path, encoding="utf-8") as f:
        return bool(json.load(f)["config"].get("shuffle", False))
