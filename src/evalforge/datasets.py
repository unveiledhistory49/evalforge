"""Datasets: JSONL registration with content hashing, verified loads, and
seeded shuffling/subsampling. Tampered files fail loudly on load."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db import store_path
from evalforge.models import Dataset
from evalforge.seeds import seeded_rng, validate_seed


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class Example:
    id: str
    input: Any
    expected: Any


def _canonical_bytes(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}: invalid JSONL line {n + 1}: {e}") from e
            if not isinstance(obj, dict) or "id" not in obj or "expected" not in obj:
                raise DatasetError(f"{path}: line {n + 1} needs id/input/expected keys")
            h.update(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())
            h.update(b"\n")
            n += 1
    if n == 0:
        raise DatasetError(f"{path}: no examples found")
    return h.hexdigest(), n


def register_dataset(db: Session, name: str, path: str) -> Dataset:
    """Hash a JSONL file and register it as the next version of `name`.
    The file is snapshotted into the store so later edits can't move history."""
    clean = name.strip()
    if not clean or len(clean) > 200:
        raise DatasetError("dataset name must be 1..200 chars")
    if not os.path.isfile(path):
        raise DatasetError(f"dataset file not found: {path}")
    digest, n = _canonical_bytes(path)
    row = db.execute(
        select(Dataset.version)
        .where(Dataset.name == clean)
        .order_by(Dataset.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    version = (row or 0) + 1
    snap_dir = os.path.join(store_path(), "datasets", clean)
    os.makedirs(snap_dir, exist_ok=True)
    snap = os.path.join(snap_dir, f"v{version}.jsonl")
    with open(path, "rb") as src, open(snap, "wb") as dst:
        dst.write(src.read())
    rec = Dataset(
        name=clean, version=version, sha256=digest, n_examples=n, source_path=os.path.abspath(path)
    )
    db.add(rec)
    db.flush()
    return rec


def _read_all(path: str) -> list[Example]:
    out: list[Example] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(Example(id=str(obj["id"]), input=obj.get("input"), expected=obj["expected"]))
    return out


def load_examples(
    db: Session,
    name: str,
    version: int | None = None,
    *,
    seed: int | None = None,
    shuffle: bool = False,
    limit: int | None = None,
) -> tuple[list[Example], Dataset]:
    """Load a registered dataset, verifying its bytes against the registry.
    `version=None` pins to latest — callers that need exact reproducibility
    must pass an explicit version (runs always record it)."""
    q = select(Dataset).where(Dataset.name == name)
    q = (
        q.where(Dataset.version == version)
        if version is not None
        else q.order_by(Dataset.version.desc())
    )
    rec = db.execute(q.limit(1)).scalar_one_or_none()
    if rec is None:
        raise DatasetError(f"dataset {name} v{version} not registered")
    snap = os.path.join(store_path(), "datasets", name, f"v{rec.version}.jsonl")
    if not os.path.isfile(snap):
        raise DatasetError(f"dataset snapshot missing: {snap}")
    digest, _ = _canonical_bytes(snap)
    if digest != rec.sha256:
        raise DatasetError(f"dataset {name} v{rec.version} failed hash verification (tampered?)")
    examples = _read_all(snap)
    if shuffle:
        if seed is None:
            raise DatasetError("shuffle requires an explicit seed")
        validate_seed(seed)
        rng = seeded_rng(seed, f"dataset:{name}:{rec.version}")
        rng.shuffle(examples)
    if limit is not None:
        if limit <= 0:
            raise DatasetError("limit must be > 0")
        examples = examples[:limit]
    return examples, rec


def synthesize_binary(path: str, n: int, seed: int, separation: float = 2.0) -> str:
    """Write a deterministic 2D binary-classification JSONL for demos/tests.
    Class centers at (-s/2,0) and (+s/2,0) with unit Gaussian noise."""
    import numpy as np

    validate_seed(seed)
    if n <= 0:
        raise DatasetError("n must be > 0")
    rng = np.random.default_rng(seed)
    half = int(n // 2)
    lines: list[str] = []
    for i in range(n):
        label = 0 if i < n - half else 1
        cx = -separation / 2 if label == 0 else separation / 2
        x = rng.normal(cx, 1.0)
        y = rng.normal(0.0, 1.0)
        lines.append(
            json.dumps(
                {
                    "id": f"ex-{i:05d}",
                    "input": [round(float(x), 4), round(float(y), 4)],
                    "expected": label,
                }
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
