import json
import os

import pytest
from sqlalchemy.orm import Session

from evalforge.datasets import DatasetError, load_examples, register_dataset, synthesize_binary


def _write_jsonl(tmp_path: object, name: str, rows: list[dict[str, object]]) -> str:
    from pathlib import Path

    p = str(Path(str(tmp_path)) / name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_register_versions_and_verifies(
    db: Session, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    store = str(Path(str(tmp_path)) / "store")
    monkeypatch.setenv("EVALFORGE_STORE", store)
    rows = [{"id": f"e{i}", "input": i, "expected": i % 2} for i in range(10)]
    path = _write_jsonl(tmp_path, "d.jsonl", rows)
    v1 = register_dataset(db, "demo", path)
    assert (v1.version, v1.n_examples) == (1, 10)
    v2 = register_dataset(db, "demo", path)
    assert v2.version == 2  # same bytes, new version (explicit re-registration)
    assert v2.sha256 == v1.sha256
    examples, rec = load_examples(db, "demo")
    assert rec.version == 2 and len(examples) == 10
    assert examples[0].id == "e0"
    # tamper with the snapshot -> verification fails
    snap = os.path.join(store, "datasets", "demo", "v2.jsonl")
    with open(snap, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "evil", "input": 0, "expected": 1}) + "\n")
    with pytest.raises(DatasetError, match="hash verification"):
        load_examples(db, "demo", version=2)


def test_shuffle_needs_seed_and_is_deterministic(
    db: Session, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    monkeypatch.setenv("EVALFORGE_STORE", str(Path(str(tmp_path)) / "s"))
    rows = [{"id": f"e{i}", "input": i, "expected": i} for i in range(50)]
    register_dataset(db, "shuf", _write_jsonl(tmp_path, "s.jsonl", rows))
    with pytest.raises(DatasetError, match="seed"):
        load_examples(db, "shuf", shuffle=True)
    a, _ = load_examples(db, "shuf", seed=11, shuffle=True)
    b, _ = load_examples(db, "shuf", seed=11, shuffle=True)
    c, _ = load_examples(db, "shuf", seed=12, shuffle=True)
    assert [e.id for e in a] == [e.id for e in b]
    assert [e.id for e in a] != [e.id for e in c]
    limited, _ = load_examples(db, "shuf", limit=5)
    assert len(limited) == 5


def test_register_rejects_garbage(db: Session, tmp_path: object) -> None:
    from pathlib import Path

    bad = str(Path(str(tmp_path)) / "bad.jsonl")
    with open(bad, "w", encoding="utf-8") as f:
        f.write('{"id": "x"}\n')  # missing expected
    with pytest.raises(DatasetError):
        register_dataset(db, "bad", bad)
    with pytest.raises(DatasetError):
        register_dataset(db, "missing", "/nope.jsonl")
    with pytest.raises(DatasetError):
        load_examples(db, "ghost")


def test_synthesize_is_deterministic(tmp_path: object) -> None:
    from pathlib import Path

    p1 = str(Path(str(tmp_path)) / "a.jsonl")
    p2 = str(Path(str(tmp_path)) / "b.jsonl")
    synthesize_binary(p1, 100, seed=5, separation=3.0)
    synthesize_binary(p2, 100, seed=5, separation=3.0)
    assert open(p1, "rb").read() == open(p2, "rb").read()
    p3 = str(Path(str(tmp_path)) / "c.jsonl")
    synthesize_binary(p3, 100, seed=6, separation=3.0)
    assert open(p1, "rb").read() != open(p3, "rb").read()
