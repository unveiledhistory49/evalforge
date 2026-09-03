import random
from typing import Any

import pytest
from sqlalchemy.orm import Session

from evalforge.compare import CompareError, compare_runs
from evalforge.datasets import register_dataset, synthesize_binary
from evalforge.gates import GateError, check_gate
from evalforge.runs import RunError, load_model, reproduce_run, run_evaluation


def _data(db: Session, tmp_path: object, monkeypatch: pytest.MonkeyPatch, seed: int = 21) -> str:
    from pathlib import Path

    monkeypatch.setenv("EVALFORGE_STORE", str(Path(str(tmp_path)) / "st"))
    data = str(Path(str(tmp_path)) / "data.jsonl")
    synthesize_binary(data, 200, seed=seed, separation=2.5)
    register_dataset(db, "syn", data)
    db.commit()
    return "syn"


def _threshold(inp: Any, _rng: random.Random) -> int:
    return 1 if float(inp[0]) > 0 else 0


def _constant(inp: Any, _rng: random.Random) -> int:
    return 0


def test_run_digest_stable_and_seed_pinned(
    db: Session, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = _data(db, tmp_path, monkeypatch)

    def go(seed: int) -> dict[str, Any]:
        return run_evaluation(
            db,
            model=_threshold,
            model_ref="t",
            dataset_name=name,
            dataset_version=1,
            seed=seed,
            metrics=["accuracy"],
            limit=None,
            params={},
            shuffle=False,
        )

    r1 = go(7)
    r2 = go(7)
    assert r1["digest"] == r2["digest"]
    assert r1["aggregates"]["accuracy"] > 0.8
    r3 = go(8)
    assert r3["digest"] != r1["digest"]  # seed is part of the digest
    assert r3["aggregates"]["accuracy"] == r1["aggregates"]["accuracy"]  # threshold is seed-free


def test_reproduce_matches_and_detects_tampering(
    db: Session, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as js

    from evalforge.runs import digest_results, get_run

    name = _data(db, tmp_path, monkeypatch)
    r = run_evaluation(
        db,
        model=_threshold,
        model_ref="t",
        dataset_name=name,
        dataset_version=1,
        seed=7,
        metrics=["accuracy"],
        limit=None,
        params={},
        shuffle=False,
    )
    db.commit()
    ok = reproduce_run(db, r["id"], _threshold)
    assert ok["match"] is True
    assert ok["expected_digest"] == r["digest"] == ok["actual_digest"]
    assert ok["reproduction_run_id"] != r["id"]
    # Tamper with the stored artifact: the digest must stop matching content.
    art_path = get_run(db, r["id"]).artifact_path
    with open(art_path, encoding="utf-8") as f:
        doc = js.load(f)
    doc["outputs"][0]["output"] = 999
    with open(art_path, "w", encoding="utf-8") as f:
        js.dump(doc, f)
    with open(art_path, encoding="utf-8") as f:
        tampered = js.load(f)
    recomputed = digest_results(
        [(o["id"], o["output"]) for o in tampered["outputs"]], tampered["config"]
    )
    assert recomputed != r["digest"]


def test_reproduce_missing_run(db: Session) -> None:
    with pytest.raises(RunError):
        reproduce_run(db, "missing", _threshold)


def test_load_model_errors() -> None:
    with pytest.raises(RunError):
        load_model("no-colon")
    with pytest.raises(RunError):
        load_model("no_such_module_xyz:fn")


def test_subprocess_model_echo(tmp_path: object) -> None:
    from pathlib import Path

    from evalforge.runs import SubprocessModel

    script = str(Path(str(tmp_path)) / "echo_model.py")
    model_src = (
        "import json,sys\n"
        "payload=json.load(sys.stdin)\n"
        "print(json.dumps({'output': payload['input']}))\n"
    )
    with open(script, "w", encoding="utf-8") as f:
        f.write(model_src)
    m = SubprocessModel(["python3", script])
    assert m({"a": 1}, random.Random(0)) == {"a": 1}
    bad = SubprocessModel(["python3", "-c", "import sys; sys.exit(3)"])
    with pytest.raises(RunError):
        bad(0, random.Random(0))


def test_compare_threshold_beats_constant(
    db: Session, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = _data(db, tmp_path, monkeypatch)
    a = run_evaluation(
        db,
        model=_threshold,
        model_ref="t",
        dataset_name=name,
        dataset_version=1,
        seed=7,
        metrics=["accuracy"],
        limit=None,
        params={},
        shuffle=False,
    )
    b = run_evaluation(
        db,
        model=_constant,
        model_ref="c",
        dataset_name=name,
        dataset_version=1,
        seed=7,
        metrics=["accuracy"],
        limit=None,
        params={},
        shuffle=False,
    )
    db.commit()
    out = compare_runs(db, a["id"], b["id"], "accuracy", seed=1, resamples=300)
    assert out["diff_b_minus_a"] < -0.2  # constant minus threshold
    assert out["ci_95"]["hi"] < 0
    assert 0 <= out["mcnemar_p"] <= 1
    assert out["wins"] > out["losses"]  # threshold-only correct dominates
    with pytest.raises(RunError):
        compare_runs(db, a["id"], "missing", "accuracy")
    # different datasets cannot be compared
    from pathlib import Path as _P

    other = str(_P(str(tmp_path)) / "other.jsonl")
    synthesize_binary(other, 50, seed=99, separation=2.5)
    register_dataset(db, "other", other)
    db.commit()
    c = run_evaluation(
        db,
        model=_threshold,
        model_ref="t",
        dataset_name="other",
        dataset_version=1,
        seed=7,
        metrics=["accuracy"],
        limit=None,
        params={},
        shuffle=False,
    )
    db.commit()
    with pytest.raises(CompareError):
        compare_runs(db, a["id"], c["id"], "accuracy")


def test_gates() -> None:
    ok = check_gate({"accuracy": 0.9}, {"accuracy": {"min": 0.8}})
    assert ok["passed"] is True
    bad = check_gate({"accuracy": 0.7}, {"accuracy": {"min": 0.8}})
    assert bad["passed"] is False
    reg = check_gate(
        {"accuracy": 0.89}, {"accuracy": {"max_drop": 0.02}}, baseline={"accuracy": 0.9}
    )
    assert reg["passed"] is True
    reg2 = check_gate(
        {"accuracy": 0.8}, {"accuracy": {"max_drop": 0.02}}, baseline={"accuracy": 0.9}
    )
    assert reg2["passed"] is False
    missing = check_gate({"accuracy": 0.9}, {"f1": {"min": 0.5}})
    assert missing["passed"] is False
    with pytest.raises(GateError):
        check_gate({"accuracy": 0.9}, {"accuracy": {"max_drop": 0.01}})
