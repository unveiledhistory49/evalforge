import json
import os

import pytest

from evalforge.cli import main


def _run(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(argv[1:])  # main takes argparse-style args without the prog name
    assert e.value.code == 0, f"{argv} exited {e.value.code}"


def test_cli_end_to_end(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from pathlib import Path

    store = str(Path(str(tmp_path)) / "store")
    monkeypatch.setenv("EVALFORGE_STORE", store)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    data = str(Path(str(tmp_path)) / "demo.jsonl")

    _run(["evalforge", "init"])
    _run(["evalforge", "synthesize", "--path", data, "--n", "120", "--seed", "4"])
    _run(["evalforge", "register-dataset", "demo", data])
    _run(
        [
            "evalforge",
            "run",
            "--model",
            "examples.models:threshold_model",
            "--dataset",
            "demo",
            "--seed",
            "7",
            "--metrics",
            "accuracy,f1_binary",
            "--name",
            "t1",
        ]
    )
    out = capsys.readouterr().out
    run_id = json.loads(out.strip().splitlines()[-1])["id"]

    _run(["evalforge", "report", run_id])
    rep = json.loads(capsys.readouterr().out)
    assert rep["aggregates"]["accuracy"] > 0.8

    _run(["evalforge", "reproduce", run_id, "--model", "examples.models:threshold_model"])
    assert json.loads(capsys.readouterr().out)["match"] is True

    _run(
        [
            "evalforge",
            "run",
            "--model",
            "examples.models:constant_zero",
            "--dataset",
            "demo",
            "--seed",
            "7",
            "--metrics",
            "accuracy",
            "--name",
            "c1",
        ]
    )
    const_id = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["id"]
    _run(
        [
            "evalforge",
            "compare",
            run_id,
            const_id,
            "--metric",
            "accuracy",
            "--seed",
            "2",
            "--resamples",
            "300",
        ]
    )
    cmp = json.loads(capsys.readouterr().out)
    assert cmp["diff_b_minus_a"] < 0  # constant minus threshold

    _run(["evalforge", "gate", "--run", run_id, "--rules", '{"accuracy": {"min": 0.8}}'])
    assert json.loads(capsys.readouterr().out)["passed"] is True
    with pytest.raises(SystemExit) as e:
        main(["gate", "--run", run_id, "--rules", '{"accuracy": {"min": 0.99}}'])
    assert e.value.code == 1

    _run(["evalforge", "list"])
    assert run_id in capsys.readouterr().out
    assert os.path.isdir(store)
