import math

import pytest

from evalforge.metrics import (
    METRICS,
    accuracy,
    compute_metrics,
    confusion_binary,
    exact_match,
    f1_binary,
    macro_f1,
    mae,
    pass_at_k,
    precision_binary,
    recall_binary,
    rmse,
    token_f1,
)


def test_accuracy_and_confusion() -> None:
    e = [1, 1, 0, 0, 1]
    p = [1, 0, 0, 1, 1]
    assert accuracy(e, p) == pytest.approx(0.6)
    assert confusion_binary(e, p) == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}
    assert f1_binary(e, p) == pytest.approx(2 / 3)
    assert precision_binary(e, p) == pytest.approx(2 / 3)
    assert recall_binary(e, p) == pytest.approx(2 / 3)


def test_macro_f1_hand_computed() -> None:
    # labels A: tp=2,fp=1,fn=0 -> P=2/3,R=1,F1=0.8; labels B: tp=1,fp=0,fn=1 -> P=1,R=0.5,F1=2/3
    e = ["A", "A", "B", "B"]
    p = ["A", "A", "A", "B"]
    assert macro_f1(e, p) == pytest.approx((0.8 + 2 / 3) / 2)


def test_regression_metrics() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(1 / 3)
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(math.sqrt(1 / 3))


def test_text_metrics() -> None:
    assert exact_match([" Paris "], ["Paris"]) == 1.0
    assert exact_match(["Paris"], ["London"]) == 0.0
    # SQuAD-style: pred {quick,brown,fox} vs exp {quick,brown,fox,jumps}
    assert token_f1(["The quick brown fox"], ["quick brown fox jumps"]) == pytest.approx(1.5 / 1.75)
    assert token_f1([""], [""]) == 1.0
    assert token_f1(["hello"], [""]) == 0.0


def test_pass_at_k() -> None:
    # 1 - C(5,2)/C(10,2) = 1 - 10/45
    assert pass_at_k(10, 5, 2) == pytest.approx(1 - 10 / 45)
    assert pass_at_k(10, 10, 3) == 1.0
    assert pass_at_k(10, 0, 3) == 0.0
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)


def test_compute_metrics_registry() -> None:
    out = compute_metrics(["accuracy", "f1_binary"], [1, 0], [1, 0])
    assert out == {"accuracy": 1.0, "f1_binary": 1.0}
    with pytest.raises(ValueError):
        compute_metrics(["nope"], [1], [1])
    with pytest.raises(ValueError):
        compute_metrics(["accuracy"], [1, 0], [1])
    assert "accuracy" in METRICS and "token_f1" in METRICS
