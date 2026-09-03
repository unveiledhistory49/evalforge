import pytest

from evalforge.seeds import (
    SeedBundle,
    SeedError,
    runtime_report,
    seeded_numpy,
    seeded_rng,
    validate_seed,
)


def test_validate_seed_bounds() -> None:
    assert validate_seed(0) == 0
    assert validate_seed(2**32 - 1) == 2**32 - 1
    for bad in (-1, 2**32, "7", 7.0, True, None):
        with pytest.raises(SeedError):
            validate_seed(bad)  # type: ignore[arg-type]


def test_stream_determinism_and_independence() -> None:
    a1 = seeded_rng(7, "dataset")
    a2 = seeded_rng(7, "dataset")
    b = seeded_rng(7, "model")
    assert [a1.random() for _ in range(5)] == [a2.random() for _ in range(5)]
    assert [a1.random() for _ in range(5)] != [b.random() for _ in range(5)]
    c = seeded_rng(8, "dataset")
    assert [a2.random() for _ in range(5)] != [c.random() for _ in range(5)]
    with pytest.raises(SeedError):
        seeded_rng(7, "")


def test_numpy_stream_determinism() -> None:
    g1 = seeded_numpy(7, "bootstrap")
    g2 = seeded_numpy(7, "bootstrap")
    assert g1.integers(0, 1000, size=10).tolist() == g2.integers(0, 1000, size=10).tolist()


def test_bundle_apply_seeds_globals() -> None:
    import random

    SeedBundle(123).apply()
    first = [random.random() for _ in range(3)]
    SeedBundle(123).apply()
    assert [random.random() for _ in range(3)] == first
    SeedBundle(124).apply()
    assert [random.random() for _ in range(3)] != first


def test_bundle_warns_without_hashseed(recwarn: pytest.WarningsRecorder) -> None:
    import os

    had = os.environ.pop("PYTHONHASHSEED", None)
    try:
        SeedBundle(1).apply()
        assert any("PYTHONHASHSEED" in str(w.message) for w in recwarn.list)
    finally:
        if had is not None:
            os.environ["PYTHONHASHSEED"] = had


def test_runtime_report_documents_torch_stance() -> None:
    rep = runtime_report()
    assert rep["torch_installed"] in (True, False)
    assert "not required" in str(rep["torch_note"])
