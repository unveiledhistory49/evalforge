import math

import pytest

from evalforge.seeds import seeded_rng
from evalforge.stats import bootstrap_mean_ci, mcnemar_exact, paired_diff_ci, win_loss_tie


def test_bootstrap_ci_deterministic_and_covering() -> None:
    vals = [float(i) for i in range(1, 101)]  # mean 50.5
    ci1 = bootstrap_mean_ci(seeded_rng(3, "bs"), vals, resamples=500)
    ci2 = bootstrap_mean_ci(seeded_rng(3, "bs"), vals, resamples=500)
    assert (ci1.lo, ci1.hi, ci1.estimate) == (ci2.lo, ci2.hi, ci2.estimate)
    assert ci1.lo < 50.5 < ci1.hi
    assert ci1.estimate == pytest.approx(50.5)
    ci3 = bootstrap_mean_ci(seeded_rng(4, "bs"), vals, resamples=500)
    assert (ci3.lo, ci3.hi) != (ci1.lo, ci1.hi)  # different seed, different resample
    with pytest.raises(ValueError):
        bootstrap_mean_ci(seeded_rng(1, "x"), [], resamples=500)


def test_paired_diff_ci_sign() -> None:
    a = [1.0] * 80 + [0.0] * 20
    b = [1.0] * 60 + [0.0] * 40
    ci = paired_diff_ci(seeded_rng(9, "cmp"), a, b, resamples=500)
    assert ci.estimate == pytest.approx(0.2)
    assert ci.lo > 0  # A clearly better than B


def test_mcnemar_exact_hand_computed() -> None:
    # n=10, k=min(8,2)=2: tail = C(10,0)+C(10,1)+C(10,2) = 56; p = 2*56/1024
    expected = 2 * (math.comb(10, 0) + math.comb(10, 1) + math.comb(10, 2)) / 2**10
    assert mcnemar_exact(8, 2) == pytest.approx(expected)
    assert mcnemar_exact(2, 8) == pytest.approx(expected)  # symmetric
    assert mcnemar_exact(0, 0) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_win_loss_tie() -> None:
    assert win_loss_tie([True, True, False], [True, False, True]) == {
        "wins": 1,
        "losses": 1,
        "ties": 1,
    }
    with pytest.raises(ValueError):
        win_loss_tie([True], [True, False])
