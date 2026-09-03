"""Demo models (no ML libraries). Each is a pure function of (input, rng) —
stateless, inspectable, and honest about what they demonstrate."""
from __future__ import annotations

import random
from typing import Any


def constant_model(label: int = 0):  # type: ignore[no-untyped-def]
    """Always predicts `label`. The floor every real model must beat."""

    def predict(example_input: Any, _rng: random.Random) -> int:
        return label

    return predict


def threshold_model(example_input: Any, _rng: random.Random) -> int:
    """x[0] > 0 -> 1. The Bayes-ish classifier for synthesize_binary demos."""
    try:
        return 1 if float(example_input[0]) > 0 else 0
    except (TypeError, IndexError, ValueError):
        return 0


def keyword_model(words: tuple[str, ...] = ("good", "great", "excellent")) -> Any:
    """Text contains any keyword -> 1 else 0. Shows text-metric plumbing."""

    def predict(example_input: Any, _rng: random.Random) -> int:
        text = str(example_input).lower()
        return 1 if any(w in text for w in words) else 0

    return predict


def seeded_random_model(example_input: Any, rng: random.Random) -> int:
    """Coin flip from the run's model stream. Same seed -> same flips (this is
    the point: unseeded randomness is the reproducibility hole this
    framework closes)."""
    _ = example_input
    return 1 if rng.random() < 0.5 else 0


def noisy_threshold_model(noise: float = 0.1) -> Any:
    """threshold_model with label flips at rate `noise` drawn from run rng."""

    def predict(example_input: Any, rng: random.Random) -> int:
        base = threshold_model(example_input, rng)
        return 1 - base if rng.random() < noise else base

    return predict


# Ready-to-reference model instances for `--model examples.models:<name>`.
constant_zero = constant_model(0)
constant_one = constant_model(1)
keyword_default = keyword_model()
noisy_threshold_10 = noisy_threshold_model(0.1)
