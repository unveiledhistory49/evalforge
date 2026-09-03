"""Reproducible evaluation: pinned seeds, hashed datasets, digest-verified runs."""

from evalforge.datasets import Example, load_examples, register_dataset
from evalforge.gates import check_gate
from evalforge.metrics import METRICS, compute_metrics
from evalforge.runs import reproduce_run, run_evaluation
from evalforge.seeds import SeedBundle, seeded_rng

__all__ = [
    "Example",
    "METRICS",
    "SeedBundle",
    "check_gate",
    "compute_metrics",
    "load_examples",
    "register_dataset",
    "reproduce_run",
    "run_evaluation",
    "seeded_rng",
]
__version__ = "1.0.0"
