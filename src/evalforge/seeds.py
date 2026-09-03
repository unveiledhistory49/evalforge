"""Seed discipline. One integer seed fans out into independent per-stream RNGs,
so dataset shuffling, model sampling, and bootstrap resampling never share
state — changing one stream cannot perturb another (see ADR-0002)."""

from __future__ import annotations

import hashlib
import os
import random
import warnings

import numpy as np

MAX_SEED = 2**32 - 1


class SeedError(ValueError):
    pass


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SeedError(f"seed must be an integer, got {seed!r}")
    if not 0 <= seed <= MAX_SEED:
        raise SeedError(f"seed must be in [0, {MAX_SEED}], got {seed}")
    return seed


def seeded_rng(seed: int, stream: str) -> random.Random:
    """Deterministic Random for one named stream. Different streams (even with
    the same seed) produce independent sequences; same (seed, stream) replays."""
    validate_seed(seed)
    if not stream:
        raise SeedError("stream name must not be empty")
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    # Deterministic replay RNG by design; explicitly NOT for secrets/tokens.
    return random.Random(int.from_bytes(digest, "big"))  # noqa: S311


def seeded_numpy(seed: int, stream: str) -> np.random.Generator:
    """Numpy counterpart of seeded_rng (same derivation, PCG64 stream)."""
    validate_seed(seed)
    if not stream:
        raise SeedError("stream name must not be empty")
    digest = hashlib.sha256(f"np:{seed}:{stream}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big")))


class SeedBundle:
    """Applies a pinned seed to every supported RNG and records the provenance."""

    def __init__(self, seed: int) -> None:
        self.seed = validate_seed(seed)

    def apply(self) -> dict[str, str]:
        """Seed stdlib random + numpy legacy global state. Returns what was set.
        Warns (does not fail) when PYTHONHASHSEED is unset, since hash
        randomization can affect iteration order of sets/frozensets."""
        random.seed(self.seed)
        np.random.seed(self.seed % MAX_SEED)
        provenance: dict[str, str] = {
            "seed": str(self.seed),
            "python_random": "seeded",
            "numpy_legacy": "seeded",
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "UNSET"),
        }
        if "PYTHONHASHSEED" not in os.environ:
            warnings.warn(
                "PYTHONHASHSEED is unset: set it (e.g. PYTHONHASHSEED=0) for "
                "fully deterministic runs involving sets/frozensets",
                stacklevel=2,
            )
        return provenance

    def rng(self, stream: str) -> random.Random:
        return seeded_rng(self.seed, stream)

    def numpy(self, stream: str) -> np.random.Generator:
        return seeded_numpy(self.seed, stream)


def runtime_report() -> dict[str, object]:
    """Report RNG-capable frameworks present in this interpreter. EvalForge
    never requires torch — but if it IS installed, silent unseeded use is the
    classic reproducibility hole, so we surface it instead of ignoring it."""
    import importlib.util

    return {
        "python": "stdlib random (seeded via SeedBundle)",
        "numpy": str(np.__version__),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "torch_note": "not required; if used by a model, seed it explicitly (see RUNBOOK)",
    }
