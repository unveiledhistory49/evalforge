# ADR-0003: No framework dependency

Date: 2026-09-04. Status: accepted.

## Context

Eval harnesses welded to torch/sklearn inherit their install weight, version
churn, and GPU nondeterminism — and exclude every model that isn't written in
that framework.

## Decision

Models are `(input, rng) -> output` callables loaded from `module:attr`, plus
a `SubprocessModel` adapter (JSON over stdio) for anything else. Metrics and
statistics are implemented from scratch; the only third-party dependency is
numpy. `runtime_report()` surfaces whether torch happens to be installed so
silent unseeded use gets noticed instead of ignored.

## Consequences

GPU models are supported exactly to the extent their owners make them
deterministic — the framework refuses to pretend otherwise (see RUNBOOK).
Dependency footprint stays at numpy + stdlib, so `pip-audit` surface and
install time stay trivial.

## Rejected

torch/sklearn integration: heavy, exclusionary, and at odds with the
reproducibility story (opaque nondeterminism sources).
