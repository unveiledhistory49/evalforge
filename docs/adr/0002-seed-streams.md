# ADR-0002: Stream-separated seeds

Date: 2026-09-04. Status: accepted.

## Context

A single global RNG shared between dataset shuffling, model sampling, and
bootstrap resampling couples unrelated choices: bumping bootstrap resamples
changes which examples the model sees. Debugging becomes archaeology.

## Decision

`seeded_rng(seed, stream)` derives an independent `random.Random` per named
stream from `SHA-256(seed:stream)` (numpy variant via PCG64 with a distinct
domain prefix). Streams used: `dataset:{name}:{version}`, `model`,
`compare:{a}:{b}:{metric}`.

Global `random.seed`/`np.random.seed` are still applied for legacy model code,
and an unset `PYTHONHASHSEED` warns instead of failing (failing would break
environments the user doesn't control; the warning names the exact fix).

## Rejected

One RNG for everything (coupling); requiring callers to invent their own
derivation (inconsistent, untestable).
