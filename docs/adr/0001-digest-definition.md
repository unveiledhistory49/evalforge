# ADR-0001: Digest over sorted outputs plus full config

Date: 2026-09-04. Status: accepted.

## Context

A "reproducible run" needs a checkable definition. Comparing aggregate
metrics is too weak (different outputs can share a mean); comparing raw logs
is too brittle (timestamps, paths, ordering).

## Decision

`digest_results` hashes canonical JSON of `{config, outputs-sorted-by-id}`.
The config embeds model ref, dataset name/version/sha256, seed, metric list,
limit, params, and shuffle flag. Example order is normalized by sorting, so
shuffled and unshuffled executions of the same logical run share a digest.

Reproduction = re-execute the recorded config, compare digests, report
`match` boolean. Anything else (dataset edit, code change, seed drift,
param tweak) changes the digest and fails loudly.

## Rejected

Metric-only comparison (collisions hide real divergence); log diffing
(brittle, unactionable).
