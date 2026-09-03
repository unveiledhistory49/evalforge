# Threat Model — EvalForge

## Assets

1. Evaluation integrity (numbers that gate releases and papers)
2. Dataset provenance (what was actually measured)
3. Run history (comparisons across time)

## Threats and mitigations

| Threat | Mitigation | Verified by |
| --- | --- | --- |
| Silent dataset edits | sha256 registry + snapshot + verify-on-load | tamper test |
| Seed gaming (pick the lucky seed) | Seeds recorded in run config + digest; cherry-picking is visible, not preventable — and documented as such | digest-includes-seed test |
| Metric shopping (report the best of 9) | Metrics list pinned in config + digest; reports show exactly what ran | config test |
| Unseeded framework nondeterminism | `runtime_report` surfaces torch presence; seed streams isolate randomness | stance test |
| Subprocess model abuse | Command is operator-supplied local config; timeout enforced; stderr captured | timeout/exit tests |
| Artifact tampering | Digest recomputation diverges; `reproduce` re-executes from source of truth | tamper test |
| Comparing incomparable runs | Dataset mismatch and empty id-overlap fail loudly | mismatch tests |
| Gate misconfiguration | Unknown metrics fail; `max_drop` without baseline raises | gate tests |

## Residual risks (honest)

- A determined actor with store write access can rewrite history (git-commit
  the `.evalforge/` dir or back it up to make this detectable).
- Statistical comparisons assume i.i.d. examples; violated assumptions are a
  methodology problem no tooling fixes (documented in RUNBOOK).
- p-values invite p-hacking; CIs are reported alongside point estimates to
  keep uncertainty visible.
- Wall-clock `duration_ms` is informational only, never part of the digest.
