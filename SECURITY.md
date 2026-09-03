# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.x | Yes |

## Reporting a vulnerability

Open a **private security advisory** on GitHub. Include reproduction steps and
the commit hash. Acknowledgement target: 3 business days.

## Posture

- Threat model: `docs/THREAT_MODEL.md`.
- Only third-party runtime dependency is numpy; `pip-audit` runs in CI.
- Dependabot covers pip and GitHub Actions.
- No network access at runtime (models run in-process or via local subprocess).
