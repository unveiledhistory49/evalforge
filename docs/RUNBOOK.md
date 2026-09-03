# Runbook — EvalForge

## First evaluation

```bash
export EVALFORGE_STORE=./.evalforge PYTHONHASHSEED=0
evalforge init
evalforge synthesize --path data.jsonl --n 500 --seed 4
evalforge register-dataset mytask data.jsonl
evalforge run --model mypkg.models:baseline --dataset mytask \
  --seed 7 --metrics accuracy,macro_f1 --param temp=0
```

## Comparing checkpoints

```bash
evalforge run --model mypkg.models:candidate --dataset mytask --seed 7 \
  --metrics accuracy --name candidate
evalforge compare <baseline-id> <candidate-id> --metric accuracy
# diff + 95% CI + McNemar p + wins/losses/ties
```

## Gating releases in CI

```bash
evalforge gate --run <id> --rules '{"accuracy": {"min": 0.85}}'
evalforge gate --run <id> --rules '{"accuracy": {"max_drop": 0.01}}' \
  --baseline '{"accuracy": 0.87}'
```

Non-zero exit = fail the pipeline. Pin the baseline JSON in version control
next to the code it guards.

## Reproducing someone else's number

```bash
evalforge report <id>     # config, digest, aggregates, provenance
evalforge reproduce <id> --model mypkg.models:baseline
# {"match": true, ...} or a loud mismatch with both digests
```

If it mismatches: check dataset hash (`report` shows it), model code version,
and `PYTHONHASHSEED`. The digest tells you *that* something moved, the config
tells you *where to look*.

## Seeding torch models (if you must)

EvalForge never touches torch. In your model adapter, seed explicitly:

```python
import random, numpy as np, torch
torch.manual_seed(seed_from_evalforge)
```

and record the torch version in `--param torch=2.4.0`. GPU kernels may still
vary — note it in the run name rather than pretending otherwise.

## Backup

`.evalforge/` holds the registry DB, dataset snapshots, and run artifacts.
Back it up as a unit; restoring half a store (DB without snapshots) fails
closed on hash verification, which is the point.
