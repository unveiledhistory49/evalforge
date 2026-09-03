"""EvalForge CLI (argparse only). Every command prints JSON to stdout."""

from __future__ import annotations

import argparse
import json

from sqlalchemy.orm import Session

from evalforge.compare import compare_runs
from evalforge.datasets import register_dataset, synthesize_binary
from evalforge.db import create_app_engine, create_session_factory, init_db
from evalforge.gates import check_gate
from evalforge.runs import list_runs, load_model, reproduce_run, run_evaluation, run_report
from evalforge.seeds import SeedBundle, runtime_report


def _session() -> Session:
    engine = create_app_engine()
    init_db(engine)
    return create_session_factory(engine)()


def _parse_params(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"bad --param {item!r}: want key=value")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def cmd_init(args: argparse.Namespace) -> int:
    db = _session()
    db.close()
    from evalforge.db import store_path

    print(json.dumps({"store": store_path(), "runtime": runtime_report()}))
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    db = _session()
    try:
        rec = register_dataset(db, args.name, args.path)
        db.commit()
        print(
            json.dumps(
                {
                    "name": rec.name,
                    "version": rec.version,
                    "sha256": rec.sha256,
                    "n": rec.n_examples,
                }
            )
        )
        return 0
    finally:
        db.close()


def cmd_synthesize(args: argparse.Namespace) -> int:
    path = synthesize_binary(args.path, args.n, args.seed, args.separation)
    print(json.dumps({"path": path, "n": args.n, "seed": args.seed}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    db = _session()
    try:
        model = load_model(args.model)
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        report = run_evaluation(
            db,
            model=model,
            model_ref=args.model,
            dataset_name=args.dataset,
            dataset_version=args.version,
            seed=args.seed,
            metrics=metrics,
            limit=args.limit,
            params=_parse_params(args.param),
            shuffle=args.shuffle,
            run_name=args.name or "",
        )
        db.commit()
        print(json.dumps({k: report[k] for k in ("id", "digest", "aggregates", "n_examples")}))
        return 0
    finally:
        db.close()


def cmd_reproduce(args: argparse.Namespace) -> int:
    db = _session()
    try:
        model = load_model(args.model) if args.model else None
        out = reproduce_run(db, args.run_id, model)
        db.commit()
        print(json.dumps(out))
        return 0 if out["match"] else 1
    finally:
        db.close()


def cmd_compare(args: argparse.Namespace) -> int:
    db = _session()
    try:
        out = compare_runs(
            db, args.run_a, args.run_b, args.metric, seed=args.seed, resamples=args.resamples
        )
        print(json.dumps(out, default=str))
        return 0
    finally:
        db.close()


def cmd_gate(args: argparse.Namespace) -> int:
    db = _session()
    try:
        if args.run:
            report = run_report(db, args.run)
            metrics = dict(report["aggregates"])
        else:
            metrics = json.loads(args.metrics_json or "{}")
        rules = json.loads(args.rules)
        baseline = json.loads(args.baseline) if args.baseline else None
        out = check_gate(metrics, rules, baseline)
        print(json.dumps(out))
        return 0 if out["passed"] else 1
    finally:
        db.close()


def cmd_report(args: argparse.Namespace) -> int:
    db = _session()
    try:
        print(json.dumps(run_report(db, args.run_id), default=str))
        return 0
    finally:
        db.close()


def cmd_list(args: argparse.Namespace) -> int:
    db = _session()
    try:
        rows = list_runs(db, args.limit)
        print(
            json.dumps(
                [
                    {
                        "id": r.id,
                        "name": r.name,
                        "model": r.model_ref,
                        "dataset": f"{r.dataset_name}v{r.dataset_version}",
                        "seed": r.seed,
                        "digest": r.digest[:12],
                    }
                    for r in rows
                ]
            )
        )
        return 0
    finally:
        db.close()


def cmd_seed_info(args: argparse.Namespace) -> int:
    b = SeedBundle(args.seed)
    print(json.dumps({"provenance": b.apply(), "runtime": runtime_report()}, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalforge")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)

    r = sub.add_parser("register-dataset")
    r.add_argument("name")
    r.add_argument("path")
    r.set_defaults(fn=cmd_register)

    s = sub.add_parser("synthesize")
    s.add_argument("--path", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--separation", type=float, default=2.0)
    s.set_defaults(fn=cmd_synthesize)

    u = sub.add_parser("run")
    u.add_argument("--model", required=True, help="module:attr callable")
    u.add_argument("--dataset", required=True)
    u.add_argument("--version", type=int, default=None)
    u.add_argument("--seed", type=int, required=True)
    u.add_argument("--metrics", required=True, help="comma-separated")
    u.add_argument("--limit", type=int, default=None)
    u.add_argument("--param", action="append", default=[], help="key=value")
    u.add_argument("--shuffle", action="store_true")
    u.add_argument("--name", default="")
    u.set_defaults(fn=cmd_run)

    rp = sub.add_parser("reproduce")
    rp.add_argument("run_id")
    rp.add_argument("--model", default=None)
    rp.set_defaults(fn=cmd_reproduce)

    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--metric", required=True)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--resamples", type=int, default=2000)
    c.set_defaults(fn=cmd_compare)

    g = sub.add_parser("gate")
    g.add_argument("--run", default=None)
    g.add_argument("--metrics-json", default=None)
    g.add_argument("--rules", required=True, help="JSON rules")
    g.add_argument("--baseline", default=None, help="JSON baseline values")
    g.set_defaults(fn=cmd_gate)

    for name, fn in (("report", cmd_report), ("list", cmd_list), ("seed-info", cmd_seed_info)):
        q = sub.add_parser(name)
        if name == "report":
            q.add_argument("run_id")
        if name == "list":
            q.add_argument("--limit", type=int, default=50)
        if name == "seed-info":
            q.add_argument("--seed", type=int, required=True)
        q.set_defaults(fn=fn)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.fn(args))


if __name__ == "__main__":
    main()
