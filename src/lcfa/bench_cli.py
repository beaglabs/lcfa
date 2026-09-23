"""Command-line interface for LCFA-Bench."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from .bench import (
    BenchmarkRunner,
    ReasonerSubject,
    compare_reports,
    dumps_benchmark_report,
    dumps_comparison,
    load_benchmark_report,
    load_benchmark_suite,
)
from .engine import LCFA


def _load_factory(spec: str) -> Any:
    module_name, separator, attr = spec.partition(":")
    if not separator or not module_name or not attr:
        raise ValueError("subject factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory()


def _subject_from_args(args: argparse.Namespace) -> ReasonerSubject:
    if args.factory:
        engine = _load_factory(args.factory)
        if isinstance(engine, ReasonerSubject):
            if args.name and args.name != engine.name:
                return ReasonerSubject(
                    name=args.name,
                    reasoner=engine.reason,
                    metadata=engine.metadata,
                )
            return engine
        return ReasonerSubject.from_engine(
            args.name or args.factory,
            engine,
            metadata={"factory": args.factory},
        )

    engine = LCFA.from_profile(args.profile) if args.profile else LCFA()
    return ReasonerSubject.from_engine(
        args.name or "lcfa-zero",
        engine,
        metadata={
            "backend": "deterministic",
            "profile": engine.profile.id if engine.profile is not None else None,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcfa-bench",
        description="Run and compare LCFA reasoner benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a serialized benchmark suite")
    run.add_argument("suite", help="path to an lcfa.bench.v1 suite")
    run.add_argument("--name", help="subject name in the report")
    run.add_argument("--profile", help="LCFA profile path for the built-in runtime")
    run.add_argument(
        "--factory",
        help="module:callable returning an engine with reason(plan, context)",
    )
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--output", "-o", help="write report JSON to this path")

    compare = subparsers.add_parser("compare", help="compare benchmark reports")
    compare.add_argument("reports", nargs="+", help="lcfa.bench.report.v1 files")
    compare.add_argument("--baseline", help="subject name to use as the baseline")
    compare.add_argument("--output", "-o", help="write comparison JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        suite = load_benchmark_suite(args.suite)
        subject = _subject_from_args(args)
        report = BenchmarkRunner(repeats=args.repeats, warmup=args.warmup).run(
            subject,
            suite,
        )
        text = dumps_benchmark_report(report)
    else:
        reports = tuple(load_benchmark_report(path) for path in args.reports)
        text = dumps_comparison(
            compare_reports(reports, baseline_subject=args.baseline)
        )

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
