"""Command-line interface for LCFA-Bench."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from .artifact import load_artifact_reasoner
from .bench import (
    BenchmarkRunner,
    BenchmarkSuite,
    ReasonerSubject,
    compare_reports,
    dumps_benchmark_report,
    dumps_benchmark_suite,
    dumps_comparison,
    load_benchmark_report,
    load_benchmark_suite,
)
from .bench_corpus import all_suites
from .engine import LCFA


def _load_factory(spec: str) -> Any:
    module_name, separator, attr = spec.partition(":")
    if not separator or not module_name or not attr:
        raise ValueError("subject factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory()


def _subject_from_args(args: argparse.Namespace) -> ReasonerSubject:
    base_engine = LCFA.from_profile(args.profile) if args.profile else LCFA()

    if args.artifact:
        reasoner = load_artifact_reasoner(args.artifact, base_engine=base_engine)
        return ReasonerSubject(
            name=args.name or reasoner.artifact.id,
            reasoner=reasoner.reason,
            metadata=dict(reasoner.metadata),
        )

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

    return ReasonerSubject.from_engine(
        args.name or "lcfa-zero",
        base_engine,
        metadata={
            "backend": "deterministic",
            "profile": base_engine.profile.id if base_engine.profile is not None else None,
        },
    )


def _builtins() -> dict[str, BenchmarkSuite]:
    suites = all_suites()
    result = {suite.id.removeprefix("lcfa-core-"): suite for suite in suites}
    result.update({suite.id: suite for suite in suites})
    result["all"] = BenchmarkSuite(
        id="lcfa-core-all",
        cases=tuple(case for suite in suites for case in suite.cases),
        metadata={
            "family": "all",
            "source_suites": [suite.id for suite in suites],
        },
    )
    return result


def _load_suite(spec: str) -> BenchmarkSuite:
    if spec.startswith("builtin:"):
        name = spec.removeprefix("builtin:")
        try:
            return _builtins()[name]
        except KeyError as exc:
            available = ", ".join(sorted(_builtins()))
            raise ValueError(f"unknown built-in suite {name!r}; available: {available}") from exc
    return load_benchmark_suite(spec)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcfa-bench",
        description="Run and compare LCFA reasoner benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a serialized or built-in benchmark suite")
    run.add_argument(
        "suite",
        help="path to an lcfa.bench.v1 suite or builtin:<family>/builtin:all",
    )
    run.add_argument("--name", help="subject name in the report")
    run.add_argument("--profile", help="LCFA profile path for the built-in/base runtime")
    backend = run.add_mutually_exclusive_group()
    backend.add_argument(
        "--factory",
        help="module:callable returning an engine with reason(plan, context)",
    )
    backend.add_argument(
        "--artifact",
        help="path to an lcfa.artifact.v1 directory or artifact.json",
    )
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--output", "-o", help="write report JSON to this path")

    compare = subparsers.add_parser("compare", help="compare benchmark reports")
    compare.add_argument("reports", nargs="+", help="lcfa.bench.report.v1 files")
    compare.add_argument("--baseline", help="subject name to use as the baseline")
    compare.add_argument("--output", "-o", help="write comparison JSON to this path")

    list_command = subparsers.add_parser("list", help="list built-in benchmark suites")
    list_command.add_argument("--json", action="store_true", help="emit suite metadata as JSON")

    export = subparsers.add_parser("export", help="export a built-in suite as lcfa.bench.v1 JSON")
    export.add_argument("suite", help="builtin:<family> or builtin:all")
    export.add_argument("--output", "-o", required=True, help="write suite JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        suite = _load_suite(args.suite)
        subject = _subject_from_args(args)
        report = BenchmarkRunner(repeats=args.repeats, warmup=args.warmup).run(
            subject,
            suite,
        )
        text = dumps_benchmark_report(report)
    elif args.command == "compare":
        reports = tuple(load_benchmark_report(path) for path in args.reports)
        text = dumps_comparison(compare_reports(reports, baseline_subject=args.baseline))
    elif args.command == "export":
        suite = _load_suite(args.suite)
        Path(args.output).write_text(dumps_benchmark_suite(suite) + "\n", encoding="utf-8")
        return 0
    else:
        suites = all_suites()
        if args.json:
            import json

            print(
                json.dumps(
                    [
                        {
                            "id": suite.id,
                            "alias": suite.id.removeprefix("lcfa-core-"),
                            "case_count": len(suite.cases),
                            "metadata": dict(suite.metadata),
                        }
                        for suite in suites
                    ],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            for suite in suites:
                alias = suite.id.removeprefix("lcfa-core-")
                print(f"{alias:12} {len(suite.cases):>3} cases  {suite.id}")
            print(f"{'all':12} {sum(len(suite.cases) for suite in suites):>3} cases  lcfa-core-all")
        return 0

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
