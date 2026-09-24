"""Command-line interface for LCFA-Bench."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from .artifact import load_artifact_reasoner
from .bench import BenchmarkRunner, BenchmarkSuite, ReasonerSubject, compare_reports, dumps_benchmark_report, dumps_benchmark_suite, dumps_comparison, load_benchmark_report, load_benchmark_suite
from .bench_corpus import all_suites
from .engine import LCFA


def _load_factory(spec: str) -> Any:
    module_name, separator, attr = spec.partition(":")
    if not separator or not module_name or not attr:
        raise ValueError("subject factory must use module:callable syntax")
    return getattr(importlib.import_module(module_name), attr)()


def _subject_from_args(args: argparse.Namespace) -> ReasonerSubject:
    base_engine = LCFA.from_profile(args.profile) if args.profile else LCFA()
    if args.artifact:
        runtime_options = {k: v for k, v in {
            "backbone_path": getattr(args, "backbone_path", None),
            "backbone_type": getattr(args, "backbone_type", None),
            "device_map": getattr(args, "device_map", None),
            "dtype": getattr(args, "dtype", None),
            "n_ctx": getattr(args, "n_ctx", None),
            "n_threads": getattr(args, "n_threads", None),
            "n_gpu_layers": getattr(args, "n_gpu_layers", None),
            "adaptation_type": getattr(args, "adaptation_type", None),
            "adaptation_learning_rate": getattr(args, "adaptation_learning_rate", None),
            "adaptation_decay": getattr(args, "adaptation_decay", None),
            "prior_snapshot": getattr(args, "prior_snapshot", None),
        }.items() if v is not None}
        reasoner = load_artifact_reasoner(args.artifact, base_engine=base_engine, runtime_options=runtime_options)
        return ReasonerSubject(name=args.name or reasoner.artifact.id, reasoner=reasoner.reason, metadata=dict(reasoner.metadata))
    if args.factory:
        engine = _load_factory(args.factory)
        if isinstance(engine, ReasonerSubject):
            if args.name and args.name != engine.name:
                return ReasonerSubject(name=args.name, reasoner=engine.reason, metadata=engine.metadata)
            return engine
        return ReasonerSubject.from_engine(args.name or args.factory, engine, metadata={"factory": args.factory})
    return ReasonerSubject.from_engine(args.name or "lcfa-zero", base_engine,
        metadata={"backend": "deterministic", "profile": base_engine.profile.id if base_engine.profile is not None else None})


def _builtins() -> dict[str, BenchmarkSuite]:
    suites = all_suites()
    result = {s.id.removeprefix("lcfa-core-"): s for s in suites}
    result.update({s.id: s for s in suites})
    result["all"] = BenchmarkSuite(id="lcfa-core-all", cases=tuple(c for s in suites for c in s.cases),
        metadata={"family": "all", "source_suites": [s.id for s in suites]})
    return result


def _load_suite(spec: str) -> BenchmarkSuite:
    if spec.startswith("builtin:"):
        name = spec.removeprefix("builtin:")
        try:
            return _builtins()[name]
        except KeyError as exc:
            raise ValueError(f"unknown built-in suite {name!r}; available: {', '.join(sorted(_builtins()))}") from exc
    return load_benchmark_suite(spec)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcfa-bench", description="Run and compare LCFA reasoner benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a serialized or built-in benchmark suite")
    run.add_argument("suite", help="path to an lcfa.bench.v1 suite or builtin:<family>/builtin:all")
    run.add_argument("--name")
    run.add_argument("--profile", help="LCFA profile path for the built-in/base runtime")
    backend = run.add_mutually_exclusive_group()
    backend.add_argument("--factory")
    backend.add_argument("--artifact")
    run.add_argument("--backbone-type", choices=("transformers-local", "mlx-local", "llama-cpp", "reference"), help="override config.backbone.type")
    run.add_argument("--backbone-path", help="override config.backbone.path")
    run.add_argument("--device-map", help="Transformers device_map override")
    run.add_argument("--dtype", help="Transformers dtype override")
    run.add_argument("--n-ctx", type=int, help="llama.cpp context size")
    run.add_argument("--n-threads", type=int, help="llama.cpp CPU thread count")
    run.add_argument("--n-gpu-layers", type=int, help="llama.cpp GPU/Metal layer count")
    run.add_argument("--adaptation-type", choices=("none", "numpy-fast", "mlx-fast"), help="override config.adaptation.type")
    run.add_argument("--adaptation-learning-rate", type=float)
    run.add_argument("--adaptation-decay", type=float)
    run.add_argument("--prior-snapshot", help="write final fast prior to a safetensors file")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--output", "-o")
    compare = subparsers.add_parser("compare")
    compare.add_argument("reports", nargs="+")
    compare.add_argument("--baseline")
    compare.add_argument("--output", "-o")
    listing = subparsers.add_parser("list")
    listing.add_argument("--json", action="store_true")
    export = subparsers.add_parser("export")
    export.add_argument("suite")
    export.add_argument("--output", "-o", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        text = dumps_benchmark_report(BenchmarkRunner(repeats=args.repeats, warmup=args.warmup).run(_subject_from_args(args), _load_suite(args.suite)))
    elif args.command == "compare":
        text = dumps_comparison(compare_reports(tuple(load_benchmark_report(p) for p in args.reports), baseline_subject=args.baseline))
    elif args.command == "export":
        Path(args.output).write_text(dumps_benchmark_suite(_load_suite(args.suite)) + "\n", encoding="utf-8")
        return 0
    else:
        suites = all_suites()
        if args.json:
            import json
            print(json.dumps([{"id": s.id, "alias": s.id.removeprefix("lcfa-core-"), "case_count": len(s.cases), "metadata": dict(s.metadata)} for s in suites], indent=2, sort_keys=True))
        else:
            for suite in suites:
                print(f"{suite.id.removeprefix('lcfa-core-'):12} {len(suite.cases):>3} cases  {suite.id}")
            print(f"{'all':12} {sum(len(s.cases) for s in suites):>3} cases  lcfa-core-all")
        return 0
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
