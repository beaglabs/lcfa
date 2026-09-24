"""CLI for the balanced mixed real-world LCFA benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from .bench import dumps_benchmark_report, dumps_benchmark_suite, load_benchmark_suite
from .bench_cli import _ProgressPrinter, _subject_from_args
from .bench_mixed import (
    DEFAULT_SEED,
    MIXED_PRESETS,
    MixedBenchmarkRunner,
    build_mixed_suite,
)


def _add_subject_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name")
    parser.add_argument("--profile", help="LCFA profile path for the built-in/base runtime")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--factory")
    backend.add_argument("--artifact")
    parser.add_argument(
        "--backbone-type",
        choices=("transformers-local", "mlx-local", "llama-cpp", "reference"),
        help="override config.backbone.type",
    )
    parser.add_argument("--backbone-path", help="override config.backbone.path")
    parser.add_argument("--device-map", help="Transformers device_map override")
    parser.add_argument("--dtype", help="Transformers dtype override")
    parser.add_argument("--n-ctx", type=int, help="llama.cpp context size")
    parser.add_argument("--n-threads", type=int, help="llama.cpp CPU thread count")
    parser.add_argument("--n-gpu-layers", type=int, help="llama.cpp GPU/Metal layer count")
    parser.add_argument(
        "--adaptation-type",
        choices=("none", "numpy-fast", "mlx-fast"),
        help="override config.adaptation.type",
    )
    parser.add_argument("--adaptation-learning-rate", type=float)
    parser.add_argument("--adaptation-decay", type=float)
    parser.add_argument("--prior-snapshot", help="fast-prior safetensors snapshot path")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcfa-bench-mixed",
        description=(
            "Build and run a balanced LCFA benchmark from GSM8K, HumanEval, "
            "SWE-bench Lite, RULER, AA-LCR, and MMLU."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="download/sample and materialize the mixed suite")
    build.add_argument("--preset", choices=tuple(MIXED_PRESETS), default="quick")
    build.add_argument(
        "--cases-per-source",
        type=int,
        help="override preset size while keeping all six sources equally weighted",
    )
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--ruler-context", type=int, choices=(4096, 8192, 16384), default=4096)
    build.add_argument("--cache-dir", help="Hugging Face cache directory")
    build.add_argument("--output", "-o", required=True)

    run = subparsers.add_parser("run", help="run a materialized mixed suite")
    run.add_argument("suite", help="path produced by lcfa-bench-mixed build")
    _add_subject_args(run)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument(
        "--allow-code-exec",
        action="store_true",
        help="execute HumanEval candidate code in an isolated local Python subprocess",
    )
    run.add_argument(
        "--swebench-eval",
        action="store_true",
        help="run SWE-bench cases through the official Docker evaluation harness",
    )
    run.add_argument("--code-timeout", type=int, default=10, help="HumanEval seconds per candidate")
    run.add_argument(
        "--swebench-timeout",
        type=int,
        default=1800,
        help="SWE-bench test timeout in seconds per instance",
    )
    run.add_argument(
        "--lcr-judge-cmd",
        help=(
            "optional command receiving JSON on stdin and returning {\"passed\": bool}; "
            "otherwise AA-LCR uses conservative local equivalence"
        ),
    )
    run.add_argument("--quiet", action="store_true", help="disable progress reporting")
    run.add_argument("--output", "-o")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "build":
        suite = build_mixed_suite(
            preset=args.preset,
            cases_per_source=args.cases_per_source,
            seed=args.seed,
            ruler_context=args.ruler_context,
            cache_dir=args.cache_dir,
        )
        Path(args.output).write_text(dumps_benchmark_suite(suite) + "\n", encoding="utf-8")
        print(
            f"wrote {len(suite.cases)} cases "
            f"({suite.metadata['cases_per_source']} per source) to {args.output}"
        )
        return 0

    suite = load_benchmark_suite(args.suite)
    progress = None if args.quiet else _ProgressPrinter(
        suite=suite,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    if progress is not None:
        progress.start(args.artifact or args.factory or args.name or "lcfa-zero")
    subject = _subject_from_args(args, progress)
    if progress is not None:
        progress.loaded(subject)
        subject = progress.wrap_subject(subject)

    report = MixedBenchmarkRunner(
        repeats=args.repeats,
        warmup=args.warmup,
        allow_code_exec=args.allow_code_exec,
        swebench_eval=args.swebench_eval,
        code_timeout=args.code_timeout,
        swebench_timeout=args.swebench_timeout,
        lcr_judge_cmd=args.lcr_judge_cmd,
    ).run(subject, suite)

    if progress is not None:
        progress.complete(report)
    text = dumps_benchmark_report(report)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
