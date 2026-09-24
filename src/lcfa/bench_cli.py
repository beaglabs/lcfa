"""Command-line interface for LCFA-Bench."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from time import perf_counter
import sys
from typing import Any, Mapping

from .artifact import load_artifact_reasoner
from .bench import BenchmarkReport, BenchmarkRunner, BenchmarkSuite, ReasonerSubject, compare_reports, dumps_benchmark_report, dumps_benchmark_suite, dumps_comparison, load_benchmark_report, load_benchmark_suite
from .bench_corpus import all_suites
from .engine import LCFA


class _ProgressPrinter:
    """Human-readable progress to stderr so JSON/stdout remains machine-safe."""

    def __init__(self, *, suite: BenchmarkSuite, repeats: int, warmup: int) -> None:
        self.suite = suite
        self.repeats = repeats
        self.warmup = warmup
        self.calls_per_case = repeats + warmup
        self.total_calls = len(suite.cases) * self.calls_per_case
        self.reason_call = 0
        self.backbone_call = 0

    def _print(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def start(self, subject_hint: str) -> None:
        self._print(
            f"[lcfa] loading {subject_hint}; suite={self.suite.id} "
            f"cases={len(self.suite.cases)} repeats={self.repeats} warmup={self.warmup}"
        )

    def loaded(self, subject: ReasonerSubject) -> None:
        backend = subject.metadata.get("backend") if isinstance(subject.metadata, Mapping) else None
        suffix = f" backend={backend}" if backend else ""
        self._print(f"[lcfa] subject ready: {subject.name}{suffix}")

    def wrap_subject(self, subject: ReasonerSubject) -> ReasonerSubject:
        original = subject.reason

        def reason(plan, context):
            self.reason_call += 1
            call = self.reason_call
            if self.calls_per_case:
                case_index = min((call - 1) // self.calls_per_case + 1, len(self.suite.cases))
                slot = (call - 1) % self.calls_per_case + 1
            else:
                case_index = 1
                slot = 1
            case_id = (
                self.suite.cases[case_index - 1].id
                if self.suite.cases and 0 < case_index <= len(self.suite.cases)
                else plan.id
            )
            if slot <= self.warmup:
                phase = f"warmup {slot}/{self.warmup}"
            else:
                repeat = slot - self.warmup
                phase = f"repeat {repeat}/{self.repeats}"
            self._print(
                f"[lcfa] case {case_index}/{len(self.suite.cases)} {case_id} "
                f"({phase}) start [{call}/{self.total_calls}]"
            )
            started = perf_counter()
            try:
                result = original(plan, context)
            except Exception as exc:
                elapsed = perf_counter() - started
                self._print(
                    f"[lcfa] case {case_index}/{len(self.suite.cases)} {case_id} "
                    f"error after {elapsed:.1f}s: {type(exc).__name__}: {exc}"
                )
                raise
            elapsed = perf_counter() - started
            flow = result.metadata.get("stochastic_flow", {}) if result.metadata else {}
            adaptive = flow.get("adaptive_compute", {}) if isinstance(flow, Mapping) else {}
            detail = ""
            if isinstance(adaptive, Mapping) and adaptive.get("enabled"):
                detail = (
                    f" steps={adaptive.get('steps_used')}"
                    f" candidates={adaptive.get('generated_candidates')}"
                    f" verifiers={adaptive.get('verifier_calls')}"
                )
            self._print(
                f"[lcfa] case {case_index}/{len(self.suite.cases)} {case_id} "
                f"done in {elapsed:.1f}s{detail}"
            )
            return result

        return ReasonerSubject(subject.name, reason, metadata=subject.metadata)

    def note(self, payload: Mapping[str, Any]) -> None:
        event = str(payload.get("event", ""))
        if event == "adaptive_reason_start":
            self._print(
                f"[lcfa]   adaptive budget: start={payload.get('initial_branches')} branch, "
                f"max={payload.get('max_branches')} branches x {payload.get('max_steps')} steps"
            )
        elif event == "adaptive_step_start":
            self._print(
                f"[lcfa]   step {payload.get('step')}/{payload.get('max_steps')} "
                f"parents={payload.get('parents')} initial_branches={payload.get('initial_branches')}"
            )
        elif event == "adaptive_expand":
            reasons = ",".join(payload.get("reasons", [])) or "uncertain"
            self._print(
                f"[lcfa]   adaptive expand: +{payload.get('added_candidates')} candidate(s) "
                f"because {reasons}"
            )
        elif event == "adaptive_verify":
            reasons = ",".join(payload.get("reasons", [])) or "resolved"
            self._print(
                f"[lcfa]   verifier: {payload.get('verifier_calls')} call(s); remaining={reasons}"
            )
        elif event == "adaptive_step_complete":
            reasons = ",".join(payload.get("reasons", [])) or "none"
            action = "stop" if payload.get("stop") else "continue"
            self._print(
                f"[lcfa]   step {payload.get('step')} -> {action}; "
                f"confidence={float(payload.get('confidence', 0.0)):.2f} "
                f"evidence={float(payload.get('evidence_score', 0.0)):.2f} "
                f"uncertainty={reasons}"
            )
        elif event == "adaptive_reason_complete":
            self._print(
                f"[lcfa]   adaptive complete: steps={payload.get('steps_used')} "
                f"candidates={payload.get('generated_candidates')} "
                f"verifiers={payload.get('verifier_calls')} expansions={payload.get('expansions')}"
            )

    def complete(self, report: BenchmarkReport) -> None:
        summary = report.summary
        self._print(
            f"[lcfa] complete: {summary.passed_cases}/{summary.case_count} cases passed; "
            f"assertions={summary.assertion_accuracy:.3f} error_rate={summary.error_rate:.3f}"
        )


class _ProgressBackbone:
    """Transparent backbone proxy that reports each expensive model invocation."""

    def __init__(self, inner: Any, progress: _ProgressPrinter) -> None:
        self.inner = inner
        self.progress = progress
        self.metadata = getattr(inner, "metadata", {})

    def note(self, payload: Mapping[str, Any]) -> None:
        self.progress.note(payload)

    def sample(self, **kwargs: Any):
        self.progress.backbone_call += 1
        call = self.progress.backbone_call
        system_prompt = str(kwargs.get("system_prompt", ""))
        kind = "verify" if "LCFA_VERIFIER" in system_prompt else "proposal"
        branches = int(kwargs.get("branches", 1))
        max_new_tokens = int(kwargs.get("max_new_tokens", 0))
        self.progress._print(
            f"[lcfa]     model call {call} {kind}: branches={branches} max_new_tokens={max_new_tokens}"
        )
        started = perf_counter()
        try:
            result = self.inner.sample(**kwargs)
        except Exception as exc:
            elapsed = perf_counter() - started
            self.progress._print(
                f"[lcfa]     model call {call} {kind} failed after {elapsed:.1f}s: "
                f"{type(exc).__name__}: {exc}"
            )
            raise
        elapsed = perf_counter() - started
        self.progress._print(
            f"[lcfa]     model call {call} {kind} done in {elapsed:.1f}s; samples={len(result)}"
        )
        return result


def _load_factory(spec: str) -> Any:
    module_name, separator, attr = spec.partition(":")
    if not separator or not module_name or not attr:
        raise ValueError("subject factory must use module:callable syntax")
    return getattr(importlib.import_module(module_name), attr)()


def _subject_from_args(args: argparse.Namespace, progress: _ProgressPrinter | None = None) -> ReasonerSubject:
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
        if progress is not None and hasattr(reasoner, "backbone"):
            reasoner.backbone = _ProgressBackbone(reasoner.backbone, progress)
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
    run.add_argument("--quiet", action="store_true", help="disable human-readable progress reporting on stderr")
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
        suite = _load_suite(args.suite)
        progress = None if args.quiet else _ProgressPrinter(suite=suite, repeats=args.repeats, warmup=args.warmup)
        if progress is not None:
            progress.start(args.artifact or args.factory or args.name or "lcfa-zero")
        subject = _subject_from_args(args, progress)
        if progress is not None:
            progress.loaded(subject)
            subject = progress.wrap_subject(subject)
        report = BenchmarkRunner(repeats=args.repeats, warmup=args.warmup).run(subject, suite)
        if progress is not None:
            progress.complete(report)
        text = dumps_benchmark_report(report)
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
