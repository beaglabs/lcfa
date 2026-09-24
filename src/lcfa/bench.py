"""LCFA-Bench: backend-neutral comparison harness for LCFA reasoners."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import json
import math
from pathlib import Path
import platform
from statistics import fmean
from time import perf_counter_ns
from typing import Any, Callable, Mapping, Protocol, Sequence

from .protocol import ExecutionContext, ReasoningPlan, SolutionState
from .serde import from_ir_dict, to_ir_dict

BENCH_FORMAT = "lcfa.bench.v1"
REPORT_FORMAT = "lcfa.bench.report.v1"
COMPARISON_FORMAT = "lcfa.bench.compare.v1"


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExpectedValue:
    path: str
    value: Any
    abs_tol: float = 0.0
    rel_tol: float = 0.0


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    plan: ReasoningPlan
    context: ExecutionContext
    expectations: tuple[ExpectedValue, ...] = ()
    expected_evidence_ids: tuple[str, ...] | None = None
    evidence_mode: str = "exact"
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_mode not in {"exact", "contains"}:
            raise BenchmarkError("evidence_mode must be 'exact' or 'contains'")


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    id: str
    cases: tuple[BenchmarkCase, ...]
    version: str = "1"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BenchmarkSubject(Protocol):
    name: str
    metadata: Mapping[str, Any]

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        ...


@dataclass(slots=True)
class ReasonerSubject:
    name: str
    reasoner: Callable[[ReasoningPlan, ExecutionContext], SolutionState]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        return self.reasoner(plan, context)

    @classmethod
    def from_engine(
        cls,
        name: str,
        engine: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ReasonerSubject":
        if not hasattr(engine, "reason"):
            raise TypeError("benchmark engine must expose reason(plan, context)")
        return cls(name=name, reasoner=engine.reason, metadata=dict(metadata or {}))


@dataclass(frozen=True, slots=True)
class AssertionResult:
    path: str
    passed: bool
    expected: Any
    actual: Any
    abs_tol: float
    rel_tol: float


@dataclass(frozen=True, slots=True)
class CaseResult:
    id: str
    passed: bool
    assertion_passes: int
    assertion_total: int
    evidence_tp: int
    evidence_fp: int
    evidence_fn: int
    evidence_checked: bool
    evidence_passed: bool | None
    replay_deterministic: bool | None
    repeats_requested: int
    repeats_completed: int
    latencies_ms: tuple[float, ...]
    trace_steps: tuple[int, ...]
    assertions: tuple[AssertionResult, ...] = ()
    error: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    case_count: int
    passed_cases: int
    case_pass_rate: float
    assertion_accuracy: float
    evidence_precision: float | None
    evidence_recall: float | None
    evidence_f1: float | None
    replay_determinism_rate: float | None
    error_rate: float
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    mean_trace_steps: float | None
    total_repeats: int


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    subject_name: str
    suite_id: str
    suite_version: str
    summary: BenchmarkSummary
    cases: tuple[CaseResult, ...]
    tag_summaries: Mapping[str, BenchmarkSummary] = field(default_factory=dict)
    subject_metadata: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    format: str = REPORT_FORMAT


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    baseline_subject: str
    subjects: tuple[str, ...]
    metrics: Mapping[str, Mapping[str, float | None]]
    deltas_from_baseline: Mapping[str, Mapping[str, float | None]]
    format: str = COMPARISON_FORMAT


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _deep_get(value: Any, path: str) -> Any:
    current = value
    for segment in path.split("."):
        if not segment:
            continue
        if isinstance(current, Mapping):
            current = current[segment]
        elif isinstance(current, (list, tuple)) and segment.isdigit():
            current = current[int(segment)]
        else:
            current = getattr(current, segment)
    return current


def _equivalent(actual: Any, expected: Any, *, abs_tol: float, rel_tol: float) -> bool:
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(float(actual), float(expected), abs_tol=abs_tol, rel_tol=rel_tol)

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            return False
        return all(
            _equivalent(actual[key], expected[key], abs_tol=abs_tol, rel_tol=rel_tol)
            for key in expected
        )

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _equivalent(a, e, abs_tol=abs_tol, rel_tol=rel_tol)
            for a, e in zip(actual, expected)
        )

    return actual == expected


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _solution_signature(solution: SolutionState) -> str:
    semantic = {
        "plan_id": solution.plan_id,
        "values": _json_safe(solution.values),
        "findings": _json_safe(solution.findings),
        "recommendations": _json_safe(solution.recommendations),
        "evidence": sorted(ref.id for ref in solution.evidence),
        "metadata": _json_safe(solution.metadata),
    }
    return json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _evidence_counts(
    solution: SolutionState,
    expected: tuple[str, ...] | None,
) -> tuple[int, int, int, bool]:
    if expected is None:
        return 0, 0, 0, False
    predicted = {ref.id for ref in solution.evidence}
    target = set(expected)
    return (
        len(predicted & target),
        len(predicted - target),
        len(target - predicted),
        True,
    )


def _evidence_passed(
    solution: SolutionState,
    expected: tuple[str, ...] | None,
    mode: str,
) -> bool | None:
    if expected is None:
        return None
    predicted = {ref.id for ref in solution.evidence}
    target = set(expected)
    if mode == "contains":
        return target <= predicted
    return target == predicted


class BenchmarkRunner:
    def __init__(self, *, repeats: int = 3, warmup: int = 0) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        self.repeats = repeats
        self.warmup = warmup

    def run(self, subject: BenchmarkSubject, suite: BenchmarkSuite) -> BenchmarkReport:
        results = tuple(self._run_case(subject, case) for case in suite.cases)
        tags = sorted({tag for result in results for tag in result.tags})
        return BenchmarkReport(
            subject_name=subject.name,
            suite_id=suite.id,
            suite_version=suite.version,
            summary=_summarize(results),
            cases=results,
            tag_summaries={
                tag: _summarize(tuple(result for result in results if tag in result.tags))
                for tag in tags
            },
            subject_metadata=dict(subject.metadata),
            environment={
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
        )

    def _run_case(self, subject: BenchmarkSubject, case: BenchmarkCase) -> CaseResult:
        for _ in range(self.warmup):
            try:
                subject.reason(case.plan, case.context)
            except Exception as exc:
                return CaseResult(
                    id=case.id,
                    passed=False,
                    assertion_passes=0,
                    assertion_total=0,
                    evidence_tp=0,
                    evidence_fp=0,
                    evidence_fn=0,
                    evidence_checked=case.expected_evidence_ids is not None,
                    evidence_passed=None,
                    replay_deterministic=None,
                    repeats_requested=self.repeats,
                    repeats_completed=0,
                    latencies_ms=(),
                    trace_steps=(),
                    error=f"{type(exc).__name__}: {exc}",
                    tags=case.tags,
                )

        solutions: list[SolutionState] = []
        latencies: list[float] = []
        trace_steps: list[int] = []
        assertion_passes = 0
        assertion_total = 0
        evidence_tp = evidence_fp = evidence_fn = 0
        evidence_checked = False
        all_evidence_passed = True
        first_assertions: tuple[AssertionResult, ...] = ()
        error: str | None = None

        for repeat_index in range(self.repeats):
            started = perf_counter_ns()
            try:
                solution = subject.reason(case.plan, case.context)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
            solutions.append(solution)
            latencies.append(elapsed_ms)
            trace_steps.append(len(solution.trace.steps) if solution.trace is not None else 0)

            current_assertions: list[AssertionResult] = []
            for expectation in case.expectations:
                try:
                    actual = _deep_get(solution, expectation.path)
                    passed = _equivalent(
                        actual,
                        expectation.value,
                        abs_tol=expectation.abs_tol,
                        rel_tol=expectation.rel_tol,
                    )
                except Exception:
                    actual = None
                    passed = False
                assertion_total += 1
                assertion_passes += int(passed)
                current_assertions.append(
                    AssertionResult(
                        path=expectation.path,
                        passed=passed,
                        expected=_json_safe(expectation.value),
                        actual=_json_safe(actual),
                        abs_tol=expectation.abs_tol,
                        rel_tol=expectation.rel_tol,
                    )
                )
            if repeat_index == 0:
                first_assertions = tuple(current_assertions)

            tp, fp, fn, checked = _evidence_counts(solution, case.expected_evidence_ids)
            evidence_tp += tp
            evidence_fp += fp
            evidence_fn += fn
            evidence_checked = evidence_checked or checked
            passed_evidence = _evidence_passed(
                solution,
                case.expected_evidence_ids,
                case.evidence_mode,
            )
            if passed_evidence is False:
                all_evidence_passed = False

        deterministic: bool | None
        if not solutions:
            deterministic = None
        elif len(solutions) == 1:
            deterministic = True
        else:
            signatures = {_solution_signature(solution) for solution in solutions}
            deterministic = len(signatures) == 1

        evidence_passed = all_evidence_passed if evidence_checked else None
        completed_all = len(solutions) == self.repeats
        assertions_all_passed = assertion_passes == assertion_total
        passed = completed_all and error is None and assertions_all_passed
        if evidence_checked:
            passed = passed and bool(evidence_passed)

        return CaseResult(
            id=case.id,
            passed=passed,
            assertion_passes=assertion_passes,
            assertion_total=assertion_total,
            evidence_tp=evidence_tp,
            evidence_fp=evidence_fp,
            evidence_fn=evidence_fn,
            evidence_checked=evidence_checked,
            evidence_passed=evidence_passed,
            replay_deterministic=deterministic,
            repeats_requested=self.repeats,
            repeats_completed=len(solutions),
            latencies_ms=tuple(latencies),
            trace_steps=tuple(trace_steps),
            assertions=first_assertions,
            error=error,
            tags=case.tags,
        )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _summarize(results: Sequence[CaseResult]) -> BenchmarkSummary:
    case_count = len(results)
    passed_cases = sum(int(result.passed) for result in results)
    assertion_passes = sum(result.assertion_passes for result in results)
    assertion_total = sum(result.assertion_total for result in results)

    tp = sum(result.evidence_tp for result in results)
    fp = sum(result.evidence_fp for result in results)
    fn = sum(result.evidence_fn for result in results)
    evidence_cases = [result for result in results if result.evidence_checked]

    if evidence_cases:
        precision = 1.0 if tp == fp == 0 else _safe_ratio(tp, tp + fp)
        recall = 1.0 if tp == fn == 0 else _safe_ratio(tp, tp + fn)
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = recall = f1 = None

    determinism_values = [
        result.replay_deterministic
        for result in results
        if result.replay_deterministic is not None
    ]
    latencies = [latency for result in results for latency in result.latencies_ms]
    steps = [count for result in results for count in result.trace_steps]

    return BenchmarkSummary(
        case_count=case_count,
        passed_cases=passed_cases,
        case_pass_rate=(passed_cases / case_count) if case_count else 0.0,
        assertion_accuracy=(
            assertion_passes / assertion_total if assertion_total else 1.0
        ),
        evidence_precision=precision,
        evidence_recall=recall,
        evidence_f1=f1,
        replay_determinism_rate=(
            sum(int(value) for value in determinism_values) / len(determinism_values)
            if determinism_values
            else None
        ),
        error_rate=(
            sum(int(result.error is not None) for result in results) / case_count
            if case_count
            else 0.0
        ),
        latency_mean_ms=fmean(latencies) if latencies else None,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        mean_trace_steps=fmean(steps) if steps else None,
        total_repeats=sum(result.repeats_completed for result in results),
    )


def benchmark_suite_to_dict(suite: BenchmarkSuite) -> dict[str, Any]:
    return {
        "format": BENCH_FORMAT,
        "id": suite.id,
        "version": suite.version,
        "metadata": _json_safe(suite.metadata),
        "cases": [
            {
                "id": case.id,
                "plan": to_ir_dict(case.plan),
                "context": to_ir_dict(case.context),
                "expectations": [
                    {
                        "path": expectation.path,
                        "value": _json_safe(expectation.value),
                        "abs_tol": expectation.abs_tol,
                        "rel_tol": expectation.rel_tol,
                    }
                    for expectation in case.expectations
                ],
                "expected_evidence_ids": (
                    list(case.expected_evidence_ids)
                    if case.expected_evidence_ids is not None
                    else None
                ),
                "evidence_mode": case.evidence_mode,
                "tags": list(case.tags),
                "metadata": _json_safe(case.metadata),
            }
            for case in suite.cases
        ],
    }


def benchmark_suite_from_dict(data: Mapping[str, Any]) -> BenchmarkSuite:
    if data.get("format") != BENCH_FORMAT:
        raise BenchmarkError(f"unsupported benchmark format: {data.get('format')!r}")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise BenchmarkError("benchmark cases must be an array")

    cases: list[BenchmarkCase] = []
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise BenchmarkError("benchmark case must be an object")
        plan = from_ir_dict(raw["plan"])
        context = from_ir_dict(raw["context"])
        if not isinstance(plan, ReasoningPlan):
            raise BenchmarkError("benchmark plan must decode to ReasoningPlan")
        if not isinstance(context, ExecutionContext):
            raise BenchmarkError("benchmark context must decode to ExecutionContext")
        expectations = tuple(
            ExpectedValue(
                path=str(item["path"]),
                value=item.get("value"),
                abs_tol=float(item.get("abs_tol", 0.0)),
                rel_tol=float(item.get("rel_tol", 0.0)),
            )
            for item in raw.get("expectations", [])
        )
        evidence_ids = raw.get("expected_evidence_ids")
        cases.append(
            BenchmarkCase(
                id=str(raw["id"]),
                plan=plan,
                context=context,
                expectations=expectations,
                expected_evidence_ids=(
                    tuple(str(item) for item in evidence_ids)
                    if evidence_ids is not None
                    else None
                ),
                evidence_mode=str(raw.get("evidence_mode", "exact")),
                tags=tuple(str(item) for item in raw.get("tags", [])),
                metadata=dict(raw.get("metadata", {})),
            )
        )
    return BenchmarkSuite(
        id=str(data["id"]),
        version=str(data.get("version", "1")),
        cases=tuple(cases),
        metadata=dict(data.get("metadata", {})),
    )


def dumps_benchmark_suite(suite: BenchmarkSuite, *, indent: int | None = 2) -> str:
    return json.dumps(
        benchmark_suite_to_dict(suite),
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def loads_benchmark_suite(text: str) -> BenchmarkSuite:
    raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise BenchmarkError("benchmark suite must be a JSON object")
    return benchmark_suite_from_dict(raw)


def dump_benchmark_suite(suite: BenchmarkSuite, path: str | Path) -> None:
    Path(path).write_text(dumps_benchmark_suite(suite) + "\n", encoding="utf-8")


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    return loads_benchmark_suite(Path(path).read_text(encoding="utf-8"))


def benchmark_report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return _json_safe(report)


def benchmark_report_from_dict(data: Mapping[str, Any]) -> BenchmarkReport:
    if data.get("format") != REPORT_FORMAT:
        raise BenchmarkError(f"unsupported report format: {data.get('format')!r}")
    summary = BenchmarkSummary(**dict(data["summary"]))
    cases = tuple(
        CaseResult(
            id=str(item["id"]),
            passed=bool(item["passed"]),
            assertion_passes=int(item["assertion_passes"]),
            assertion_total=int(item["assertion_total"]),
            evidence_tp=int(item["evidence_tp"]),
            evidence_fp=int(item["evidence_fp"]),
            evidence_fn=int(item["evidence_fn"]),
            evidence_checked=bool(item["evidence_checked"]),
            evidence_passed=item.get("evidence_passed"),
            replay_deterministic=item.get("replay_deterministic"),
            repeats_requested=int(item["repeats_requested"]),
            repeats_completed=int(item["repeats_completed"]),
            latencies_ms=tuple(float(v) for v in item.get("latencies_ms", [])),
            trace_steps=tuple(int(v) for v in item.get("trace_steps", [])),
            assertions=tuple(
                AssertionResult(**dict(assertion))
                for assertion in item.get("assertions", [])
            ),
            error=item.get("error"),
            tags=tuple(str(tag) for tag in item.get("tags", [])),
        )
        for item in data.get("cases", [])
    )
    tag_summaries = {
        str(tag): BenchmarkSummary(**dict(value))
        for tag, value in data.get("tag_summaries", {}).items()
    }
    return BenchmarkReport(
        subject_name=str(data["subject_name"]),
        suite_id=str(data["suite_id"]),
        suite_version=str(data["suite_version"]),
        summary=summary,
        cases=cases,
        tag_summaries=tag_summaries,
        subject_metadata=dict(data.get("subject_metadata", {})),
        environment=dict(data.get("environment", {})),
    )


def dumps_benchmark_report(report: BenchmarkReport, *, indent: int | None = 2) -> str:
    return json.dumps(
        benchmark_report_to_dict(report),
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def loads_benchmark_report(text: str) -> BenchmarkReport:
    raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise BenchmarkError("benchmark report must be a JSON object")
    return benchmark_report_from_dict(raw)


def dump_benchmark_report(report: BenchmarkReport, path: str | Path) -> None:
    Path(path).write_text(dumps_benchmark_report(report) + "\n", encoding="utf-8")


def load_benchmark_report(path: str | Path) -> BenchmarkReport:
    return loads_benchmark_report(Path(path).read_text(encoding="utf-8"))


_COMPARISON_METRICS = (
    "case_pass_rate",
    "assertion_accuracy",
    "evidence_precision",
    "evidence_recall",
    "evidence_f1",
    "replay_determinism_rate",
    "error_rate",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "mean_trace_steps",
)


def compare_reports(
    reports: Sequence[BenchmarkReport],
    *,
    baseline_subject: str | None = None,
) -> ComparisonReport:
    if len(reports) < 2:
        raise BenchmarkError("comparison requires at least two reports")
    suite_ids = {(report.suite_id, report.suite_version) for report in reports}
    if len(suite_ids) != 1:
        raise BenchmarkError("reports must come from the same benchmark suite/version")
    by_name = {report.subject_name: report for report in reports}
    if len(by_name) != len(reports):
        raise BenchmarkError("subject names must be unique")
    baseline_name = baseline_subject or reports[0].subject_name
    if baseline_name not in by_name:
        raise BenchmarkError(f"unknown baseline subject: {baseline_name}")

    metrics: dict[str, dict[str, float | None]] = {}
    deltas: dict[str, dict[str, float | None]] = {}
    baseline = by_name[baseline_name]
    for metric in _COMPARISON_METRICS:
        metric_values = {
            name: getattr(report.summary, metric)
            for name, report in by_name.items()
        }
        baseline_value = getattr(baseline.summary, metric)
        metric_deltas: dict[str, float | None] = {}
        for name, value in metric_values.items():
            if value is None or baseline_value is None:
                metric_deltas[name] = None
            else:
                metric_deltas[name] = float(value) - float(baseline_value)
        metrics[metric] = metric_values
        deltas[metric] = metric_deltas

    return ComparisonReport(
        baseline_subject=baseline_name,
        subjects=tuple(by_name),
        metrics=metrics,
        deltas_from_baseline=deltas,
    )


def dumps_comparison(report: ComparisonReport, *, indent: int | None = 2) -> str:
    return json.dumps(_json_safe(report), indent=indent, sort_keys=True, allow_nan=False)
