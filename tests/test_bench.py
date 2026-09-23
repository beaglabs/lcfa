from __future__ import annotations

from dataclasses import replace

import pytest

from lcfa import (
    BenchmarkCase,
    BenchmarkRunner,
    BenchmarkSuite,
    EvidenceRef,
    EvidenceValue,
    ExecutionContext,
    ExpectedValue,
    LCFA,
    PlanNode,
    ReasonerSubject,
    ReasoningPlan,
    compare_reports,
    dumps_benchmark_report,
    dumps_benchmark_suite,
    loads_benchmark_report,
    loads_benchmark_suite,
)


def _plan() -> ReasoningPlan:
    return ReasoningPlan(
        id="bench-delta",
        nodes=(
            PlanNode("current", "stats.mean", {"values": "$state.current"}),
            PlanNode("baseline", "stats.mean", {"values": "$state.baseline"}),
            PlanNode(
                "delta",
                "stats.delta",
                {"current": "$node.current", "baseline": "$node.baseline"},
                depends_on=("current", "baseline"),
            ),
        ),
        outputs=("delta",),
    )


def _context() -> ExecutionContext:
    return ExecutionContext(
        state={
            "current": EvidenceValue(
                [70, 72, 71],
                (EvidenceRef("obs:current", source="bench"),),
            ),
            "baseline": EvidenceValue(
                [79, 80, 78],
                (EvidenceRef("obs:baseline", source="bench"),),
            ),
        }
    )


def _suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        id="core-smoke",
        cases=(
            BenchmarkCase(
                id="delta",
                plan=_plan(),
                context=_context(),
                expectations=(
                    ExpectedValue("values.delta", -8.0, abs_tol=1e-9),
                ),
                expected_evidence_ids=("obs:current", "obs:baseline"),
                tags=("numeric", "evidence"),
            ),
        ),
    )


def test_benchmark_suite_round_trips_with_plan_and_context_ir() -> None:
    restored = loads_benchmark_suite(dumps_benchmark_suite(_suite()))
    assert restored == _suite()


def test_lcfa_zero_scores_correctly_and_replays_deterministically() -> None:
    subject = ReasonerSubject.from_engine("lcfa-zero", LCFA())
    report = BenchmarkRunner(repeats=3).run(subject, _suite())

    assert report.summary.case_pass_rate == 1.0
    assert report.summary.assertion_accuracy == 1.0
    assert report.summary.evidence_precision == 1.0
    assert report.summary.evidence_recall == 1.0
    assert report.summary.evidence_f1 == 1.0
    assert report.summary.replay_determinism_rate == 1.0
    assert report.summary.error_rate == 0.0
    assert report.summary.total_repeats == 3
    assert report.summary.latency_mean_ms is not None
    assert report.cases[0].replay_deterministic is True
    assert report.tag_summaries["numeric"].assertion_accuracy == 1.0
    assert report.tag_summaries["evidence"].evidence_f1 == 1.0


def test_report_round_trip_preserves_metrics() -> None:
    report = BenchmarkRunner(repeats=2).run(
        ReasonerSubject.from_engine("lcfa-zero", LCFA()),
        _suite(),
    )
    restored = loads_benchmark_report(dumps_benchmark_report(report))

    assert restored.subject_name == report.subject_name
    assert restored.summary == report.summary
    assert restored.cases[0].assertion_passes == 2


def test_stochastic_semantics_are_reported_separately_from_correctness() -> None:
    engine = LCFA()
    counter = {"value": 0}

    def alternating(plan, context):
        solution = engine.reason(plan, context)
        counter["value"] += 1
        if counter["value"] % 2 == 0:
            return replace(
                solution,
                values={"delta": solution.values["delta"] + 1.0},
            )
        return solution

    report = BenchmarkRunner(repeats=4).run(
        ReasonerSubject("alternating", alternating),
        _suite(),
    )

    assert report.summary.replay_determinism_rate == 0.0
    assert report.summary.assertion_accuracy == pytest.approx(0.5)
    assert report.summary.case_pass_rate == 0.0


def test_compare_reports_uses_same_suite_and_exposes_metric_deltas() -> None:
    zero = BenchmarkRunner(repeats=2).run(
        ReasonerSubject.from_engine("zero", LCFA()),
        _suite(),
    )

    engine = LCFA()

    def shifted(plan, context):
        solution = engine.reason(plan, context)
        return replace(solution, values={"delta": -7.0})

    other = BenchmarkRunner(repeats=2).run(
        ReasonerSubject("shifted", shifted),
        _suite(),
    )

    comparison = compare_reports((zero, other), baseline_subject="zero")

    assert comparison.baseline_subject == "zero"
    assert comparison.metrics["assertion_accuracy"]["zero"] == 1.0
    assert comparison.metrics["assertion_accuracy"]["shifted"] == 0.0
    assert comparison.deltas_from_baseline["assertion_accuracy"]["shifted"] == -1.0
    assert comparison.deltas_from_baseline["latency_mean_ms"]["zero"] == 0.0


def test_runner_captures_subject_errors_without_crashing_suite() -> None:
    def broken(_plan, _context):
        raise RuntimeError("backend unavailable")

    report = BenchmarkRunner(repeats=2).run(
        ReasonerSubject("broken", broken),
        _suite(),
    )

    assert report.summary.error_rate == 1.0
    assert report.summary.case_pass_rate == 0.0
    assert report.cases[0].error == "RuntimeError: backend unavailable"
    assert report.cases[0].repeats_completed == 0
