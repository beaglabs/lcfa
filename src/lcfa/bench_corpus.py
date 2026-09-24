"""Built-in LCFA-Bench reference corpus.

The corpus is deliberately domain-neutral and deterministic. It defines the
first stable workloads that LCFA-Zero and future learned/safetensors reasoners
must execute through the same ReasoningPlan -> SolutionState contract.
"""

from __future__ import annotations

from .bench import BenchmarkCase, BenchmarkSuite, ExpectedValue
from .protocol import EvidenceRef, EvidenceValue, ExecutionContext, PlanNode, ReasoningPlan


def _evidence(value: object, *ids: str) -> EvidenceValue:
    return EvidenceValue(value=value, evidence=tuple(EvidenceRef(item, source="lcfa-bench") for item in ids))


def retrieval_suite() -> BenchmarkSuite:
    candidates = [
        {"id": "course:calc1", "text": "first year calculus assessment scores"},
        {"id": "course:bio1", "text": "first year biology laboratory attendance"},
        {"id": "course:hist1", "text": "history writing seminar assessment rubric"},
    ]
    return BenchmarkSuite(
        id="lcfa-core-retrieval",
        metadata={"family": "retrieval", "purpose": "deterministic lexical localization baseline"},
        cases=(
            BenchmarkCase(
                id="retrieval.rank-exact-domain-terms",
                plan=ReasoningPlan(
                    id="retrieval-rank",
                    nodes=(
                        PlanNode(
                            "rank",
                            "retrieval.rank",
                            {"query": "$state.query", "candidates": "$state.candidates"},
                            {"top_k": 2},
                        ),
                        PlanNode(
                            "recall",
                            "retrieval.recall_at_k",
                            {"ranked": "$node.rank", "relevant": ["course:calc1"]},
                            {"k": 1},
                            depends_on=("rank",),
                        ),
                    ),
                    outputs=("rank", "recall"),
                ),
                context=ExecutionContext(
                    state={
                        "query": _evidence("calculus first year assessment", "obs:query:1"),
                        "candidates": _evidence(candidates, "obs:candidates:1"),
                    }
                ),
                expectations=(
                    ExpectedValue("values.rank.0.id", "course:calc1"),
                    ExpectedValue("values.recall.recall", 1.0),
                    ExpectedValue("values.recall.hits.0", "course:calc1"),
                ),
                expected_evidence_ids=("obs:query:1", "obs:candidates:1"),
                tags=("retrieval", "localization", "evidence"),
            ),
            BenchmarkCase(
                id="retrieval.rank-multi-field",
                plan=ReasoningPlan(
                    id="retrieval-multifield",
                    nodes=(
                        PlanNode(
                            "rank",
                            "retrieval.rank",
                            {"query": "$state.query", "candidates": "$state.candidates"},
                            {"top_k": 2, "text_fields": ["name", "description"]},
                        ),
                    ),
                    outputs=("rank",),
                ),
                context=ExecutionContext(
                    state={
                        "query": "machine maintenance temperature",
                        "candidates": [
                            {"id": "machine:m1", "name": "M290", "description": "temperature maintenance telemetry"},
                            {"id": "part:p1", "name": "bracket", "description": "dimensional inspection result"},
                            {"id": "machine:m2", "name": "mill", "description": "spindle vibration telemetry"},
                        ],
                    }
                ),
                expectations=(
                    ExpectedValue("values.rank.0.id", "machine:m1"),
                    ExpectedValue("values.rank.0.matched_terms", ["maintenance", "temperature"]),
                ),
                tags=("retrieval", "localization"),
            ),
        ),
    )


def temporal_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        id="lcfa-core-temporal",
        metadata={"family": "temporal"},
        cases=(
            BenchmarkCase(
                id="temporal.linear-trend",
                plan=ReasoningPlan(
                    id="temporal-linear-trend",
                    nodes=(PlanNode("slope", "temporal.slope", {"values": "$state.values"}),),
                    outputs=("slope",),
                ),
                context=ExecutionContext(state={"values": _evidence([10, 12, 14, 16], "obs:series:trend")}),
                expectations=(ExpectedValue("values.slope", 2.0),),
                expected_evidence_ids=("obs:series:trend",),
                tags=("temporal", "trend", "evidence"),
            ),
            BenchmarkCase(
                id="temporal.change-point",
                plan=ReasoningPlan(
                    id="temporal-change-point",
                    nodes=(PlanNode("change", "temporal.change_point", {"values": "$state.values"}, {"min_segment": 2}),),
                    outputs=("change",),
                ),
                context=ExecutionContext(state={"values": [10, 11, 9, 30, 31, 29]}),
                expectations=(
                    ExpectedValue("values.change.index", 3),
                    ExpectedValue("values.change.before_mean", 10.0),
                    ExpectedValue("values.change.after_mean", 30.0),
                    ExpectedValue("values.change.delta", 20.0),
                ),
                tags=("temporal", "change-point"),
            ),
            BenchmarkCase(
                id="temporal.window-inclusive",
                plan=ReasoningPlan(
                    id="temporal-window",
                    nodes=(
                        PlanNode(
                            "window",
                            "temporal.window",
                            {"items": "$state.items"},
                            {"start": "2026-01-02T00:00:00Z", "end": "2026-01-03T00:00:00Z"},
                        ),
                    ),
                    outputs=("window",),
                ),
                context=ExecutionContext(
                    state={
                        "items": [
                            {"id": "a", "timestamp": "2026-01-01T00:00:00Z"},
                            {"id": "b", "timestamp": "2026-01-02T00:00:00Z"},
                            {"id": "c", "timestamp": "2026-01-03T00:00:00Z"},
                            {"id": "d", "timestamp": "2026-01-04T00:00:00Z"},
                        ]
                    }
                ),
                expectations=(
                    ExpectedValue("values.window.0.id", "b"),
                    ExpectedValue("values.window.1.id", "c"),
                ),
                tags=("temporal", "window"),
            ),
        ),
    )


def hierarchy_suite() -> BenchmarkSuite:
    relations = [
        {"source": "dept:math", "target": "college:science", "predicate": "belongs_to"},
        {"source": "dept:bio", "target": "college:science", "predicate": "belongs_to"},
        {"source": "course:calc1", "target": "dept:math", "predicate": "belongs_to"},
        {"source": "course:bio1", "target": "dept:bio", "predicate": "belongs_to"},
    ]
    return BenchmarkSuite(
        id="lcfa-core-hierarchy",
        metadata={"family": "hierarchy"},
        cases=(
            BenchmarkCase(
                id="hierarchy.expand-descendants",
                plan=ReasoningPlan(
                    id="hierarchy-expand",
                    nodes=(
                        PlanNode(
                            "expand",
                            "hierarchy.expand",
                            {"relations": "$state.relations", "roots": "college:science"},
                            {"direction": "children", "predicate": "belongs_to", "max_depth": 2},
                        ),
                    ),
                    outputs=("expand",),
                ),
                context=ExecutionContext(state={"relations": relations}),
                expectations=(
                    ExpectedValue("values.expand.nodes", ["college:science", "dept:math", "dept:bio", "course:calc1", "course:bio1"]),
                    ExpectedValue("values.expand.depth.course:calc1", 2),
                ),
                tags=("hierarchy", "multi-hop"),
            ),
            BenchmarkCase(
                id="hierarchy.path-and-aggregate",
                plan=ReasoningPlan(
                    id="hierarchy-path-aggregate",
                    nodes=(
                        PlanNode(
                            "path",
                            "hierarchy.path",
                            {"relations": "$state.relations", "source": "course:calc1", "target": "college:science"},
                            {"direction": "forward", "predicate": "belongs_to"},
                        ),
                        PlanNode(
                            "aggregate",
                            "hierarchy.aggregate",
                            {"nodes": "$node.path", "values": "$state.values"},
                            {"op": "mean"},
                            depends_on=("path",),
                        ),
                    ),
                    outputs=("path", "aggregate"),
                ),
                context=ExecutionContext(
                    state={
                        "relations": relations,
                        "values": {"course:calc1": 70, "dept:math": 80, "college:science": 90},
                    }
                ),
                expectations=(
                    ExpectedValue("values.path", ["course:calc1", "dept:math", "college:science"]),
                    ExpectedValue("values.aggregate", 80.0),
                ),
                tags=("hierarchy", "path", "aggregate"),
            ),
        ),
    )


def cohort_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        id="lcfa-core-cohort",
        metadata={"family": "cohort"},
        cases=(
            BenchmarkCase(
                id="cohort.mean-shift-effect-size",
                plan=ReasoningPlan(
                    id="cohort-compare",
                    nodes=(PlanNode("compare", "cohort.compare", {"current": "$state.current", "baseline": "$state.baseline"}),),
                    outputs=("compare",),
                ),
                context=ExecutionContext(state={"current": [70, 72, 74], "baseline": [80, 82, 84]}),
                expectations=(
                    ExpectedValue("values.compare.current_mean", 72.0),
                    ExpectedValue("values.compare.baseline_mean", 82.0),
                    ExpectedValue("values.compare.delta", -10.0),
                    ExpectedValue("values.compare.effect_size", -5.0),
                ),
                tags=("cohort", "comparison", "effect-size"),
            ),
            BenchmarkCase(
                id="cohort.percentile-and-summary",
                plan=ReasoningPlan(
                    id="cohort-percentile-summary",
                    nodes=(
                        PlanNode("percentile", "cohort.percentile", {"cohort": [10, 20, 30, 40], "value": 30}),
                        PlanNode("summary", "cohort.summarize", {"groups": "$state.groups"}),
                    ),
                    outputs=("percentile", "summary"),
                ),
                context=ExecutionContext(state={"groups": {"a": [1, 2, 3], "b": [10, 20, 30, 40]}}),
                expectations=(
                    ExpectedValue("values.percentile", 62.5),
                    ExpectedValue("values.summary.a.mean", 2.0),
                    ExpectedValue("values.summary.b.median", 25.0),
                ),
                tags=("cohort", "percentile", "summary"),
            ),
        ),
    )


def anomaly_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        id="lcfa-core-anomaly",
        metadata={"family": "anomaly"},
        cases=(
            BenchmarkCase(
                id="anomaly.robust-outlier",
                plan=ReasoningPlan(
                    id="anomaly-outlier",
                    nodes=(PlanNode("outliers", "anomaly.outliers", {"values": "$state.values"}, {"threshold": 3.5}),),
                    outputs=("outliers",),
                ),
                context=ExecutionContext(state={"values": [10, 10, 11, 10, 100]}),
                expectations=(
                    ExpectedValue("values.outliers.0.index", 4),
                    ExpectedValue("values.outliers.0.value", 100.0),
                ),
                tags=("anomaly", "robust", "outlier"),
            ),
            BenchmarkCase(
                id="anomaly.ewma",
                plan=ReasoningPlan(
                    id="anomaly-ewma",
                    nodes=(PlanNode("smooth", "anomaly.ewma", {"values": "$state.values"}, {"alpha": 0.5}),),
                    outputs=("smooth",),
                ),
                context=ExecutionContext(state={"values": [10, 20, 30]}),
                expectations=(ExpectedValue("values.smooth", [10.0, 15.0, 22.5]),),
                tags=("anomaly", "smoothing"),
            ),
        ),
    )


def evidence_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        id="lcfa-core-evidence",
        metadata={"family": "evidence"},
        cases=(
            BenchmarkCase(
                id="evidence.propagates-through-derived-delta",
                plan=ReasoningPlan(
                    id="evidence-propagation",
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
                ),
                context=ExecutionContext(
                    state={
                        "current": _evidence([70, 72, 74], "obs:current"),
                        "baseline": _evidence([80, 82, 84], "obs:baseline"),
                    }
                ),
                expectations=(ExpectedValue("values.delta", -10.0),),
                expected_evidence_ids=("obs:current", "obs:baseline"),
                tags=("evidence", "provenance", "numeric"),
            ),
            BenchmarkCase(
                id="evidence.coverage-and-contradiction",
                plan=ReasoningPlan(
                    id="evidence-quality",
                    nodes=(
                        PlanNode(
                            "coverage",
                            "evidence.coverage",
                            {"required": ["obs:a", "obs:b", "obs:c"], "available": ["obs:a", "obs:c"]},
                        ),
                        PlanNode(
                            "contradictions",
                            "evidence.contradictions",
                            {"claims": "$state.claims"},
                        ),
                    ),
                    outputs=("coverage", "contradictions"),
                ),
                context=ExecutionContext(
                    state={
                        "claims": [
                            {"key": "status", "value": "active", "source": "a"},
                            {"key": "status", "value": "inactive", "source": "b"},
                            {"key": "owner", "value": "team-1", "source": "a"},
                        ]
                    }
                ),
                expectations=(
                    ExpectedValue("values.coverage.coverage", 2 / 3, abs_tol=1e-12),
                    ExpectedValue("values.coverage.missing_ids", ["obs:b"]),
                    ExpectedValue("values.contradictions.0.key", "status"),
                    ExpectedValue("values.contradictions.0.variant_count", 2),
                ),
                tags=("evidence", "coverage", "contradiction"),
            ),
        ),
    )


def constraints_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        id="lcfa-core-constraints",
        metadata={"family": "constraints"},
        cases=(
            BenchmarkCase(
                id="constraints.filter-and-check",
                plan=ReasoningPlan(
                    id="constraints-filter-check",
                    nodes=(
                        PlanNode(
                            "eligible",
                            "constraints.filter",
                            {"items": "$state.items"},
                            {"field": "score", "op": "gte", "value": 80},
                        ),
                        PlanNode(
                            "check",
                            "constraints.check",
                            {"lhs": "$state.status", "rhs": "active"},
                            {"op": "eq"},
                        ),
                    ),
                    outputs=("eligible", "check"),
                ),
                context=ExecutionContext(
                    state={
                        "items": [
                            {"id": "a", "score": 79},
                            {"id": "b", "score": 80},
                            {"id": "c", "score": 95},
                        ],
                        "status": "active",
                    }
                ),
                expectations=(
                    ExpectedValue("values.eligible", [{"id": "b", "score": 80}, {"id": "c", "score": 95}]),
                    ExpectedValue("values.check.passed", True),
                ),
                tags=("constraints", "filter", "policy"),
            ),
            BenchmarkCase(
                id="constraints.compose-all",
                plan=ReasoningPlan(
                    id="constraints-compose",
                    nodes=(
                        PlanNode("left", "constraints.check", {"lhs": 10, "rhs": 5}, {"op": "gt"}),
                        PlanNode("right", "constraints.check", {"lhs": "ready", "rhs": "ready"}, {"op": "eq"}),
                        PlanNode(
                            "all",
                            "constraints.all",
                            {"conditions": [True, True, True]},
                            depends_on=("left", "right"),
                        ),
                    ),
                    outputs=("left", "right", "all"),
                ),
                context=ExecutionContext(),
                expectations=(
                    ExpectedValue("values.left.passed", True),
                    ExpectedValue("values.right.passed", True),
                    ExpectedValue("values.all", True),
                ),
                tags=("constraints", "composition"),
            ),
        ),
    )


def end_to_end_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        id="lcfa-core-end-to-end",
        metadata={"family": "end_to_end", "purpose": "multi-operator solution-state reasoning"},
        cases=(
            BenchmarkCase(
                id="end-to-end.trajectory-investigation",
                plan=ReasoningPlan(
                    id="trajectory-investigation",
                    nodes=(
                        PlanNode("current_mean", "stats.mean", {"values": "$state.current"}),
                        PlanNode("baseline_mean", "stats.mean", {"values": "$state.baseline"}),
                        PlanNode(
                            "delta",
                            "stats.delta",
                            {"current": "$node.current_mean", "baseline": "$node.baseline_mean"},
                            depends_on=("current_mean", "baseline_mean"),
                        ),
                        PlanNode("trend", "temporal.slope", {"values": "$state.current"}),
                        PlanNode(
                            "declined",
                            "constraints.check",
                            {"lhs": "$node.delta", "rhs": 0},
                            {"op": "lt"},
                            depends_on=("delta",),
                        ),
                        PlanNode(
                            "finding",
                            "core.finding",
                            {"value": "$node.delta"},
                            {"kind": "trajectory_change", "unit": "points"},
                            depends_on=("delta",),
                        ),
                        PlanNode(
                            "recommend",
                            "core.recommend",
                            {"value": "$node.delta"},
                            {"kind": "review", "parameters": {"reason": "negative_trajectory"}},
                            depends_on=("delta",),
                        ),
                    ),
                    outputs=("delta", "trend", "declined", "finding", "recommend"),
                ),
                context=ExecutionContext(
                    state={
                        "current": _evidence([72, 70, 68, 66], "obs:current:e2e"),
                        "baseline": _evidence([80, 80, 80, 80], "obs:baseline:e2e"),
                    }
                ),
                expectations=(
                    ExpectedValue("values.delta", -11.0),
                    ExpectedValue("values.trend", -2.0),
                    ExpectedValue("values.declined.passed", True),
                    ExpectedValue("values.finding.kind", "trajectory_change"),
                    ExpectedValue("values.recommend.kind", "review"),
                ),
                expected_evidence_ids=("obs:current:e2e", "obs:baseline:e2e"),
                tags=("end_to_end", "temporal", "constraints", "evidence", "agentic-boundary"),
            ),
            BenchmarkCase(
                id="end-to-end.retrieve-then-analyze",
                plan=ReasoningPlan(
                    id="retrieve-then-analyze",
                    nodes=(
                        PlanNode(
                            "rank",
                            "retrieval.rank",
                            {"query": "$state.query", "candidates": "$state.candidates"},
                            {"top_k": 1},
                        ),
                        PlanNode("change", "temporal.change_point", {"values": "$state.series"}, {"min_segment": 2}),
                        PlanNode(
                            "large_shift",
                            "constraints.check",
                            {"lhs": "$node.change.score", "rhs": 10},
                            {"op": "gte"},
                            depends_on=("change",),
                        ),
                    ),
                    outputs=("rank", "change", "large_shift"),
                ),
                context=ExecutionContext(
                    state={
                        "query": "calculus assessment",
                        "candidates": [
                            {"id": "course:calc1", "text": "calculus assessment"},
                            {"id": "course:bio1", "text": "biology laboratory"},
                        ],
                        "series": [10, 11, 9, 30, 31, 29],
                    }
                ),
                expectations=(
                    ExpectedValue("values.rank.0.id", "course:calc1"),
                    ExpectedValue("values.change.index", 3),
                    ExpectedValue("values.large_shift.passed", True),
                ),
                tags=("end_to_end", "retrieval", "temporal", "constraints"),
            ),
        ),
    )


def all_suites() -> tuple[BenchmarkSuite, ...]:
    return (
        retrieval_suite(),
        temporal_suite(),
        hierarchy_suite(),
        cohort_suite(),
        anomaly_suite(),
        evidence_suite(),
        constraints_suite(),
        end_to_end_suite(),
    )


def suite_by_id(suite_id: str) -> BenchmarkSuite:
    for suite in all_suites():
        if suite.id == suite_id:
            return suite
    raise KeyError(f"unknown built-in benchmark suite: {suite_id}")


__all__ = [
    "all_suites",
    "anomaly_suite",
    "cohort_suite",
    "constraints_suite",
    "end_to_end_suite",
    "evidence_suite",
    "hierarchy_suite",
    "retrieval_suite",
    "suite_by_id",
    "temporal_suite",
]
