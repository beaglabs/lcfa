from __future__ import annotations

import math

import pytest

from lcfa import ConstraintViolation, EvidenceRef, EvidenceValue, ExecutionContext, LCFA, PlanNode, ReasoningPlan


def _run(operator: str, inputs: dict[str, object], params: dict[str, object] | None = None, state: dict[str, object] | None = None):
    engine = LCFA()
    plan = ReasoningPlan(
        id=f"test:{operator}",
        nodes=(PlanNode("result", operator, inputs, params or {}),),
        outputs=("result",),
    )
    return engine.reason(plan, ExecutionContext(state=state or {}))


def test_temporal_operators_and_evidence_propagation() -> None:
    solution = _run(
        "temporal.slope",
        {"values": "$state.values"},
        state={"values": EvidenceValue([1.0, 3.0, 5.0], (EvidenceRef("obs:series"),))},
    )
    assert solution.values["result"] == pytest.approx(2.0)
    assert [item.id for item in solution.evidence] == ["obs:series"]

    change = _run("temporal.change_point", {"values": [10, 10, 10, 20, 20, 20]}).values["result"]
    assert change["index"] == 3
    assert change["delta"] == pytest.approx(10.0)

    window = _run(
        "temporal.window",
        {
            "items": [
                {"at": "2026-01-01T00:00:00Z", "value": 1},
                {"at": "2026-02-01T00:00:00Z", "value": 2},
                {"at": "2026-03-01T00:00:00Z", "value": 3},
            ],
            "start": "2026-02-01T00:00:00Z",
            "end": "2026-02-28T23:59:59Z",
        },
        {"timestamp_field": "at"},
    ).values["result"]
    assert [item["value"] for item in window] == [2]


def test_hierarchy_expand_path_and_aggregate() -> None:
    relations = [
        {"source": "section:a", "target": "course:1", "predicate": "belongs_to"},
        {"source": "section:b", "target": "course:1", "predicate": "belongs_to"},
        {"source": "course:1", "target": "department:x", "predicate": "belongs_to"},
    ]
    expanded = _run(
        "hierarchy.expand",
        {"relations": relations, "roots": "department:x"},
        {"direction": "children", "predicate": "belongs_to", "max_depth": 2},
    ).values["result"]
    assert expanded["nodes"] == ["department:x", "course:1", "section:a", "section:b"]
    assert expanded["depth"]["section:a"] == 2

    path = _run(
        "hierarchy.path",
        {"relations": relations, "source": "section:a", "target": "department:x"},
        {"direction": "forward", "predicate": "belongs_to"},
    ).values["result"]
    assert path == ["section:a", "course:1", "department:x"]

    mean = _run(
        "hierarchy.aggregate",
        {"nodes": ["section:a", "section:b"], "values": {"section:a": 70, "section:b": 80}},
        {"op": "mean"},
    ).values["result"]
    assert mean == pytest.approx(75.0)


def test_cohort_compare_percentile_and_summary() -> None:
    comparison = _run(
        "cohort.compare",
        {"current": [70, 72, 74], "baseline": [80, 82, 84]},
    ).values["result"]
    assert comparison["current_mean"] == pytest.approx(72.0)
    assert comparison["baseline_mean"] == pytest.approx(82.0)
    assert comparison["delta"] == pytest.approx(-10.0)
    assert comparison["effect_size"] is not None

    percentile = _run("cohort.percentile", {"value": 30, "cohort": [10, 20, 30, 40]}).values["result"]
    assert percentile == pytest.approx(62.5)

    summary = _run("cohort.summarize", {"groups": {"a": [1, 2, 3], "b": [4, 5, 6]}}).values["result"]
    assert summary["a"]["mean"] == pytest.approx(2.0)
    assert summary["b"]["median"] == pytest.approx(5.0)


def test_anomaly_operators_detect_outlier_and_smooth() -> None:
    outliers = _run(
        "anomaly.outliers",
        {"values": [10, 10, 11, 10, 100]},
        {"threshold": 3.5},
    ).values["result"]
    assert [item["index"] for item in outliers] == [4]
    assert math.isinf(outliers[0]["score"]) or outliers[0]["score"] > 3.5

    ewma = _run("anomaly.ewma", {"values": [10, 20, 20]}, {"alpha": 0.5}).values["result"]
    assert ewma == pytest.approx([10.0, 15.0, 17.5])


def test_evidence_coverage_and_contradictions() -> None:
    coverage = _run(
        "evidence.coverage",
        {"required": ["a", "b", "c"], "available": [EvidenceRef("a"), {"id": "c"}]},
    ).values["result"]
    assert coverage["coverage"] == pytest.approx(2 / 3)
    assert coverage["missing_ids"] == ["b"]

    contradictions = _run(
        "evidence.contradictions",
        {
            "claims": [
                {"key": "status", "value": "open", "source": "a"},
                {"key": "status", "value": "closed", "source": "b"},
                {"key": "owner", "value": "x", "source": "a"},
                {"key": "owner", "value": "x", "source": "b"},
            ]
        },
    ).values["result"]
    assert len(contradictions) == 1
    assert contradictions[0]["key"] == "status"
    assert contradictions[0]["variant_count"] == 2


def test_constraints_check_filter_and_require() -> None:
    checked = _run("constraints.check", {"lhs": 5, "rhs": 3}, {"op": "gt"}).values["result"]
    assert checked["passed"] is True

    filtered = _run(
        "constraints.filter",
        {"items": [{"score": 1}, {"score": 4}, {"score": 7}]},
        {"field": "score", "op": "gte", "value": 4},
    ).values["result"]
    assert [item["score"] for item in filtered] == [4, 7]

    engine = LCFA()
    plan = ReasoningPlan(
        id="constraint-failure",
        nodes=(PlanNode("guard", "constraints.require", {"lhs": 1, "rhs": 2}, {"op": "gte", "message": "too small"}),),
        outputs=("guard",),
    )
    with pytest.raises(ConstraintViolation, match="too small"):
        engine.reason(plan, ExecutionContext())


def test_default_runtime_registers_all_operator_families() -> None:
    names = set(LCFA().operators.names())
    prefixes = {name.split(".", 1)[0] for name in names}
    assert {"temporal", "hierarchy", "cohort", "anomaly", "evidence", "constraints"} <= prefixes
