from __future__ import annotations

import pytest

from lcfa import (
    ActionNode,
    ActionPolicyError,
    ActionResult,
    ActionSpec,
    EvidenceRef,
    EvidenceValue,
    ExecutionContext,
    LCFA,
    PlanNode,
    ReasoningPlan,
)


def _plan() -> ReasoningPlan:
    return ReasoningPlan(
        id="course-change",
        nodes=(
            PlanNode("current_mean", "stats.mean", {"values": "$state.current"}),
            PlanNode("baseline_mean", "stats.mean", {"values": "$state.baseline"}),
            PlanNode(
                "delta",
                "stats.delta",
                {"current": "$node.current_mean", "baseline": "$node.baseline_mean"},
                depends_on=("current_mean", "baseline_mean"),
            ),
            PlanNode(
                "recommend",
                "core.recommend",
                {"value": "$node.delta"},
                {"kind": "review", "parameters": {"reason": "trajectory_change"}},
                depends_on=("delta",),
            ),
        ),
        outputs=("delta", "recommend"),
    )


def _context() -> ExecutionContext:
    return ExecutionContext(
        state={
            "current": EvidenceValue(
                [70, 72, 71],
                (EvidenceRef("obs:current", source="demo"),),
            ),
            "baseline": EvidenceValue(
                [79, 80, 78],
                (EvidenceRef("obs:baseline", source="demo"),),
            ),
        }
    )


def test_reasoning_propagates_evidence_and_emits_recommendation() -> None:
    engine = LCFA()
    solution = engine.reason(_plan(), _context())

    assert solution.values["delta"] == pytest.approx(-8.0)
    assert [item.id for item in solution.evidence] == ["obs:current", "obs:baseline"]
    assert len(solution.recommendations) == 1
    assert solution.recommendations[0].kind == "review"
    assert [step.status for step in solution.trace.steps] == ["ok", "ok", "ok", "ok"]


def test_solution_state_can_compile_and_execute_governed_action_graph() -> None:
    engine = LCFA()
    solution = engine.reason(_plan(), _context())

    def build_review(recommendation, _solution, index):
        return (
            ActionNode(
                id=f"review-{index}",
                action="demo.create_review",
                inputs={
                    "delta": "$solution.values.delta",
                    "reason": recommendation.parameters["reason"],
                },
                effects=("external_write",),
                required_capabilities=("review.create",),
                requires_approval=True,
                approval_key="advisor-approved",
            ),
        )

    engine.action_compiler.register("review", build_review)
    engine.actions.register(
        ActionSpec(
            name="demo.create_review",
            effects=("external_write",),
            handler=lambda _ctx, inputs: ActionResult(
                value={"created": True, **dict(inputs)},
                observations={"review_created": True},
            ),
        )
    )

    graph = engine.compile_actions(solution)
    run = engine.act(
        graph,
        solution,
        ExecutionContext(
            capabilities=frozenset({"review.create"}),
            approvals=frozenset({"advisor-approved"}),
        ),
    )

    assert run.results["review-0"].value["created"] is True
    assert run.results["review-0"].value["delta"] == pytest.approx(-8.0)
    assert run.observations["review-0.review_created"] is True


def test_action_execution_denies_missing_capability() -> None:
    engine = LCFA()
    solution = engine.reason(_plan(), _context())

    engine.action_compiler.register(
        "review",
        lambda _rec, _solution, _index: (
            ActionNode(
                id="review",
                action="demo.create_review",
                required_capabilities=("review.create",),
            ),
        ),
    )
    engine.actions.register(
        ActionSpec(
            name="demo.create_review",
            handler=lambda _ctx, _inputs: ActionResult(value=True),
        )
    )

    graph = engine.compile_actions(solution)
    with pytest.raises(ActionPolicyError, match="missing capabilities"):
        engine.act(graph, solution, ExecutionContext())
