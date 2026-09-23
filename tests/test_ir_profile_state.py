from __future__ import annotations

from pathlib import Path

import pytest

from lcfa import (
    ActionGraph,
    ActionNode,
    LCFA,
    MemoryStateStore,
    PlanNode,
    ReasoningPlan,
    StateConflictError,
    content_hash,
    dumps_ir,
    load_profile,
    loads_ir,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reasoning_plan_round_trips_through_versioned_json_ir() -> None:
    plan = ReasoningPlan(
        id="round-trip",
        nodes=(
            PlanNode("left", "core.identity", {"value": 4}),
            PlanNode(
                "right",
                "core.identity",
                {"value": "$node.left"},
                depends_on=("left",),
            ),
        ),
        outputs=("right",),
        metadata={"purpose": "test"},
    )

    text = dumps_ir(plan)
    restored = loads_ir(text)

    assert '"format": "lcfa.ir.v1"' in text
    assert restored == plan
    assert restored.schema_version == "lcfa.plan.v1"


def test_action_graph_round_trips_without_losing_governance_fields() -> None:
    graph = ActionGraph(
        id="actions:1",
        source_solution_id="solution:1",
        nodes=(
            ActionNode(
                id="notify",
                action="example.notify",
                inputs={"subject": "review"},
                effects=("external_message",),
                required_capabilities=("notify.send",),
                requires_approval=True,
                approval_key="human-approved",
            ),
        ),
    )

    assert loads_ir(dumps_ir(graph)) == graph


def test_reference_education_profile_loads_and_binds_to_runtime() -> None:
    profile = load_profile(ROOT / "profiles" / "education")
    engine = LCFA.from_profile(profile)

    assert profile.id == "education"
    assert "student" in profile.entity_types
    assert "enrolled_in" in profile.relation_types
    assert engine.profile == profile
    assert set(profile.enabled_operators) <= set(engine.operators.names())


def test_semantic_identity_stays_stable_across_immutable_versions() -> None:
    store = MemoryStateStore()
    first = store.put("student:123", {"gpa": 3.2, "credits": 48})
    second = store.put(
        "student:123",
        {"gpa": 3.4, "credits": 60},
        expected_version=1,
    )

    assert first.identity.semantic_id == second.identity.semantic_id == "student:123"
    assert first.identity.version == 1
    assert second.identity.version == 2
    assert first.identity.content_hash != second.identity.content_hash
    assert store.get("student:123", 1) == first
    assert store.get("student:123") == second
    assert store.history("student:123") == (first, second)


def test_state_hash_is_canonical_and_optimistic_versioning_is_enforced() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})

    store = MemoryStateStore()
    store.put("course:calc1", {"status": "active"})
    with pytest.raises(StateConflictError, match="expected version"):
        store.put(
            "course:calc1",
            {"status": "archived"},
            expected_version=0,
        )
