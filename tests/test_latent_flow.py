from __future__ import annotations

from pathlib import Path

from lcfa import (
    EvidenceRef,
    EvidenceValue,
    ExecutionContext,
    LCFA,
    LATENT_FLOW_ARCHITECTURE,
    PlanNode,
    ReasoningPlan,
    artifact_architectures,
    load_artifact_reasoner,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "lcfa-latent-flow-reference"


def _plan() -> ReasoningPlan:
    return ReasoningPlan(
        id="latent-mean",
        metadata={"query": "What is the mean?"},
        nodes=(PlanNode("mean", "stats.mean", {"values": "$state.values"}),),
        outputs=("mean",),
    )


def test_latent_flow_architecture_self_registers() -> None:
    assert LATENT_FLOW_ARCHITECTURE in artifact_architectures()


def test_reference_latent_flow_preserves_grounded_solution_and_adds_latent_state() -> None:
    reasoner = load_artifact_reasoner(REFERENCE, base_engine=LCFA())
    context = ExecutionContext(
        state={"values": EvidenceValue([1.0, 3.0], (EvidenceRef("obs:values"),))},
        metadata={"query": "What is the mean?"},
    )
    solution = reasoner.reason(_plan(), context)
    assert solution.values["mean"] == 2.0
    assert [e.id for e in solution.evidence] == ["obs:values"]
    flow = solution.metadata["latent_flow"]
    assert flow["architecture"] == LATENT_FLOW_ARCHITECTURE
    assert flow["status"] == "representation-ready"
    assert flow["feature_dim"] == 64
    assert flow["latent_dim"] == 16
    assert flow["solution_decoder_active"] is False
    assert flow["state_id"].startswith("latent:")


def test_latent_flow_is_repeatable_for_reference_zplug() -> None:
    reasoner = load_artifact_reasoner(REFERENCE)
    context = ExecutionContext(state={"values": EvidenceValue([2.0, 4.0])})
    first = reasoner.reason(_plan(), context)
    second = reasoner.reason(_plan(), context)
    assert first.metadata["latent_flow"]["state_id"] == second.metadata["latent_flow"]["state_id"]
