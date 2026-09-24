from __future__ import annotations

from pathlib import Path

from safetensors import safe_open

from lcfa import EvidenceValue, ExecutionContext, LCFA, NumpyFastPrior, PlanNode, ReasoningPlan, load_artifact_reasoner

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "lcfa-stochastic-flow-reference"


def test_numpy_fast_prior_updates_and_snapshots(tmp_path: Path) -> None:
    prior = NumpyFastPrior(7, learning_rate=0.2, decay=1.0)
    features = (0.0, 0.8, 1.0, 0.0, 1.0, 1.0, 0.9)
    before = prior.score(features)
    update = prior.observe(features, 0.95)
    after = prior.score(features)
    assert after > before
    assert update["surprise"] > 0
    path = prior.snapshot(tmp_path / "prior.safetensors")
    with safe_open(str(path), framework="np", device="cpu") as handle:
        assert "prior.weights" in handle.keys()
        assert handle.get_tensor("prior.weights").shape == (7,)


def test_runtime_can_override_fast_prior_without_changing_artifact() -> None:
    reasoner = load_artifact_reasoner(
        REFERENCE,
        base_engine=LCFA(),
        runtime_options={"adaptation_type": "numpy-fast", "adaptation_learning_rate": 0.2},
    )
    assert reasoner.metadata["adaptation"]["type"] == "numpy-fast"
    plan = ReasoningPlan(
        id="adaptive-mean",
        nodes=(PlanNode("mean", "stats.mean", {"values": "$state.values"}),),
        outputs=("mean",),
    )
    solution = reasoner.reason(plan, ExecutionContext(state={"values": EvidenceValue([1.0, 3.0])}))
    updates = solution.metadata["stochastic_flow"]["adaptation"]["updates"]
    assert updates
    assert updates[0]["surprise"] >= 0.0


def test_fast_prior_resets_between_reason_calls() -> None:
    reasoner = load_artifact_reasoner(
        REFERENCE,
        runtime_options={"adaptation_type": "numpy-fast", "adaptation_learning_rate": 0.2},
    )
    plan = ReasoningPlan(
        id="adaptive-reset",
        nodes=(PlanNode("mean", "stats.mean", {"values": "$state.values"}),),
        outputs=("mean",),
    )
    context = ExecutionContext(state={"values": EvidenceValue([1.0, 3.0])})
    first = reasoner.reason(plan, context).metadata["stochastic_flow"]["adaptation"]["updates"]
    second = reasoner.reason(plan, context).metadata["stochastic_flow"]["adaptation"]["updates"]
    assert first[0]["prediction"] == second[0]["prediction"]
