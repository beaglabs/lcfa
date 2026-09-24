from __future__ import annotations

import json
from pathlib import Path

from lcfa import ARTIFACT_FORMAT, BackboneSample, BenchmarkRunner, BenchmarkSuite, EvidenceRef, EvidenceValue, ExecutionContext, LCFA, NullFastPrior, PlanNode, ReasonerSubject, ReasoningPlan, STOCHASTIC_FLOW_ARCHITECTURE, StochasticFlowReasoner, all_suites, load_artifact_manifest, load_artifact_reasoner

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "lcfa-stochastic-flow-reference"


class ScriptedBackbone:
    metadata = {"type": "scripted-test"}

    def __init__(self) -> None:
        self.round = 0

    def sample(self, *, system_prompt: str, user_prompt: str, branches: int,
               temperature: float, top_p: float, max_new_tokens: int,
               seed: int | None = None) -> tuple[BackboneSample, ...]:
        del user_prompt, temperature, top_p, max_new_tokens, seed
        if "LCFA_VERIFIER" in system_prompt:
            return (BackboneSample('{"score": 0.95}', -0.01),)
        self.round += 1
        if self.round == 1:
            values = [
                {"answer": "I need an exact mean before finalizing.",
                 "rationale": "Use the trusted stats operator instead of estimating.",
                 "confidence": 0.8, "evidence_ids": ["obs:values"],
                 "tool_requests": [{"id": "mean-1", "operator": "stats.mean", "inputs": {"values": [1.0, 3.0]}, "params": {}}],
                 "final": False},
                {"answer": "Probably about two.", "rationale": "Estimate only.",
                 "confidence": 0.2, "evidence_ids": [], "tool_requests": [], "final": True},
            ]
        else:
            values = [{"answer": "The exact mean is 2.0.",
                       "rationale": "The trusted stats.mean observation returned 2.0.",
                       "confidence": 0.99, "evidence_ids": ["obs:values"],
                       "tool_requests": [], "final": True}]
        return tuple(BackboneSample(json.dumps(values[i % len(values)]), -0.05 - i * 0.01) for i in range(branches))


def test_reference_stochastic_artifact_loads() -> None:
    manifest = load_artifact_manifest(REFERENCE)
    assert manifest.format == ARTIFACT_FORMAT
    assert manifest.architecture == STOCHASTIC_FLOW_ARCHITECTURE
    reasoner = load_artifact_reasoner(REFERENCE)
    assert reasoner.metadata["backend"] == "stochastic-flow"
    assert reasoner.metadata["backbone"]["type"] == "reference"


def test_stochastic_flow_can_call_pure_operator_and_return_language() -> None:
    manifest = load_artifact_manifest(REFERENCE)
    reasoner = StochasticFlowReasoner(manifest, weights_path=REFERENCE / "model.safetensors",
                                      backbone=ScriptedBackbone(), fast_prior=NullFastPrior(), base_engine=LCFA())
    plan = ReasoningPlan(id="stochastic-mean", metadata={"query": "What is the mean?"},
        nodes=(PlanNode("mean", "stats.mean", {"values": "$state.values"}),), outputs=("mean",))
    context = ExecutionContext(state={"values": EvidenceValue([1.0, 3.0], (EvidenceRef("obs:values"),))})
    solution = reasoner.reason(plan, context)
    assert solution.values["mean"] == 2.0
    assert solution.metadata["language"] == "The exact mean is 2.0."
    flow = solution.metadata["stochastic_flow"]
    assert flow["status"] == "ok"
    assert flow["steps"] == 2
    assert "stats.mean" in [op for entry in flow["trace"] for op in entry["operators"]]


def test_reference_stochastic_artifact_preserves_builtin_all_regression() -> None:
    combined = BenchmarkSuite(id="lcfa-core-all", cases=tuple(case for suite in all_suites() for case in suite.cases))
    reasoner = load_artifact_reasoner(REFERENCE, base_engine=LCFA())
    report = BenchmarkRunner(repeats=2).run(ReasonerSubject(name=reasoner.artifact.id,
        reasoner=reasoner.reason, metadata=reasoner.metadata), combined)
    assert report.summary.case_count == 17
    assert report.summary.case_pass_rate == 1.0
    assert report.summary.assertion_accuracy == 1.0
    assert report.summary.error_rate == 0.0
    assert report.summary.evidence_f1 == 1.0
