from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from lcfa import EvidenceValue, ExecutionContext, LCFA, NullFastPrior, PlanNode, ReasoningPlan, load_artifact_manifest
from lcfa.adaptive_flow import AdaptiveStochasticFlowReasoner
from lcfa.backbones import BackboneSample

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "lcfa-stochastic-flow-reference"


def _manifest():
    manifest = load_artifact_manifest(REFERENCE)
    flow = dict(manifest.config.get("flow", {}))
    flow.update({
        "branches": 2,
        "beam_width": 2,
        "min_steps": 1,
        "max_steps": 3,
        "adaptive_compute": {
            "enabled": True,
            "initial_branches": 1,
            "confidence_threshold": 0.8,
            "evidence_threshold": 0.5,
            "score_margin_threshold": 0.15,
            "verifier_accept_threshold": 0.85,
            "expand_when_uncertain": True,
            "verify_when_uncertain": True,
            "require_parsed": True,
        },
    })
    return replace(
        manifest,
        config={
            **dict(manifest.config),
            "flow": flow,
            "verifier": {"enabled": True, "top_k": 1, "max_new_tokens": 16},
        },
    )


def _plan() -> ReasoningPlan:
    return ReasoningPlan(
        id="adaptive-smoke",
        nodes=(PlanNode("mean", "stats.mean", {"values": "$state.values"}),),
        outputs=("mean",),
    )


def _context() -> ExecutionContext:
    return ExecutionContext(state={"values": EvidenceValue([1.0, 3.0])})


class EasyBackbone:
    metadata = {"type": "easy-test"}

    def __init__(self) -> None:
        self.proposal_calls = 0
        self.verifier_calls = 0
        self.branch_requests: list[int] = []

    def sample(self, *, system_prompt: str, user_prompt: str, branches: int,
               temperature: float, top_p: float, max_new_tokens: int,
               seed: int | None = None) -> tuple[BackboneSample, ...]:
        del user_prompt, temperature, top_p, max_new_tokens, seed
        if "LCFA_VERIFIER" in system_prompt:
            self.verifier_calls += 1
            return (BackboneSample('{"score": 0.99}'),)
        self.proposal_calls += 1
        self.branch_requests.append(branches)
        payload = json.dumps({
            "answer": "The mean is 2.0.",
            "rationale": "The LCFA anchor contains the exact mean.",
            "confidence": 0.99,
            "evidence_ids": [],
            "tool_requests": [],
            "final": True,
        })
        return tuple(BackboneSample(payload) for _ in range(branches))


class UncertainBackbone(EasyBackbone):
    metadata = {"type": "uncertain-test"}

    def sample(self, *, system_prompt: str, user_prompt: str, branches: int,
               temperature: float, top_p: float, max_new_tokens: int,
               seed: int | None = None) -> tuple[BackboneSample, ...]:
        if "LCFA_VERIFIER" in system_prompt:
            self.verifier_calls += 1
            return (BackboneSample('{"score": 0.95}'),)
        del user_prompt, temperature, top_p, max_new_tokens, seed
        self.proposal_calls += 1
        self.branch_requests.append(branches)
        payload = json.dumps({
            "answer": "The mean is probably 2.0.",
            "rationale": "Candidate needs validation.",
            "confidence": 0.20,
            "evidence_ids": [],
            "tool_requests": [],
            "final": True,
        })
        return tuple(BackboneSample(payload) for _ in range(branches))


def _reasoner(backbone):
    return AdaptiveStochasticFlowReasoner(
        _manifest(),
        weights_path=REFERENCE / "model.safetensors",
        backbone=backbone,
        fast_prior=NullFastPrior(),
        base_engine=LCFA(),
    )


def test_easy_case_uses_minimum_compute_and_skips_verifier() -> None:
    backbone = EasyBackbone()
    solution = _reasoner(backbone).reason(_plan(), _context())
    adaptive = solution.metadata["stochastic_flow"]["adaptive_compute"]

    assert solution.values["mean"] == 2.0
    assert backbone.proposal_calls == 1
    assert backbone.branch_requests == [1]
    assert backbone.verifier_calls == 0
    assert adaptive["steps_used"] == 1
    assert adaptive["generated_candidates"] == 1
    assert adaptive["verifier_calls"] == 0
    assert adaptive["expansions"] == 0
    assert adaptive["trace"][0]["stop"] is True


def test_uncertain_case_expands_then_verifies_before_stopping() -> None:
    backbone = UncertainBackbone()
    solution = _reasoner(backbone).reason(_plan(), _context())
    adaptive = solution.metadata["stochastic_flow"]["adaptive_compute"]

    assert solution.values["mean"] == 2.0
    assert backbone.proposal_calls == 2
    assert backbone.branch_requests == [1, 1]
    assert backbone.verifier_calls == 1
    assert adaptive["steps_used"] == 1
    assert adaptive["generated_candidates"] == 2
    assert adaptive["verifier_calls"] == 1
    assert adaptive["expansions"] == 1
    assert adaptive["trace"][0]["stop"] is True
