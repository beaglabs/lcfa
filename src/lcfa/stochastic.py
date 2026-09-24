"""Probabilistic LCFA reasoning with configurable backbones and fast priors."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from fnmatch import fnmatch
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open

from .adaptation import FEATURE_NAMES, FastPrior, create_fast_prior
from .artifact import STOCHASTIC_FLOW_ARCHITECTURE, ArtifactError, ArtifactManifest
from .backbones import BackboneSample, ReferenceBackbone, StochasticBackbone, TransformersCausalBackbone, MLXCausalBackbone, LlamaCppBackbone, create_backbone
from .engine import LCFA
from .protocol import EvidenceRef, EvidenceValue, ExecutionContext, OperatorResult, ReasoningPlan, SolutionState


@dataclass(frozen=True, slots=True)
class FlowToolRequest:
    operator: str
    inputs: Mapping[str, Any]
    params: Mapping[str, Any]
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class FlowCandidate:
    id: str
    step: int
    parent_id: str | None
    answer: str
    rationale: str
    confidence: float
    evidence_ids: tuple[str, ...]
    tool_requests: tuple[FlowToolRequest, ...]
    tool_observations: tuple[Mapping[str, Any], ...]
    final: bool
    parsed: bool
    logprob: float
    heuristic_score: float
    verifier_score: float | None
    prior_score: float
    score: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, EvidenceValue):
        return {"value": _json_safe(value.value), "evidence": [_json_safe(i) for i in value.evidence]}
    if isinstance(value, EvidenceRef):
        return {"id": value.id, "source": value.source, "metadata": _json_safe(value.metadata)}
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _json_text(value: Any, *, max_chars: int) -> str:
    text = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)
    return text if len(text) <= max_chars else text[: max(0, max_chars - 32)] + "...<truncated>"


def _extract_json_object(text: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scalar(weights_path: Path, name: str, default: float) -> float:
    with safe_open(str(weights_path), framework="np", device="cpu") as handle:
        if name not in handle.keys():
            return default
        tensor = handle.get_tensor(name)
        if tensor.size != 1:
            raise ArtifactError(f"stochastic-flow score tensor {name!r} must be scalar")
        return float(tensor.reshape(-1)[0])


class StochasticFlowReasoner:
    """Probabilistic branch/verify/tool reasoning over a grounded LCFA anchor."""

    SYSTEM_PROMPT = '''You are the reasoning layer inside LCFA.
Return ONLY one JSON object with keys: answer, rationale, confidence, evidence_ids, tool_requests, final.
`rationale` must be a brief evidence-grounded rationale, not hidden chain-of-thought.
Treat LCFA anchor values and tool observations as authoritative computations.
Do not invent evidence identifiers. Request only listed pure operators. Set final=true when sufficiently supported.'''

    VERIFIER_SYSTEM_PROMPT = '''LCFA_VERIFIER
Judge the candidate against the grounded anchor, evidence, and tool observations.
Return ONLY {"score": number} where score is between 0 and 1.'''

    def __init__(self, artifact: ArtifactManifest, *, weights_path: Path,
                 backbone: StochasticBackbone, fast_prior: FastPrior,
                 base_engine: LCFA | None = None,
                 adaptation_config: Mapping[str, Any] | None = None) -> None:
        if artifact.architecture != STOCHASTIC_FLOW_ARCHITECTURE:
            raise ArtifactError("wrong stochastic-flow architecture")
        if artifact.base_backend != "lcfa-zero":
            raise ArtifactError("unsupported stochastic-flow base backend")
        self.artifact = artifact
        self.weights_path = weights_path
        self.base_engine = base_engine or LCFA()
        self.backbone = backbone
        self.fast_prior = fast_prior
        config = dict(artifact.config)
        flow = dict(config.get("flow", {}))
        verifier = dict(config.get("verifier", {}))
        tools = dict(config.get("tools", {}))
        adaptation = dict(adaptation_config or config.get("adaptation", {}))
        self.branches = max(1, int(flow.get("branches", 4)))
        self.beam_width = max(1, int(flow.get("beam_width", 2)))
        self.min_steps = max(1, int(flow.get("min_steps", 1)))
        self.max_steps = max(self.min_steps, int(flow.get("max_steps", 4)))
        self.temperature = float(flow.get("temperature", 0.7))
        self.top_p = float(flow.get("top_p", 0.95))
        self.max_new_tokens = max(32, int(flow.get("max_new_tokens", 512)))
        self.max_prompt_chars = max(2048, int(flow.get("max_prompt_chars", 24000)))
        self.max_tool_requests = max(0, int(flow.get("max_tool_requests", 4)))
        self.fallback_to_anchor = bool(flow.get("fallback_to_anchor", True))
        self.seed = int(flow["seed"]) if flow.get("seed") is not None else None
        self.verifier_enabled = bool(verifier.get("enabled", True))
        self.verify_top_k = max(0, int(verifier.get("top_k", 2)))
        self.verifier_max_new_tokens = max(8, int(verifier.get("max_new_tokens", 32)))
        self.allowed_operator_patterns = tuple(str(i) for i in tools.get("allowed", []))
        self.prior_weight = float(adaptation.get("score_weight", 0.5))
        self.adaptation_enabled = str(adaptation.get("type", "none")) != "none"
        self.adaptation_snapshot_path = adaptation.get("snapshot_path")
        self.score_weights = {
            name: _scalar(weights_path, f"score.{name}", default)
            for name, default in {
                "logprob": .10,
                "confidence": 1.0,
                "evidence": .75,
                "tool": .25,
                "parse": .25,
                "final": .15,
                "verifier": 1.0,
            }.items()
        }
        self.metadata = {
            "backend": "stochastic-flow",
            "artifact_id": artifact.id,
            "artifact_version": artifact.version,
            "architecture": artifact.architecture,
            "weights_sha256": artifact.weights.sha256,
            "backbone": dict(backbone.metadata),
            "adaptation": dict(fast_prior.metadata),
            "branches": self.branches,
            "beam_width": self.beam_width,
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_manifest(cls, root: Path, manifest: ArtifactManifest, *, weights_path: Path,
                      base_engine: LCFA | None = None,
                      runtime_options: Mapping[str, Any] | None = None) -> "StochasticFlowReasoner":
        runtime = dict(runtime_options or {})
        backbone_cfg = dict(manifest.config.get("backbone", {}))
        kind = str(backbone_cfg.get("type", "transformers-local"))
        raw_path = runtime.get("backbone_path") or backbone_cfg.get("path")
        model_path: str | Path | None = raw_path
        if raw_path is not None and kind != "reference":
            candidate = Path(str(raw_path))
            model_path = candidate if candidate.is_absolute() else (root / candidate).resolve()
        backbone = create_backbone(
            kind,
            model_path,
            config=backbone_cfg,
            runtime=runtime,
            reference_allowed=bool(manifest.metadata.get("reference_only", False)),
        )
        adaptation_cfg = dict(manifest.config.get("adaptation", {}))
        adaptation_type = str(runtime.get("adaptation_type") or adaptation_cfg.get("type", "none"))
        fast_prior = create_fast_prior(
            adaptation_type,
            feature_count=len(FEATURE_NAMES),
            config=adaptation_cfg,
            runtime=runtime,
        )
        return cls(
            manifest,
            weights_path=weights_path,
            backbone=backbone,
            fast_prior=fast_prior,
            base_engine=base_engine,
            adaptation_config={**adaptation_cfg, "type": adaptation_type,
                               **({"snapshot_path": runtime["prior_snapshot"]} if runtime.get("prior_snapshot") else {})},
        )

    def _operator_allowed(self, name: str) -> bool:
        return any(fnmatch(name, p) for p in self.allowed_operator_patterns)

    def _catalog(self) -> list[dict[str, str]]:
        return [
            {"name": n, "description": self.base_engine.operators.get(n).description}
            for n in self.base_engine.operators.names()
            if self._operator_allowed(n)
        ]

    def _anchor_payload(self, anchor: SolutionState) -> dict[str, Any]:
        return {
            "plan_id": anchor.plan_id,
            "values": _json_safe(anchor.values),
            "findings": _json_safe(anchor.findings),
            "recommendations": _json_safe(anchor.recommendations),
            "evidence_ids": [e.id for e in anchor.evidence],
        }

    def _prompt(self, plan: ReasoningPlan, context: ExecutionContext,
                anchor: SolutionState, parent: FlowCandidate | None) -> str:
        payload = {
            "task": context.metadata.get("query") or plan.metadata.get("query") or plan.metadata.get("task") or plan.id,
            "plan": {
                "id": plan.id,
                "metadata": _json_safe(plan.metadata),
                "outputs": list(plan.outputs),
                "nodes": [
                    {"id": n.id, "operator": n.operator, "inputs": _json_safe(n.inputs),
                     "params": _json_safe(n.params), "depends_on": list(n.depends_on)}
                    for n in plan.nodes
                ],
            },
            "state": _json_safe(context.state),
            "anchor": self._anchor_payload(anchor),
            "allowed_operators": self._catalog(),
            "previous_candidate": None if parent is None else {
                "answer": parent.answer,
                "rationale": parent.rationale,
                "confidence": parent.confidence,
                "evidence_ids": list(parent.evidence_ids),
                "tool_observations": list(parent.tool_observations),
            },
        }
        return _json_text(payload, max_chars=self.max_prompt_chars)

    def _parse_tools(self, raw: Any) -> tuple[FlowToolRequest, ...]:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        out = []
        for item in raw[:self.max_tool_requests]:
            if not isinstance(item, Mapping):
                continue
            op = str(item.get("operator", ""))
            inputs = item.get("inputs", {})
            params = item.get("params", {})
            if not op or not self._operator_allowed(op) or not isinstance(inputs, Mapping) or not isinstance(params, Mapping):
                continue
            out.append(FlowToolRequest(op, dict(inputs), dict(params), str(item["id"]) if item.get("id") is not None else None))
        return tuple(out)

    def _execute_tools(self, requests: Sequence[FlowToolRequest], context: ExecutionContext) -> tuple[Mapping[str, Any], ...]:
        observations = []
        for req in requests:
            try:
                result: OperatorResult = self.base_engine.operators.get(req.operator).handler(context, req.inputs, req.params)
                observations.append({
                    "id": req.request_id,
                    "operator": req.operator,
                    "status": "ok",
                    "value": _json_safe(result.value),
                    "evidence_ids": [e.id for e in result.evidence],
                    "metadata": _json_safe(result.metadata),
                })
            except Exception as exc:
                observations.append({"id": req.request_id, "operator": req.operator,
                                     "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return tuple(observations)

    def _evidence_score(self, ids: Sequence[str], anchor: SolutionState) -> float:
        available = {e.id for e in anchor.evidence}
        selected = set(ids)
        if not available:
            return 1.0 if not selected else 0.0
        if not selected:
            return 0.0
        valid = len(selected & available)
        return .5 * valid / len(selected) + .5 * valid / len(available)

    def _tool_score(self, observations: Sequence[Mapping[str, Any]]) -> float:
        return 0.0 if not observations else sum(i.get("status") == "ok" for i in observations) / len(observations)

    def _features(self, *, logprob: float, confidence: float, evidence: float, tool: float,
                  parsed: bool, final: bool, verifier: float | None) -> tuple[float, ...]:
        return (
            float(logprob),
            float(confidence),
            float(evidence),
            float(tool),
            float(parsed),
            float(final),
            float(verifier or 0.0),
        )

    def _candidate(self, sample: BackboneSample, step: int, branch: int,
                   parent: FlowCandidate | None, anchor: SolutionState,
                   context: ExecutionContext) -> FlowCandidate:
        raw = _extract_json_object(sample.text)
        parsed = raw is not None
        data = dict(raw or {})
        answer = str(data.get("answer", sample.text.strip()))
        rationale = str(data.get("rationale", ""))
        confidence = _clamp(_float(data.get("confidence")), 0, 1)
        ids_raw = data.get("evidence_ids", [])
        ids = tuple(str(i) for i in ids_raw) if isinstance(ids_raw, Sequence) and not isinstance(ids_raw, (str, bytes)) else ()
        requests = self._parse_tools(data.get("tool_requests", []))
        observations = self._execute_tools(requests, context)
        final = bool(data.get("final", False))
        evidence_score = self._evidence_score(ids, anchor)
        tool_score = self._tool_score(observations)
        heuristic = (
            self.score_weights["logprob"] * sample.logprob
            + self.score_weights["confidence"] * confidence
            + self.score_weights["evidence"] * evidence_score
            + self.score_weights["tool"] * tool_score
            + self.score_weights["parse"] * float(parsed)
            + self.score_weights["final"] * float(final)
        )
        features = self._features(logprob=sample.logprob, confidence=confidence,
                                  evidence=evidence_score, tool=tool_score,
                                  parsed=parsed, final=final, verifier=None)
        prior_score = self.fast_prior.score(features)
        score = heuristic + self.prior_weight * prior_score
        cid = f"s{step}:b{branch}:{parent.id if parent else 'root'}"
        return FlowCandidate(cid, step, parent.id if parent else None, answer, rationale,
                             confidence, ids, requests, observations, final, parsed,
                             sample.logprob, heuristic, None, prior_score, score)

    def _verify(self, candidate: FlowCandidate, anchor: SolutionState, *, seed: int | None) -> FlowCandidate:
        if not self.verifier_enabled:
            return candidate
        payload = {
            "anchor": self._anchor_payload(anchor),
            "candidate": {
                "answer": candidate.answer,
                "rationale": candidate.rationale,
                "confidence": candidate.confidence,
                "evidence_ids": list(candidate.evidence_ids),
                "tool_observations": list(candidate.tool_observations),
            },
        }
        samples = self.backbone.sample(
            system_prompt=self.VERIFIER_SYSTEM_PROMPT,
            user_prompt=_json_text(payload, max_chars=self.max_prompt_chars),
            branches=1,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=self.verifier_max_new_tokens,
            seed=seed,
        )
        raw = _extract_json_object(samples[0].text) if samples else None
        verifier_score = _clamp(_float((raw or {}).get("score")), 0, 1)
        features = self._features(
            logprob=candidate.logprob,
            confidence=candidate.confidence,
            evidence=self._evidence_score(candidate.evidence_ids, anchor),
            tool=self._tool_score(candidate.tool_observations),
            parsed=candidate.parsed,
            final=candidate.final,
            verifier=verifier_score,
        )
        prior_score = self.fast_prior.score(features)
        score = candidate.heuristic_score + self.score_weights["verifier"] * verifier_score + self.prior_weight * prior_score
        return replace(candidate, verifier_score=verifier_score, prior_score=prior_score, score=score)

    def _adapt(self, candidate: FlowCandidate, anchor: SolutionState) -> Mapping[str, float] | None:
        if not self.adaptation_enabled:
            return None
        evidence = self._evidence_score(candidate.evidence_ids, anchor)
        tool = self._tool_score(candidate.tool_observations)
        features = self._features(
            logprob=candidate.logprob,
            confidence=candidate.confidence,
            evidence=evidence,
            tool=tool,
            parsed=candidate.parsed,
            final=candidate.final,
            verifier=candidate.verifier_score,
        )
        if candidate.verifier_score is not None:
            target = candidate.verifier_score
        else:
            target = _clamp((candidate.confidence + evidence + float(candidate.parsed) + float(candidate.final)) / 4.0, 0, 1)
        return self.fast_prior.observe(features, target)

    def _finalize(self, anchor: SolutionState, best: FlowCandidate | None,
                  trace: Sequence[FlowCandidate], steps: int,
                  adaptation_trace: Sequence[Mapping[str, Any]]) -> SolutionState:
        if best is None:
            if not self.fallback_to_anchor:
                raise ArtifactError("stochastic-flow produced no usable candidates")
            return replace(anchor, metadata={
                **anchor.metadata,
                "artifact": dict(self.metadata),
                "stochastic_flow": {
                    "status": "fallback-anchor",
                    "architecture": self.artifact.architecture,
                    "steps": steps,
                    "candidate_count": 0,
                },
            })
        flow_trace = [
            {
                "id": i.id,
                "step": i.step,
                "parent_id": i.parent_id,
                "score": i.score,
                "heuristic_score": i.heuristic_score,
                "prior_score": i.prior_score,
                "verifier_score": i.verifier_score,
                "confidence": i.confidence,
                "final": i.final,
                "parsed": i.parsed,
                "operators": [r.operator for r in i.tool_requests],
            }
            for i in trace
        ]
        flow = {
            "status": "ok",
            "architecture": self.artifact.architecture,
            "answer": best.answer,
            "rationale": best.rationale,
            "confidence": best.confidence,
            "evidence_ids": list(best.evidence_ids),
            "score": best.score,
            "steps": steps,
            "candidate_id": best.id,
            "candidate_count": len(trace),
            "tool_observations": list(best.tool_observations),
            "trace": flow_trace,
            "adaptation": {
                "backend": dict(self.fast_prior.metadata),
                "updates": list(adaptation_trace),
            },
        }
        if self.adaptation_snapshot_path:
            path = self.fast_prior.snapshot(self.adaptation_snapshot_path)
            flow["adaptation"]["snapshot"] = str(path)
        return replace(anchor, metadata={
            **anchor.metadata,
            "artifact": dict(self.metadata),
            "language": best.answer,
            "stochastic_flow": flow,
        })

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        self.fast_prior.reset()
        anchor = self.base_engine.reason(plan, context)
        beam: list[FlowCandidate | None] = [None]
        all_candidates: list[FlowCandidate] = []
        adaptation_trace: list[Mapping[str, Any]] = []
        steps = 0
        for step in range(1, self.max_steps + 1):
            steps = step
            proposed: list[FlowCandidate] = []
            for parent_index, parent in enumerate(beam):
                seed = None if self.seed is None else self.seed + step * 1000 + parent_index * 100
                samples = self.backbone.sample(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=self._prompt(plan, context, anchor, parent),
                    branches=self.branches,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    seed=seed,
                )
                proposed.extend(
                    self._candidate(sample, step, branch, parent, anchor, context)
                    for branch, sample in enumerate(samples)
                )
            if not proposed:
                break
            proposed.sort(key=lambda i: i.score, reverse=True)
            if self.verifier_enabled and self.verify_top_k:
                verified = {
                    candidate.id: self._verify(
                        candidate,
                        anchor,
                        seed=None if self.seed is None else self.seed + step * 10000 + rank,
                    )
                    for rank, candidate in enumerate(proposed[:self.verify_top_k])
                }
                proposed = [verified.get(i.id, i) for i in proposed]
            proposed.sort(key=lambda i: i.score, reverse=True)
            winner = proposed[0]
            update = self._adapt(winner, anchor)
            if update is not None:
                adaptation_trace.append({"step": step, "candidate_id": winner.id, **dict(update)})
                # Re-score after the prior update so the new prior can influence this step's beam.
                rescored: list[FlowCandidate] = []
                for candidate in proposed:
                    features = self._features(
                        logprob=candidate.logprob,
                        confidence=candidate.confidence,
                        evidence=self._evidence_score(candidate.evidence_ids, anchor),
                        tool=self._tool_score(candidate.tool_observations),
                        parsed=candidate.parsed,
                        final=candidate.final,
                        verifier=candidate.verifier_score,
                    )
                    prior_score = self.fast_prior.score(features)
                    base_score = candidate.heuristic_score + self.score_weights["verifier"] * float(candidate.verifier_score or 0.0)
                    rescored.append(replace(candidate, prior_score=prior_score,
                                            score=base_score + self.prior_weight * prior_score))
                proposed = sorted(rescored, key=lambda i: i.score, reverse=True)
            all_candidates.extend(proposed)
            beam = list(proposed[:self.beam_width])
            if step >= self.min_steps and beam[0].final and not beam[0].tool_requests:
                break
        best = max(all_candidates, key=lambda i: i.score) if all_candidates else None
        return self._finalize(anchor, best, all_candidates, steps, adaptation_trace)


__all__ = [
    "BackboneSample", "FlowCandidate", "FlowToolRequest", "ReferenceBackbone",
    "StochasticBackbone", "StochasticFlowReasoner", "TransformersCausalBackbone",
    "MLXCausalBackbone", "LlamaCppBackbone",
]
