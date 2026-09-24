"""Probabilistic LCFA reasoning over a frozen pretrained language-model backbone.

The stochastic-flow backend keeps the typed LCFA operator graph as a grounded
anchor, then spends test-time compute on candidate generation, optional pure
operator calls, verification, and branch selection. It never executes
side-effecting actions during reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from fnmatch import fnmatch
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from safetensors import safe_open

from .artifact import STOCHASTIC_FLOW_ARCHITECTURE, ArtifactError, ArtifactManifest
from .engine import LCFA
from .protocol import EvidenceRef, EvidenceValue, ExecutionContext, OperatorResult, ReasoningPlan, SolutionState


@dataclass(frozen=True, slots=True)
class BackboneSample:
    text: str
    logprob: float = 0.0


class StochasticBackbone(Protocol):
    metadata: Mapping[str, Any]

    def sample(self, *, system_prompt: str, user_prompt: str, branches: int,
               temperature: float, top_p: float, max_new_tokens: int,
               seed: int | None = None) -> tuple[BackboneSample, ...]: ...


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
    score: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, EvidenceValue):
        return {"value": _json_safe(value.value), "evidence": [_json_safe(i) for i in value.evidence]}
    if isinstance(value, EvidenceRef):
        return {"id": value.id, "source": value.source, "metadata": _json_safe(value.metadata)}
    if is_dataclass(value): return _json_safe(asdict(value))
    if isinstance(value, Mapping): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)): return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return repr(value)


def _json_text(value: Any, *, max_chars: int) -> str:
    text = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)
    return text if len(text) <= max_chars else text[: max(0, max_chars - 32)] + "...<truncated>"


def _extract_json_object(text: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{": continue
        try: value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError: continue
        if isinstance(value, Mapping): return value
    return None


def _float(value: Any, default: float = 0.0) -> float:
    try: return float(value)
    except (TypeError, ValueError): return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scalar(weights_path: Path, name: str, default: float) -> float:
    with safe_open(str(weights_path), framework="np", device="cpu") as handle:
        if name not in handle.keys(): return default
        tensor = handle.get_tensor(name)
        if tensor.size != 1: raise ArtifactError(f"stochastic-flow score tensor {name!r} must be scalar")
        return float(tensor.reshape(-1)[0])


class ReferenceBackbone:
    """Tiny non-SOTA backbone used only to exercise the stochastic runtime in CI."""
    metadata = {"type": "reference", "quality": "test-only"}

    def sample(self, *, system_prompt: str, user_prompt: str, branches: int,
               temperature: float, top_p: float, max_new_tokens: int,
               seed: int | None = None) -> tuple[BackboneSample, ...]:
        del user_prompt, temperature, top_p, max_new_tokens, seed
        if "LCFA_VERIFIER" in system_prompt:
            return (BackboneSample('{"score": 0.75}', -0.01),)
        text = json.dumps({
            "answer": "Grounded LCFA solution preserved by the reference stochastic backbone.",
            "rationale": "The typed LCFA anchor is retained while the stochastic-flow path is exercised.",
            "confidence": 0.75, "evidence_ids": [], "tool_requests": [], "final": True,
        })
        return tuple(BackboneSample(text, -0.05 - index * 0.01) for index in range(branches))


class TransformersCausalBackbone:
    """Local Hugging Face causal-LM adapter loaded only when needed."""
    def __init__(self, model_path: str | Path, *, device_map: str | Mapping[str, Any] = "auto",
                 dtype: str = "auto", local_files_only: bool = True,
                 trust_remote_code: bool = False, max_input_tokens: int = 16384) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ArtifactError("transformers stochastic backbone requires `pip install -e '.[transformers]'`") from exc
        self._torch = torch
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=local_files_only,
                                                       trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=device_map, dtype=dtype,
                                                          local_files_only=local_files_only,
                                                          trust_remote_code=trust_remote_code)
        self.model.eval()
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_input_tokens = max(256, int(max_input_tokens))
        self.metadata = {"type": "transformers-local", "model_path": self.model_path,
                         "device_map": str(device_map), "dtype": str(dtype),
                         "local_files_only": bool(local_files_only)}

    def _render(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try: return apply_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception: pass
        return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n"

    def sample(self, *, system_prompt: str, user_prompt: str, branches: int,
               temperature: float, top_p: float, max_new_tokens: int,
               seed: int | None = None) -> tuple[BackboneSample, ...]:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
        prompt = self._render(system_prompt, user_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        model_device = getattr(self.model, "device", None)
        if model_device is not None: inputs = {k: v.to(model_device) for k, v in inputs.items()}
        branches = max(1, int(branches)); do_sample = temperature > 0.0
        kwargs: dict[str, Any] = {"max_new_tokens": max(1, int(max_new_tokens)),
            "num_return_sequences": branches, "do_sample": do_sample,
            "return_dict_in_generate": True, "output_scores": True,
            "pad_token_id": self.tokenizer.pad_token_id}
        if do_sample:
            kwargs["temperature"] = max(1e-5, float(temperature)); kwargs["top_p"] = _clamp(float(top_p), 1e-5, 1.0)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
            transition = self.model.compute_transition_scores(outputs.sequences, outputs.scores,
                getattr(outputs, "beam_indices", None), normalize_logits=True)
        input_length = 1 if self.model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
        generated = outputs.sequences[:, input_length:]
        result = []
        for token_row, score_row in zip(generated, transition):
            text = self.tokenizer.decode(token_row, skip_special_tokens=True)
            usable = score_row[score_row < 0]
            avg = float(usable.mean().item()) if usable.numel() else 0.0
            result.append(BackboneSample(text=text, logprob=avg))
        return tuple(result)


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

    def __init__(self, artifact: ArtifactManifest, *, weights_path: Path, backbone: StochasticBackbone,
                 base_engine: LCFA | None = None) -> None:
        if artifact.architecture != STOCHASTIC_FLOW_ARCHITECTURE: raise ArtifactError("wrong stochastic-flow architecture")
        if artifact.base_backend != "lcfa-zero": raise ArtifactError("unsupported stochastic-flow base backend")
        self.artifact = artifact; self.weights_path = weights_path; self.base_engine = base_engine or LCFA(); self.backbone = backbone
        config = dict(artifact.config); flow = dict(config.get("flow", {})); verifier = dict(config.get("verifier", {})); tools = dict(config.get("tools", {}))
        self.branches = max(1, int(flow.get("branches", 4))); self.beam_width = max(1, int(flow.get("beam_width", 2)))
        self.min_steps = max(1, int(flow.get("min_steps", 1))); self.max_steps = max(self.min_steps, int(flow.get("max_steps", 4)))
        self.temperature = float(flow.get("temperature", 0.7)); self.top_p = float(flow.get("top_p", 0.95))
        self.max_new_tokens = max(32, int(flow.get("max_new_tokens", 512))); self.max_prompt_chars = max(2048, int(flow.get("max_prompt_chars", 24000)))
        self.max_tool_requests = max(0, int(flow.get("max_tool_requests", 4))); self.fallback_to_anchor = bool(flow.get("fallback_to_anchor", True))
        self.seed = int(flow["seed"]) if flow.get("seed") is not None else None
        self.verifier_enabled = bool(verifier.get("enabled", True)); self.verify_top_k = max(0, int(verifier.get("top_k", 2)))
        self.verifier_max_new_tokens = max(8, int(verifier.get("max_new_tokens", 32)))
        self.allowed_operator_patterns = tuple(str(i) for i in tools.get("allowed", []))
        self.score_weights = {name: _scalar(weights_path, f"score.{name}", default) for name, default in {
            "logprob": .10, "confidence": 1.0, "evidence": .75, "tool": .25,
            "parse": .25, "final": .15, "verifier": 1.0}.items()}
        self.metadata = {"backend": "stochastic-flow", "artifact_id": artifact.id,
            "artifact_version": artifact.version, "architecture": artifact.architecture,
            "weights_sha256": artifact.weights.sha256, "backbone": dict(backbone.metadata),
            "branches": self.branches, "beam_width": self.beam_width, "max_steps": self.max_steps}

    @classmethod
    def from_manifest(cls, root: Path, manifest: ArtifactManifest, *, weights_path: Path,
                      base_engine: LCFA | None = None, runtime_options: Mapping[str, Any] | None = None) -> "StochasticFlowReasoner":
        runtime = dict(runtime_options or {}); cfg = dict(manifest.config.get("backbone", {})); kind = str(cfg.get("type", "transformers-local"))
        if kind == "reference":
            if not bool(manifest.metadata.get("reference_only", False)): raise ArtifactError("reference backbone is allowed only for reference_only artifacts")
            backbone: StochasticBackbone = ReferenceBackbone()
        elif kind == "transformers-local":
            raw_path = runtime.get("backbone_path") or cfg.get("path")
            if not raw_path: raise ArtifactError("stochastic-flow artifact requires config.backbone.path or runtime backbone_path")
            model_path = Path(str(raw_path)); model_path = model_path if model_path.is_absolute() else (root / model_path).resolve()
            backbone = TransformersCausalBackbone(model_path,
                device_map=runtime.get("device_map", cfg.get("device_map", "auto")),
                dtype=str(runtime.get("dtype", cfg.get("dtype", "auto"))),
                local_files_only=bool(cfg.get("local_files_only", True)),
                trust_remote_code=bool(cfg.get("trust_remote_code", False)),
                max_input_tokens=int(cfg.get("max_input_tokens", 16384)))
        else: raise ArtifactError(f"unsupported stochastic backbone type: {kind!r}")
        return cls(manifest, weights_path=weights_path, backbone=backbone, base_engine=base_engine)

    def _operator_allowed(self, name: str) -> bool: return any(fnmatch(name, p) for p in self.allowed_operator_patterns)
    def _catalog(self) -> list[dict[str, str]]:
        return [{"name": n, "description": self.base_engine.operators.get(n).description}
                for n in self.base_engine.operators.names() if self._operator_allowed(n)]
    def _anchor_payload(self, anchor: SolutionState) -> dict[str, Any]:
        return {"plan_id": anchor.plan_id, "values": _json_safe(anchor.values), "findings": _json_safe(anchor.findings),
                "recommendations": _json_safe(anchor.recommendations), "evidence_ids": [e.id for e in anchor.evidence]}
    def _prompt(self, plan: ReasoningPlan, context: ExecutionContext, anchor: SolutionState, parent: FlowCandidate | None) -> str:
        payload = {"task": context.metadata.get("query") or plan.metadata.get("query") or plan.metadata.get("task") or plan.id,
            "plan": {"id": plan.id, "metadata": _json_safe(plan.metadata), "outputs": list(plan.outputs),
                     "nodes": [{"id": n.id, "operator": n.operator, "inputs": _json_safe(n.inputs), "params": _json_safe(n.params), "depends_on": list(n.depends_on)} for n in plan.nodes]},
            "state": _json_safe(context.state), "anchor": self._anchor_payload(anchor), "allowed_operators": self._catalog(),
            "previous_candidate": None if parent is None else {"answer": parent.answer, "rationale": parent.rationale,
                "confidence": parent.confidence, "evidence_ids": list(parent.evidence_ids), "tool_observations": list(parent.tool_observations)}}
        return _json_text(payload, max_chars=self.max_prompt_chars)

    def _parse_tools(self, raw: Any) -> tuple[FlowToolRequest, ...]:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)): return ()
        out = []
        for item in raw[:self.max_tool_requests]:
            if not isinstance(item, Mapping): continue
            op = str(item.get("operator", "")); inputs = item.get("inputs", {}); params = item.get("params", {})
            if not op or not self._operator_allowed(op) or not isinstance(inputs, Mapping) or not isinstance(params, Mapping): continue
            out.append(FlowToolRequest(op, dict(inputs), dict(params), str(item["id"]) if item.get("id") is not None else None))
        return tuple(out)

    def _execute_tools(self, requests: Sequence[FlowToolRequest], context: ExecutionContext) -> tuple[Mapping[str, Any], ...]:
        obs = []
        for req in requests:
            try:
                result: OperatorResult = self.base_engine.operators.get(req.operator).handler(context, req.inputs, req.params)
                obs.append({"id": req.request_id, "operator": req.operator, "status": "ok", "value": _json_safe(result.value),
                            "evidence_ids": [e.id for e in result.evidence], "metadata": _json_safe(result.metadata)})
            except Exception as exc:
                obs.append({"id": req.request_id, "operator": req.operator, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return tuple(obs)

    def _evidence_score(self, ids: Sequence[str], anchor: SolutionState) -> float:
        available = {e.id for e in anchor.evidence}; selected = set(ids)
        if not available: return 1.0 if not selected else 0.0
        if not selected: return 0.0
        valid = len(selected & available); return .5 * valid / len(selected) + .5 * valid / len(available)
    def _tool_score(self, obs: Sequence[Mapping[str, Any]]) -> float:
        return 0.0 if not obs else sum(i.get("status") == "ok" for i in obs) / len(obs)

    def _candidate(self, sample: BackboneSample, step: int, branch: int, parent: FlowCandidate | None,
                   anchor: SolutionState, context: ExecutionContext) -> FlowCandidate:
        raw = _extract_json_object(sample.text); parsed = raw is not None; data = dict(raw or {})
        answer = str(data.get("answer", sample.text.strip())); rationale = str(data.get("rationale", "")); confidence = _clamp(_float(data.get("confidence")), 0, 1)
        ids_raw = data.get("evidence_ids", []); ids = tuple(str(i) for i in ids_raw) if isinstance(ids_raw, Sequence) and not isinstance(ids_raw, (str, bytes)) else ()
        reqs = self._parse_tools(data.get("tool_requests", [])); observations = self._execute_tools(reqs, context); final = bool(data.get("final", False))
        heuristic = self.score_weights["logprob"] * sample.logprob + self.score_weights["confidence"] * confidence + \
            self.score_weights["evidence"] * self._evidence_score(ids, anchor) + self.score_weights["tool"] * self._tool_score(observations) + \
            self.score_weights["parse"] * float(parsed) + self.score_weights["final"] * float(final)
        cid = f"s{step}:b{branch}:{parent.id if parent else 'root'}"
        return FlowCandidate(cid, step, parent.id if parent else None, answer, rationale, confidence, ids, reqs, observations,
                             final, parsed, sample.logprob, heuristic, None, heuristic)

    def _verify(self, candidate: FlowCandidate, anchor: SolutionState, *, seed: int | None) -> FlowCandidate:
        if not self.verifier_enabled: return candidate
        payload = {"anchor": self._anchor_payload(anchor), "candidate": {"answer": candidate.answer, "rationale": candidate.rationale,
            "confidence": candidate.confidence, "evidence_ids": list(candidate.evidence_ids), "tool_observations": list(candidate.tool_observations)}}
        samples = self.backbone.sample(system_prompt=self.VERIFIER_SYSTEM_PROMPT,
            user_prompt=_json_text(payload, max_chars=self.max_prompt_chars), branches=1, temperature=0.0, top_p=1.0,
            max_new_tokens=self.verifier_max_new_tokens, seed=seed)
        raw = _extract_json_object(samples[0].text) if samples else None; score = _clamp(_float((raw or {}).get("score")), 0, 1)
        return replace(candidate, verifier_score=score, score=candidate.heuristic_score + self.score_weights["verifier"] * score)

    def _finalize(self, anchor: SolutionState, best: FlowCandidate | None, trace: Sequence[FlowCandidate], steps: int) -> SolutionState:
        if best is None:
            if not self.fallback_to_anchor: raise ArtifactError("stochastic-flow produced no usable candidates")
            return replace(anchor, metadata={**anchor.metadata, "artifact": dict(self.metadata),
                "stochastic_flow": {"status": "fallback-anchor", "architecture": self.artifact.architecture, "steps": steps, "candidate_count": 0}})
        flow_trace = [{"id": i.id, "step": i.step, "parent_id": i.parent_id, "score": i.score,
                       "heuristic_score": i.heuristic_score, "verifier_score": i.verifier_score,
                       "confidence": i.confidence, "final": i.final, "parsed": i.parsed,
                       "operators": [r.operator for r in i.tool_requests]} for i in trace]
        flow = {"status": "ok", "architecture": self.artifact.architecture, "answer": best.answer,
                "rationale": best.rationale, "confidence": best.confidence, "evidence_ids": list(best.evidence_ids),
                "score": best.score, "steps": steps, "candidate_id": best.id, "candidate_count": len(trace),
                "tool_observations": list(best.tool_observations), "trace": flow_trace}
        return replace(anchor, metadata={**anchor.metadata, "artifact": dict(self.metadata), "language": best.answer, "stochastic_flow": flow})

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        anchor = self.base_engine.reason(plan, context); beam: list[FlowCandidate | None] = [None]; all_candidates: list[FlowCandidate] = []; steps = 0
        for step in range(1, self.max_steps + 1):
            steps = step; proposed: list[FlowCandidate] = []
            for parent_index, parent in enumerate(beam):
                seed = None if self.seed is None else self.seed + step * 1000 + parent_index * 100
                samples = self.backbone.sample(system_prompt=self.SYSTEM_PROMPT, user_prompt=self._prompt(plan, context, anchor, parent),
                    branches=self.branches, temperature=self.temperature, top_p=self.top_p, max_new_tokens=self.max_new_tokens, seed=seed)
                proposed.extend(self._candidate(s, step, b, parent, anchor, context) for b, s in enumerate(samples))
            if not proposed: break
            proposed.sort(key=lambda i: i.heuristic_score, reverse=True)
            if self.verifier_enabled and self.verify_top_k:
                verified = {c.id: self._verify(c, anchor, seed=None if self.seed is None else self.seed + step * 10000 + rank)
                            for rank, c in enumerate(proposed[:self.verify_top_k])}
                proposed = [verified.get(i.id, i) for i in proposed]
            proposed.sort(key=lambda i: i.score, reverse=True); all_candidates.extend(proposed); beam = list(proposed[:self.beam_width])
            if step >= self.min_steps and beam[0].final and not beam[0].tool_requests: break
        best = max(all_candidates, key=lambda i: i.score) if all_candidates else None
        return self._finalize(anchor, best, all_candidates, steps)


__all__ = ["BackboneSample", "FlowCandidate", "FlowToolRequest", "ReferenceBackbone", "StochasticBackbone", "StochasticFlowReasoner", "TransformersCausalBackbone"]
