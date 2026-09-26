"""RWKV-7 recurrent policy controller for LCFA semantic agents.

Action, stop, value, and optional retrieval-pointer decisions are predicted
from RWKV recurrent state. Search/read arguments resolve from the same ranked
retrieval context used during training. Language generation is reserved for
edit/process arguments that cannot be resolved deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .protocol import SolutionState
from .recurrent_transitions import ACTION_VOCAB, normalize_event
from .torch_runtime import resolve_device, resolve_dtype

RWKV_CONTROLLER_FORMAT = "lcfa.rwkv-controller.v2"
DEFAULT_RWKV_MODEL = "RWKV/RWKV7-G1j-1.5B-20260831"
DEFAULT_POINTER_SLOTS = 16


class RWKVControllerError(RuntimeError):
    pass


class RecurrentPolicy(Protocol):
    def reset(self, goal: str, solution: SolutionState) -> None: ...
    def choose(
        self,
        goal: str,
        solution: SolutionState,
        recent: Sequence[Mapping[str, Any]],
        step: int,
    ) -> Mapping[str, Any]: ...
    def observe(
        self,
        action: Mapping[str, Any] | None,
        observation: Mapping[str, Any],
        solution: SolutionState,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class RecurrentDecision:
    action: str
    action_confidence: float
    stop_probability: float
    value: float
    pointer_index: int | None = None
    pointer_confidence: float | None = None


def _json_object(text: str) -> Mapping[str, Any] | None:
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


def _compact_cognition(solution: SolutionState) -> Mapping[str, Any]:
    raw = solution.values.get("cognition", {}) if isinstance(solution.values, Mapping) else {}
    if not isinstance(raw, Mapping):
        return {}
    return {
        "active_concepts": list(raw.get("active_concepts", ()))[:16],
        "hypotheses": list(raw.get("hypotheses", ()))[:8],
        "open_questions": list(raw.get("open_questions", ()))[:6],
        "candidate_locations": list(raw.get("candidate_locations", ()))[:16],
        "candidate_queries": list(raw.get("candidate_queries", ()))[:8],
        "candidate_paths": list(raw.get("candidate_paths", ()))[:16],
        "observations": list(raw.get("observations", ()))[:8],
        "terminal": bool(raw.get("terminal", False)),
    }


def argument_prompt(
    action: str,
    goal: str,
    event: Mapping[str, Any],
    *,
    target_path: str | None = None,
) -> str:
    """Prompt shared by live edit rendering and supervised argument tuning."""
    return (
        "LCFA action argument renderer. The recurrent policy already chose the action.\n"
        "Return ONLY one JSON object containing inputs for that action.\n"
        "Do not choose another action and do not include prose.\n"
        f"action={action}\n"
        f"goal={goal}\n"
        f"target_path={target_path or ''}\n"
        f"event={json.dumps(normalize_event(event), sort_keys=True, ensure_ascii=False, default=str)}\n"
    )


class RWKVRecurrentPolicy:
    """Inference policy using RWKV recurrent state plus trained LCFA heads."""

    def __init__(
        self,
        model_id: str = DEFAULT_RWKV_MODEL,
        *,
        heads_path: str | Path | None = None,
        backbone_weights_path: str | Path | None = None,
        device: str | None = "auto",
        dtype: str = "auto",
        action_names: Sequence[str] = ACTION_VOCAB,
        pointer_slots: int = 0,
        stop_threshold: float = 0.5,
        max_argument_tokens: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RWKVControllerError(
                "RWKV controller requires `pip install -e '.[rwkv]'`"
            ) from exc

        self._torch = torch
        self.model_id = str(model_id)
        self.action_names = tuple(str(item) for item in action_names)
        self.pointer_slots = max(0, int(pointer_slots))
        self.stop_threshold = float(stop_threshold)
        self.max_argument_tokens = max(64, int(max_argument_tokens))
        if "stop" not in self.action_names:
            raise RWKVControllerError("action vocabulary must contain stop")

        self.device = resolve_device(torch, device)
        try:
            self.dtype_name, dtype_value = resolve_dtype(torch, self.device, dtype)
        except ValueError as exc:
            raise RWKVControllerError(str(exc)) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype_value,
            trust_remote_code=True,
        ).to(self.device)
        if backbone_weights_path is not None:
            try:
                from safetensors.torch import load_model
            except ImportError as exc:
                raise RWKVControllerError("safetensors torch support is required") from exc
            missing, unexpected = load_model(
                self.model,
                str(backbone_weights_path),
                strict=False,
                device=self.device,
            )
            if missing or unexpected:
                raise RWKVControllerError(
                    "backbone tensor mismatch: "
                    f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
                )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise RWKVControllerError("RWKV model config does not expose hidden_size")
        self.hidden_size = hidden_size
        modules: dict[str, Any] = {
            "action": torch.nn.Linear(hidden_size, len(self.action_names)),
            "stop": torch.nn.Linear(hidden_size, 1),
            "value": torch.nn.Linear(hidden_size, 1),
        }
        if self.pointer_slots > 0:
            modules["pointer"] = torch.nn.Linear(hidden_size, self.pointer_slots)
        self.heads = torch.nn.ModuleDict(modules).to(self.device)
        self.heads.eval()
        if heads_path is not None:
            self.load_heads(heads_path)

        self._state: Any = None
        self._hidden: Any = None
        self._goal = ""
        self._event_history: list[Mapping[str, Any]] = []
        self.metadata = {
            "type": "rwkv7-recurrent-policy",
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "actions": list(self.action_names),
            "pointer_slots": self.pointer_slots,
            "stop_threshold": self.stop_threshold,
            "loader": "transformers-remote-code",
            "device": self.device,
            "dtype": self.dtype_name,
            "event_schema": "goal + retrieval + compact prior action/observation",
            "backbone_weights": str(backbone_weights_path) if backbone_weights_path else None,
        }

    def load_heads(self, path: str | Path) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RWKVControllerError("safetensors torch support is required") from exc
        state = load_file(str(path), device=self.device)
        expected = self.heads.state_dict()
        missing = set(expected) - set(state)
        extra = set(state) - set(expected)
        if missing or extra:
            raise RWKVControllerError(
                f"controller head tensor mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        self.heads.load_state_dict(state, strict=True)
        self.heads.eval()

    def _forward_event(self, payload: Mapping[str, Any]) -> None:
        torch = self._torch
        event = normalize_event(payload)
        self._event_history.append(event)
        self._event_history = self._event_history[-8:]
        text = json.dumps(
            event, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ) + "\n"
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        kwargs: dict[str, Any] = {
            "input_ids": encoded["input_ids"].to(self.device),
            "use_cache": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if self._state is not None:
            kwargs["state"] = self._state
        with torch.inference_mode():
            outputs = self.model(**kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        state = getattr(outputs, "state", None)
        if not hidden_states or state is None:
            raise RWKVControllerError(
                "RWKV forward must return hidden_states and recurrent state"
            )
        self._state = state
        self._hidden = hidden_states[-1][:, -1, :].detach()

    def reset(self, goal: str, solution: SolutionState) -> None:
        cognition = _compact_cognition(solution)
        self._state = None
        self._hidden = None
        self._goal = str(goal)
        self._event_history = []
        self._forward_event({
            "kind": "goal",
            "goal": self._goal,
            "retrieval": {
                "queries": cognition.get("candidate_queries", ()),
                "paths": cognition.get("candidate_paths", ()),
            },
        })

    def decision(self) -> RecurrentDecision:
        if self._hidden is None:
            raise RWKVControllerError("controller has not been reset")
        torch = self._torch
        with torch.inference_mode():
            action_logits = self.heads["action"](self._hidden.float())
            action_probs = torch.softmax(action_logits, dim=-1)[0]
            action_index = int(torch.argmax(action_probs).item())
            stop_probability = float(
                torch.sigmoid(self.heads["stop"](self._hidden.float()))[0, 0].item()
            )
            value = float(
                torch.sigmoid(self.heads["value"](self._hidden.float()))[0, 0].item()
            )
            pointer_index = None
            pointer_confidence = None
            if "pointer" in self.heads:
                pointer_probs = torch.softmax(
                    self.heads["pointer"](self._hidden.float()), dim=-1
                )[0]
                pointer_index = int(torch.argmax(pointer_probs).item())
                pointer_confidence = float(pointer_probs[pointer_index].item())
        return RecurrentDecision(
            action=self.action_names[action_index],
            action_confidence=float(action_probs[action_index].item()),
            stop_probability=stop_probability,
            value=value,
            pointer_index=pointer_index,
            pointer_confidence=pointer_confidence,
        )

    @staticmethod
    def _sequence(cognition: Mapping[str, Any], key: str) -> tuple[str, ...]:
        raw = cognition.get(key, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        return tuple(str(item) for item in raw if str(item))

    def _candidate_path(
        self, solution: SolutionState, pointer: int | None = None
    ) -> str | None:
        cognition = _compact_cognition(solution)
        paths = self._sequence(cognition, "candidate_paths")
        if paths:
            return paths[int(pointer or 0) % len(paths)]
        return None

    def _candidate_query(
        self, solution: SolutionState, pointer: int | None = None
    ) -> str | None:
        cognition = _compact_cognition(solution)
        queries = self._sequence(cognition, "candidate_queries")
        if not queries:
            return None
        return queries[int(pointer or 0) % len(queries)]

    def _default_inputs(
        self,
        action: str,
        goal: str,
        solution: SolutionState,
        pointer: int | None,
    ) -> Mapping[str, Any] | None:
        if action == "repo.search":
            return {"query": self._candidate_query(solution, pointer) or goal}
        if action == "repo.read":
            path = self._candidate_path(solution, pointer)
            return {"path": path} if path else None
        if action in {"test.run", "verify.run", "git.status", "git.diff"}:
            return {}
        return None

    def _generate_inputs(
        self,
        action: str,
        goal: str,
        solution: SolutionState,
        pointer: int | None,
    ) -> Mapping[str, Any] | None:
        torch = self._torch
        target_path = self._candidate_path(solution, pointer)
        event = self._event_history[-1] if self._event_history else {
            "kind": "goal", "goal": goal
        }
        prompt = argument_prompt(action, goal, event, target_path=target_path)
        encoded = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_argument_tokens,
                do_sample=False,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                pad_token_id=(getattr(self.tokenizer, "pad_token_id", None) or 0),
            )
        text = self.tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        parsed = _json_object(text)
        if parsed is None:
            return None
        inputs = dict(parsed)
        if target_path and action in {"repo.replace", "repo.edit"}:
            inputs["path"] = target_path
        return inputs

    def choose(
        self,
        goal: str,
        solution: SolutionState,
        recent: Sequence[Mapping[str, Any]],
        step: int,
    ) -> Mapping[str, Any]:
        del recent, step
        if self._hidden is None:
            self.reset(goal, solution)
        decision = self.decision()
        controller = {
            "action_confidence": decision.action_confidence,
            "stop_probability": decision.stop_probability,
            "value": decision.value,
            "pointer_index": decision.pointer_index,
            "pointer_confidence": decision.pointer_confidence,
        }
        if decision.action == "stop" or decision.stop_probability >= self.stop_threshold:
            return {
                "hypothesis": None,
                "action": None,
                "final": True,
                "controller": controller,
            }

        inputs = self._default_inputs(
            decision.action, goal, solution, decision.pointer_index
        )
        if inputs is None:
            inputs = self._generate_inputs(
                decision.action, goal, solution, decision.pointer_index
            )
        if inputs is None:
            query = self._candidate_query(solution, decision.pointer_index) or goal
            return {
                "hypothesis": None,
                "action": {"name": "repo.search", "inputs": {"query": query}},
                "final": False,
                "controller": {
                    **controller,
                    "fallback_from": decision.action,
                    "fallback_reason": "argument renderer returned no JSON object",
                },
            }
        return {
            "hypothesis": None,
            "action": {"name": decision.action, "inputs": dict(inputs)},
            "final": False,
            "controller": controller,
        }

    def observe(
        self,
        action: Mapping[str, Any] | None,
        observation: Mapping[str, Any],
        solution: SolutionState,
    ) -> None:
        del solution
        self._forward_event({
            "kind": "transition",
            "action": dict(action) if isinstance(action, Mapping) else None,
            "observation": dict(observation),
        })


def load_rwkv_policy(
    controller_dir: str | Path,
    *,
    model_id: str | None = None,
    device: str | None = "auto",
    dtype: str = "auto",
) -> RWKVRecurrentPolicy:
    root = Path(controller_dir)
    manifest_path = root / "controller.json" if root.is_dir() else root
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("controller manifest must be a JSON object")
    resolved_root = manifest_path.parent
    weights = str(raw.get("weights") or "heads.safetensors")
    resolved_model = str(model_id or raw.get("model_id") or "")
    if not resolved_model:
        raise ValueError("controller manifest requires model_id")
    action_vocab = tuple(raw.get("action_vocab") or ())
    if not action_vocab:
        raise ValueError("controller manifest requires action_vocab")
    backbone_weights = raw.get("backbone_weights")
    return RWKVRecurrentPolicy(
        resolved_model,
        heads_path=resolved_root / weights,
        backbone_weights_path=(
            resolved_root / str(backbone_weights) if backbone_weights else None
        ),
        device=device,
        dtype=dtype,
        action_names=action_vocab,
        pointer_slots=int(raw.get("pointer_slots", 0) or 0),
    )


__all__ = [
    "DEFAULT_POINTER_SLOTS",
    "DEFAULT_RWKV_MODEL",
    "RWKV_CONTROLLER_FORMAT",
    "RWKVControllerError",
    "RWKVRecurrentPolicy",
    "RecurrentDecision",
    "RecurrentPolicy",
    "argument_prompt",
    "load_rwkv_policy",
]
