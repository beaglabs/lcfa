"""RWKV-7 recurrent policy controller for LCFA semantic agents.

The controller uses RWKV's constant-size recurrent state as working cognition.
Action, stop, and value are predicted by small task heads instead of language
JSON. The same RWKV language head is retained only for arguments that cannot be
derived deterministically (for example a source edit).

Torch/Transformers are optional and imported lazily so the base LCFA runtime
and CI remain lightweight.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .protocol import SolutionState
from .recurrent_transitions import ACTION_VOCAB


RWKV_CONTROLLER_FORMAT = "lcfa.rwkv-controller.v1"
DEFAULT_RWKV_MODEL = "RWKV/RWKV7-1.5B-20260805"


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
        "candidate_locations": list(raw.get("candidate_locations", ()))[:12],
        "observations": list(raw.get("observations", ()))[:8],
        "terminal": bool(raw.get("terminal", False)),
    }


class RWKVRecurrentPolicy:
    """Inference policy using RWKV recurrent state plus trained LCFA heads."""

    def __init__(
        self,
        model_id: str = DEFAULT_RWKV_MODEL,
        *,
        heads_path: str | Path | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        action_names: Sequence[str] = ACTION_VOCAB,
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
        self.stop_threshold = float(stop_threshold)
        self.max_argument_tokens = max(64, int(max_argument_tokens))
        if "stop" not in self.action_names:
            raise RWKVControllerError("action vocabulary must contain stop")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = str(device)
        dtype_value = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(str(dtype).lower())
        if dtype_value is None:
            raise RWKVControllerError(f"unsupported dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype_value,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise RWKVControllerError("RWKV model config does not expose hidden_size")
        self.hidden_size = hidden_size
        self.heads = torch.nn.ModuleDict({
            "action": torch.nn.Linear(hidden_size, len(self.action_names)),
            "stop": torch.nn.Linear(hidden_size, 1),
            "value": torch.nn.Linear(hidden_size, 1),
        }).to(self.device)
        self.heads.eval()
        if heads_path is not None:
            self.load_heads(heads_path)

        self._state: Any = None
        self._hidden: Any = None
        self._goal = ""
        self.metadata = {
            "type": "rwkv7-recurrent-policy",
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "actions": list(self.action_names),
            "stop_threshold": self.stop_threshold,
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
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
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
        if not hidden_states:
            raise RWKVControllerError("RWKV forward did not return hidden_states")
        state = getattr(outputs, "state", None)
        if state is None:
            raise RWKVControllerError("RWKV forward did not return recurrent state")
        self._state = state
        self._hidden = hidden_states[-1][:, -1, :].detach()

    def reset(self, goal: str, solution: SolutionState) -> None:
        self._state = None
        self._hidden = None
        self._goal = str(goal)
        self._forward_event({
            "kind": "goal",
            "goal": self._goal,
            "cognition": _compact_cognition(solution),
        })

    def decision(self) -> RecurrentDecision:
        if self._hidden is None:
            raise RWKVControllerError("controller has not been reset")
        torch = self._torch
        with torch.inference_mode():
            action_logits = self.heads["action"](self._hidden.float())
            action_probs = torch.softmax(action_logits, dim=-1)[0]
            action_index = int(torch.argmax(action_probs).item())
            stop_probability = float(torch.sigmoid(self.heads["stop"](self._hidden.float()))[0, 0].item())
            value = float(torch.sigmoid(self.heads["value"](self._hidden.float()))[0, 0].item())
        return RecurrentDecision(
            action=self.action_names[action_index],
            action_confidence=float(action_probs[action_index].item()),
            stop_probability=stop_probability,
            value=value,
        )

    def _candidate_path(self, solution: SolutionState) -> str | None:
        cognition = _compact_cognition(solution)
        for concept_id in cognition.get("candidate_locations", ()):
            text = str(concept_id)
            if text.startswith("file://"):
                path = text.split("/", 3)[-1]
                if path:
                    return path
        return None

    def _default_inputs(
        self,
        action: str,
        goal: str,
        solution: SolutionState,
    ) -> Mapping[str, Any] | None:
        if action == "repo.search":
            return {"query": goal}
        if action == "repo.read":
            path = self._candidate_path(solution)
            return {"path": path} if path else None
        if action == "test.run":
            return {}
        if action in {"git.status", "git.diff"}:
            return {}
        return None

    def _generate_inputs(
        self,
        action: str,
        goal: str,
        solution: SolutionState,
        recent: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | None:
        """Use Goose's language head only to render parameters/source edits."""
        torch = self._torch
        prompt = (
            "LCFA action argument renderer. The action has already been chosen by a recurrent policy.\n"
            "Return ONLY one JSON object containing the inputs for that action. Do not choose another action.\n"
            f"action={action}\n"
            f"goal={goal}\n"
            f"cognition={json.dumps(_compact_cognition(solution), ensure_ascii=False, default=str)}\n"
            f"recent={json.dumps(list(recent)[-3:], ensure_ascii=False, default=str)}\n"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
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
        return dict(parsed) if parsed is not None else None

    def choose(
        self,
        goal: str,
        solution: SolutionState,
        recent: Sequence[Mapping[str, Any]],
        step: int,
    ) -> Mapping[str, Any]:
        del step
        if self._hidden is None:
            self.reset(goal, solution)
        decision = self.decision()
        should_stop = decision.action == "stop" or decision.stop_probability >= self.stop_threshold
        if should_stop:
            return {
                "hypothesis": None,
                "action": None,
                "final": True,
                "controller": {
                    "action_confidence": decision.action_confidence,
                    "stop_probability": decision.stop_probability,
                    "value": decision.value,
                },
            }
        inputs = self._default_inputs(decision.action, goal, solution)
        if inputs is None:
            inputs = self._generate_inputs(decision.action, goal, solution, recent)
        if inputs is None:
            return {
                "hypothesis": None,
                "action": {"name": "repo.search", "inputs": {"query": goal}},
                "final": False,
                "controller": {"fallback_from": decision.action, "value": decision.value},
            }
        return {
            "hypothesis": None,
            "action": {"name": decision.action, "inputs": dict(inputs)},
            "final": False,
            "controller": {
                "action_confidence": decision.action_confidence,
                "stop_probability": decision.stop_probability,
                "value": decision.value,
            },
        }

    def observe(
        self,
        action: Mapping[str, Any] | None,
        observation: Mapping[str, Any],
        solution: SolutionState,
    ) -> None:
        self._forward_event({
            "kind": "transition",
            "action": dict(action) if isinstance(action, Mapping) else None,
            "observation": dict(observation),
            "cognition": _compact_cognition(solution),
        })


__all__ = [
    "DEFAULT_RWKV_MODEL",
    "RWKV_CONTROLLER_FORMAT",
    "RWKVControllerError",
    "RWKVRecurrentPolicy",
    "RecurrentDecision",
    "RecurrentPolicy",
]
