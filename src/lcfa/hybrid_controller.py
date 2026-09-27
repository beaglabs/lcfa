"""Live LCFA policy combining typed latent cognition with RWKV trajectory memory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hybrid_latent import (
    HYBRID_CONTROLLER_FORMAT,
    HybridLatentConfig,
    hybrid_event,
    make_hybrid_core,
    repair_plan_from_state,
    repair_prompt,
    structured_runtime_feedback,
)
from .protocol import SolutionState
from .recurrent_transitions import ACTION_VOCAB
from .rwkv_controller import (
    DEFAULT_RWKV_MODEL,
    RWKVControllerError,
    RecurrentDecision,
    _compact_cognition,
    _json_object,
    _valid_generated_inputs,
)
from .torch_runtime import resolve_device, resolve_dtype


class SemanticPatchDecoder:
    """Dedicated renderer from a reasoned repair plan to exact action arguments.

    The alpha shares the RWKV language head to avoid adding a second large model,
    but patch rendering has its own prompt/contract and no longer doubles as the
    control policy.  The interface can later be swapped for a smaller decoder.
    """

    def __init__(self, *, model: Any, tokenizer: Any, device: str, max_tokens: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max(64, int(max_tokens))

    def render(
        self,
        *,
        action: str,
        goal: str,
        event: Mapping[str, Any],
        target_path: str | None,
        repair_plan: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by controller init
            raise RWKVControllerError("hybrid patch decoder requires torch") from exc

        prompt = repair_prompt(
            action,
            goal,
            event,
            target_path=target_path,
            repair_plan=repair_plan,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_tokens,
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
        if not _valid_generated_inputs(action, inputs):
            return None
        return inputs


class HybridLatentRWKVPolicy:
    """Hybrid software-agent policy: latent workspace + RWKV recurrent memory."""

    def __init__(
        self,
        model_id: str = DEFAULT_RWKV_MODEL,
        *,
        heads_path: str | Path,
        hybrid_weights_path: str | Path,
        backbone_weights_path: str | Path | None = None,
        hybrid_config: HybridLatentConfig | None = None,
        device: str | None = "auto",
        dtype: str = "auto",
        action_names: Sequence[str] = ACTION_VOCAB,
        pointer_slots: int = 0,
        stop_threshold: float = 0.5,
        max_argument_tokens: int = 512,
    ) -> None:
        try:
            import torch
            from safetensors.torch import load_file, load_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RWKVControllerError(
                "hybrid LCFA controller requires `pip install -e '.[rwkv]'`"
            ) from exc

        self._torch = torch
        self.model_id = str(model_id)
        self.action_names = tuple(str(item) for item in action_names)
        self.pointer_slots = max(0, int(pointer_slots))
        self.stop_threshold = float(stop_threshold)
        self.hybrid_config = (hybrid_config or HybridLatentConfig()).normalized()
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
        self.hybrid = make_hybrid_core(
            torch,
            hidden_size=hidden_size,
            config=self.hybrid_config,
            device=self.device,
        )
        hybrid_state = load_file(str(hybrid_weights_path), device=self.device)
        self.hybrid.load_state_dict(hybrid_state, strict=True)
        self.hybrid.eval()
        for parameter in self.hybrid.parameters():
            parameter.requires_grad_(False)

        modules: dict[str, Any] = {
            "action": torch.nn.Linear(hidden_size, len(self.action_names)),
            "stop": torch.nn.Linear(hidden_size, 1),
            "value": torch.nn.Linear(hidden_size, 1),
        }
        if self.pointer_slots > 0:
            modules["pointer"] = torch.nn.Linear(hidden_size, self.pointer_slots)
        self.heads = torch.nn.ModuleDict(modules).to(self.device)
        head_state = load_file(str(heads_path), device=self.device)
        self.heads.load_state_dict(head_state, strict=True)
        self.heads.eval()

        self.patch_decoder = SemanticPatchDecoder(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_tokens=max_argument_tokens,
        )
        self._rwkv_state: Any = None
        self._latent: Any = None
        self._hidden: Any = None
        self._raw_hidden: Any = None
        self._plan_logits: Any = None
        self._reasoning_depth = 0
        self._goal = ""
        self._event_history: list[Mapping[str, Any]] = []
        self.metadata = {
            "type": "hybrid-latent-rwkv-policy",
            "format": HYBRID_CONTROLLER_FORMAT,
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "actions": list(self.action_names),
            "pointer_slots": self.pointer_slots,
            "stop_threshold": self.stop_threshold,
            "loader": "transformers-remote-code",
            "device": self.device,
            "dtype": self.dtype_name,
            "event_schema": "goal + retrieval + structured tool/verifier feedback",
            "hybrid_config": dict(self.hybrid_config.to_dict()),
            "backbone_weights": str(backbone_weights_path) if backbone_weights_path else None,
            "patch_decoder": "shared-rwkv-language-head-dedicated-repair-contract",
        }

    def _forward_event(self, payload: Mapping[str, Any]) -> None:
        torch = self._torch
        event = hybrid_event(payload)
        self._event_history.append(event)
        self._event_history = self._event_history[-12:]
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
        if self._rwkv_state is not None:
            kwargs["state"] = self._rwkv_state
        with torch.inference_mode():
            outputs = self.model(**kwargs)
            hidden_states = getattr(outputs, "hidden_states", None)
            state = getattr(outputs, "state", None)
            if not hidden_states or state is None:
                raise RWKVControllerError(
                    "RWKV forward must return hidden_states and recurrent state"
                )
            raw_hidden = hidden_states[-1][:, -1, :].detach().float()
            fused, latent, plan_logits, depth = self.hybrid(raw_hidden, self._latent)
        self._rwkv_state = state
        self._raw_hidden = raw_hidden
        self._hidden = fused.detach()
        self._latent = latent.detach()
        self._plan_logits = plan_logits.detach()
        self._reasoning_depth = int(depth)

    def reset(self, goal: str, solution: SolutionState) -> None:
        cognition = _compact_cognition(solution)
        self._rwkv_state = None
        self._latent = None
        self._hidden = None
        self._raw_hidden = None
        self._plan_logits = None
        self._reasoning_depth = 0
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
            raise RWKVControllerError("hybrid controller has not been reset")
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

    def _repair_plan(
        self,
        *,
        action: str,
        pointer: int | None,
        solution: SolutionState,
    ) -> Mapping[str, Any]:
        if self._plan_logits is None:
            probabilities = [0.0, 1.0, 0.0, 0.0]
        else:
            with self._torch.inference_mode():
                probabilities = self._torch.sigmoid(self._plan_logits.float())[0].tolist()
        return repair_plan_from_state(
            goal=self._goal,
            action=action,
            pointer_index=pointer,
            plan_probabilities=probabilities,
            cognition=_compact_cognition(solution),
            slots=self.hybrid_config.slots,
            reasoning_depth=self._reasoning_depth,
        )

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
        plan = self._repair_plan(
            action=decision.action,
            pointer=decision.pointer_index,
            solution=solution,
        )
        controller = {
            "architecture": "hybrid-latent-rwkv",
            "action_confidence": decision.action_confidence,
            "stop_probability": decision.stop_probability,
            "value": decision.value,
            "pointer_index": decision.pointer_index,
            "pointer_confidence": decision.pointer_confidence,
            "reasoning_depth": self._reasoning_depth,
            "repair_plan": plan,
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
            target_path = self._candidate_path(solution, decision.pointer_index)
            event = self._event_history[-1] if self._event_history else {
                "kind": "goal", "goal": goal
            }
            inputs = self.patch_decoder.render(
                action=decision.action,
                goal=goal,
                event=event,
                target_path=target_path,
                repair_plan=plan,
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
                    "fallback_reason": "semantic patch renderer returned invalid inputs",
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
        feedback = structured_runtime_feedback(action, observation)
        self._forward_event({
            "kind": "transition",
            "action": dict(action) if isinstance(action, Mapping) else None,
            "observation": feedback,
        })


def load_hybrid_policy(
    controller_dir: str | Path,
    *,
    model_id: str | None = None,
    device: str | None = "auto",
    dtype: str = "auto",
) -> HybridLatentRWKVPolicy:
    root = Path(controller_dir)
    manifest_path = root / "controller.json" if root.is_dir() else root
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("controller manifest must be a JSON object")
    if raw.get("controller_format") != HYBRID_CONTROLLER_FORMAT:
        raise ValueError(
            f"not a hybrid LCFA controller: {raw.get('controller_format')!r}"
        )
    resolved_root = manifest_path.parent
    resolved_model = str(model_id or raw.get("model_id") or "")
    if not resolved_model:
        raise ValueError("controller manifest requires model_id")
    action_vocab = tuple(raw.get("action_vocab") or ())
    if not action_vocab:
        raise ValueError("controller manifest requires action_vocab")
    config_raw = raw.get("hybrid_config")
    config_map = config_raw if isinstance(config_raw, Mapping) else {}
    config = HybridLatentConfig(
        latent_dim=int(config_map.get("latent_dim", 256)),
        slots=int(config_map.get("slots", 8)),
        min_reasoning_steps=int(config_map.get("min_reasoning_steps", 2)),
        max_reasoning_steps=int(config_map.get("max_reasoning_steps", 6)),
        convergence_tolerance=float(config_map.get("convergence_tolerance", 1e-3)),
    )
    backbone_weights = raw.get("backbone_weights")
    return HybridLatentRWKVPolicy(
        resolved_model,
        heads_path=resolved_root / str(raw.get("weights") or "heads.safetensors"),
        hybrid_weights_path=resolved_root / str(
            raw.get("hybrid_weights") or "hybrid.safetensors"
        ),
        backbone_weights_path=(
            resolved_root / str(backbone_weights) if backbone_weights else None
        ),
        hybrid_config=config,
        device=device,
        dtype=dtype,
        action_names=action_vocab,
        pointer_slots=int(raw.get("pointer_slots", 0) or 0),
        stop_threshold=float(raw.get("stop_threshold", 0.5) or 0.5),
        max_argument_tokens=int(raw.get("max_argument_tokens", 512) or 512),
    )


__all__ = [
    "HybridLatentRWKVPolicy",
    "SemanticPatchDecoder",
    "load_hybrid_policy",
]
