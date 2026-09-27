"""Semantic-pointer extension for the intent-grounded hybrid LCFA controller."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .hybrid_controller import HybridLatentRWKVPolicy, load_hybrid_policy as load_legacy_hybrid_policy
from .hybrid_latent import HYBRID_CONTROLLER_FORMAT, HybridLatentConfig
from .pairwise_pointer import (
    PAIRWISE_POINTER_ARCHITECTURE,
    PAIRWISE_SEMANTIC_POINTER_FORMAT,
    make_pairwise_semantic_pointer,
    pairwise_pointer_prior_logits,
    pairwise_semantic_pointer_decision,
)
from .protocol import SolutionState
from .repair_phase import (
    RepairPhaseState,
    advance_repair_phase,
    phase_action_policy,
    stop_allowed,
)
from .rwkv_controller import DEFAULT_RWKV_MODEL, RecurrentDecision
from .semantic_pointer import (
    LEGACY_SEMANTIC_POINTER_FORMAT,
    SEMANTIC_POINTER_FORMAT,
    candidates_from_cognition,
    encode_candidate_semantics,
    make_legacy_semantic_pointer,
    make_semantic_pointer,
    retrieval_prior_decision,
    semantic_pointer_decision,
)


class SemanticPointerHybridPolicy(HybridLatentRWKVPolicy):
    """Hybrid controller reranking actual visible semantic candidates."""

    def __init__(
        self,
        model_id: str = DEFAULT_RWKV_MODEL,
        *,
        semantic_pointer_weights_path: str | Path,
        pointer_format: str = SEMANTIC_POINTER_FORMAT,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("semantic pointer requires safetensors torch support") from exc

        self.pointer_format = str(pointer_format or LEGACY_SEMANTIC_POINTER_FORMAT)
        if self.pointer_format == LEGACY_SEMANTIC_POINTER_FORMAT:
            self.semantic_pointer = make_legacy_semantic_pointer(
                self._torch,
                hidden_size=self.hidden_size,
                latent_dim=self.hybrid_config.latent_dim,
                device=self.device,
            )
            architecture = "candidate-conditioned-semantic-v1"
        elif self.pointer_format == PAIRWISE_SEMANTIC_POINTER_FORMAT:
            self.semantic_pointer = make_pairwise_semantic_pointer(
                self._torch,
                hidden_size=self.hidden_size,
                latent_dim=self.hybrid_config.latent_dim,
                device=self.device,
            )
            architecture = PAIRWISE_POINTER_ARCHITECTURE
        else:
            self.semantic_pointer = make_semantic_pointer(
                self._torch,
                hidden_size=self.hidden_size,
                latent_dim=self.hybrid_config.latent_dim,
                device=self.device,
            )
            architecture = "retrieval-prior+rwkv-semantic-residual"
        state = load_file(str(semantic_pointer_weights_path), device=self.device)
        self.semantic_pointer.load_state_dict(state, strict=True)
        self.semantic_pointer.eval()
        for parameter in self.semantic_pointer.parameters():
            parameter.requires_grad_(False)

        self._semantic_solution: SolutionState | None = None
        self._candidate_embedding_cache: dict[str, Any] = {}
        self._last_pointer_prior_index: int | None = None
        self._last_pointer_prior_confidence: float | None = None
        self._last_pointer_residual_gate: float | None = None
        self._last_pointer_prior_strength: float | None = None
        self._last_pointer_residual_abs: float | None = None
        self._repair_phase_state = RepairPhaseState()
        self._last_phase_allowed_actions: tuple[str, ...] = ()
        self._last_phase_raw_action: str | None = None
        self._last_phase_selected_action: str | None = None
        semantic_formats = {SEMANTIC_POINTER_FORMAT, PAIRWISE_SEMANTIC_POINTER_FORMAT}
        self.metadata = {
            **self.metadata,
            "pointer_architecture": architecture,
            "pointer_format": self.pointer_format,
            "candidate_encoder": (
                "frozen-rwkv-last-hidden-mean"
                if self.pointer_format in semantic_formats
                else "signed-hash-legacy"
            ),
            "pointer_prior": (
                "deterministic-retrieval"
                if self.pointer_format in semantic_formats
                else None
            ),
            "repair_phase_policy": "discover->ground->mutate->verify->done",
            "premutation_verification_blocked": True,
            "evidence_loop_hard_lock": 4,
        }

    def _phase_constrained_base(self, base: RecurrentDecision) -> RecurrentDecision:
        solution = self._semantic_solution
        cognition = (
            self._compact_solution_cognition(solution)
            if solution is not None
            else {}
        )
        has_target = bool(cognition.get("candidate_paths"))
        allowed, biases = phase_action_policy(
            self._repair_phase_state,
            self.action_names,
            has_target=has_target,
        )
        self._last_phase_allowed_actions = allowed
        self._last_phase_raw_action = base.action

        if self._hidden is None:
            self._last_phase_selected_action = base.action
            return base

        with self._torch.inference_mode():
            raw_logits = self.heads["action"](self._hidden.float())[0].float()
            constrained = self._torch.full_like(raw_logits, float("-inf"))
            allowed_set = set(allowed)
            for index, name in enumerate(self.action_names):
                if name in allowed_set:
                    constrained[index] = raw_logits[index] + float(biases.get(name, 0.0))
            if bool(self._torch.isneginf(constrained).all().item()):
                constrained = raw_logits
            probs = self._torch.softmax(constrained, dim=-1)
            selected_index = int(self._torch.argmax(probs).item())
            selected_action = self.action_names[selected_index]
            selected_confidence = float(probs[selected_index].item())

        self._last_phase_selected_action = selected_action
        gated_stop = base.stop_probability if stop_allowed(self._repair_phase_state) else 0.0
        return RecurrentDecision(
            action=selected_action,
            action_confidence=selected_confidence,
            stop_probability=gated_stop,
            value=base.value,
            pointer_index=base.pointer_index,
            pointer_confidence=base.pointer_confidence,
        )

    def decision(self) -> RecurrentDecision:
        base = self._phase_constrained_base(super().decision())
        solution = self._semantic_solution
        if solution is None or self._hidden is None or self._latent is None:
            return base
        cognition = self._compact_solution_cognition(solution)
        candidates = candidates_from_cognition(base.action, cognition)
        if not candidates:
            return base

        candidate_embeddings = None
        if self.pointer_format in {SEMANTIC_POINTER_FORMAT, PAIRWISE_SEMANTIC_POINTER_FORMAT}:
            candidate_embeddings = encode_candidate_semantics(
                self._torch,
                self.model,
                self.tokenizer,
                candidates,
                device=self.device,
                cache=self._candidate_embedding_cache,
            )
            if self.pointer_format == PAIRWISE_SEMANTIC_POINTER_FORMAT:
                prior_logits = pairwise_pointer_prior_logits(
                    self._torch,
                    candidates,
                    device=self.device,
                )
                prior_probs = self._torch.softmax(prior_logits.float(), dim=-1)
                local_prior = int(self._torch.argmax(prior_probs).item())
                self._last_pointer_prior_index = candidates[local_prior].index
                self._last_pointer_prior_confidence = float(prior_probs[local_prior].item())
                strength = getattr(self.semantic_pointer, "prior_strength", None)
                if strength is not None:
                    self._last_pointer_prior_strength = float(
                        strength.detach().float().clamp(0.0, 2.0).item()
                    )
                with self._torch.inference_mode():
                    _final, _prior, residual = self.semantic_pointer.components(
                        self._hidden.float(),
                        self._latent,
                        candidates,
                        candidate_embeddings,
                    )
                    self._last_pointer_residual_abs = float(residual.abs().mean().item())
            else:
                prior_index, prior_confidence, _ = retrieval_prior_decision(
                    self._torch,
                    candidates,
                    device=self.device,
                )
                self._last_pointer_prior_index = prior_index
                self._last_pointer_prior_confidence = prior_confidence
                gate = getattr(self.semantic_pointer, "residual_gate", None)
                if gate is not None:
                    self._last_pointer_residual_gate = float(
                        self._torch.tanh(gate.detach().float()).item()
                    )

        with self._torch.inference_mode():
            if self.pointer_format == PAIRWISE_SEMANTIC_POINTER_FORMAT:
                pointer_index, pointer_confidence, _ = pairwise_semantic_pointer_decision(
                    self._torch,
                    self.semantic_pointer,
                    self._hidden.float(),
                    self._latent,
                    candidates,
                    candidate_embeddings=candidate_embeddings,
                )
            else:
                pointer_index, pointer_confidence, _ = semantic_pointer_decision(
                    self._torch,
                    self.semantic_pointer,
                    self._hidden.float(),
                    self._latent,
                    candidates,
                    candidate_embeddings=candidate_embeddings,
                )
        if pointer_index is None:
            return base
        return RecurrentDecision(
            action=base.action,
            action_confidence=base.action_confidence,
            stop_probability=base.stop_probability,
            value=base.value,
            pointer_index=pointer_index,
            pointer_confidence=pointer_confidence,
        )

    @staticmethod
    def _compact_solution_cognition(solution: SolutionState) -> Mapping[str, Any]:
        values = solution.values if isinstance(solution.values, Mapping) else {}
        cognition = values.get("cognition")
        return cognition if isinstance(cognition, Mapping) else {}

    def reset(self, goal: str, solution: SolutionState) -> None:
        self._candidate_embedding_cache = {}
        self._last_pointer_prior_index = None
        self._last_pointer_prior_confidence = None
        self._last_pointer_residual_gate = None
        self._last_pointer_prior_strength = None
        self._last_pointer_residual_abs = None
        self._repair_phase_state = RepairPhaseState()
        self._last_phase_allowed_actions = ()
        self._last_phase_raw_action = None
        self._last_phase_selected_action = None
        super().reset(goal, solution)

    def choose(self, goal: str, solution: SolutionState, recent: Any, step: int) -> Mapping[str, Any]:
        self._semantic_solution = solution
        try:
            result = super().choose(goal, solution, recent, step)
        finally:
            self._semantic_solution = None
        controller = result.get("controller") if isinstance(result, Mapping) else None
        if isinstance(controller, dict):
            controller["pointer_architecture"] = self.metadata["pointer_architecture"]
            controller["pointer_prior_index"] = self._last_pointer_prior_index
            controller["pointer_prior_confidence"] = self._last_pointer_prior_confidence
            controller["pointer_residual_gate"] = self._last_pointer_residual_gate
            controller["pointer_prior_strength"] = self._last_pointer_prior_strength
            controller["pointer_residual_abs"] = self._last_pointer_residual_abs
            controller["repair_phase"] = self._repair_phase_state.phase
            controller["repair_phase_state"] = dict(self._repair_phase_state.to_dict())
            controller["phase_allowed_actions"] = list(self._last_phase_allowed_actions)
            controller["phase_raw_action"] = self._last_phase_raw_action
            controller["phase_selected_action"] = self._last_phase_selected_action
            controller["phase_overrode_action"] = (
                self._last_phase_raw_action is not None
                and self._last_phase_selected_action is not None
                and self._last_phase_raw_action != self._last_phase_selected_action
            )
        return result

    def observe(
        self,
        action: Mapping[str, Any] | None,
        observation: Mapping[str, Any],
        solution: SolutionState,
    ) -> None:
        super().observe(action, observation, solution)
        self._repair_phase_state = advance_repair_phase(
            self._repair_phase_state,
            action,
            observation,
        )


def load_hybrid_policy(
    controller_dir: str | Path,
    *,
    model_id: str | None = None,
    device: str | None = "auto",
    dtype: str = "auto",
) -> HybridLatentRWKVPolicy:
    """Load semantic-pointer artifacts, falling back to legacy hybrid artifacts."""
    root = Path(controller_dir)
    manifest_path = root / "controller.json" if root.is_dir() else root
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("controller manifest must be a JSON object")
    semantic_weights = raw.get("semantic_pointer_weights")
    if not semantic_weights:
        return load_legacy_hybrid_policy(
            controller_dir,
            model_id=model_id,
            device=device,
            dtype=dtype,
        )
    if raw.get("controller_format") != HYBRID_CONTROLLER_FORMAT:
        raise ValueError(f"not a hybrid LCFA controller: {raw.get('controller_format')!r}")

    resolved_root = manifest_path.parent
    resolved_model = str(model_id or raw.get("model_id") or "")
    if not resolved_model:
        raise ValueError("controller manifest requires model_id")
    actions = tuple(raw.get("action_vocab") or ())
    if not actions:
        raise ValueError("controller manifest requires action_vocab")
    cfg_raw = raw.get("hybrid_config") if isinstance(raw.get("hybrid_config"), Mapping) else {}
    config = HybridLatentConfig(
        latent_dim=int(cfg_raw.get("latent_dim", 256)),
        slots=int(cfg_raw.get("slots", 9)),
        min_reasoning_steps=int(cfg_raw.get("min_reasoning_steps", 2)),
        max_reasoning_steps=int(cfg_raw.get("max_reasoning_steps", 6)),
        convergence_tolerance=float(cfg_raw.get("convergence_tolerance", 1e-3)),
        intent_anchor_strength=float(cfg_raw.get("intent_anchor_strength", 0.98)),
    )
    backbone = raw.get("backbone_weights")
    return SemanticPointerHybridPolicy(
        resolved_model,
        heads_path=resolved_root / str(raw.get("weights") or "heads.safetensors"),
        hybrid_weights_path=resolved_root / str(raw.get("hybrid_weights") or "hybrid.safetensors"),
        semantic_pointer_weights_path=resolved_root / str(semantic_weights),
        pointer_format=str(raw.get("pointer_format") or LEGACY_SEMANTIC_POINTER_FORMAT),
        backbone_weights_path=(resolved_root / str(backbone) if backbone else None),
        hybrid_config=config,
        device=device,
        dtype=dtype,
        action_names=actions,
        pointer_slots=int(raw.get("pointer_slots", 0) or 0),
        stop_threshold=float(raw.get("stop_threshold", 0.5) or 0.5),
        max_argument_tokens=int(raw.get("max_argument_tokens", 512) or 512),
    )


__all__ = ["SemanticPointerHybridPolicy", "load_hybrid_policy"]
