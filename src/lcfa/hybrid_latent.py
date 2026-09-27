"""Typed latent workspace for the LCFA + RWKV hybrid software agent.

The semantic repository graph remains the source of truth.  User intent is
compiled into an explicit contract and represented by a persistent anchored
latent slot; mutable reasoning slots refine hypotheses/evidence/repair state
around that anchor.  Concrete edits are compiled through grounded RepairIR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence

from .intent_ir import (
    compile_repair_ir,
    compile_task_intent,
    ground_task_intent,
    rectify_repair_ir,
    validate_repair_ir,
)
from .recurrent_transitions import normalize_event

HYBRID_CONTROLLER_FORMAT = "lcfa.hybrid-latent-rwkv.v2"
HYBRID_LATENT_FORMAT = "lcfa.hybrid-latent-workspace.v2"

DEFAULT_SLOT_NAMES: tuple[str, ...] = (
    "user_intent",
    "problem",
    "hypothesis_primary",
    "hypothesis_alternative",
    "evidence",
    "target",
    "repair_intent",
    "constraints",
    "verifier_expectation",
)

PLAN_FIELDS: tuple[str, ...] = (
    "repair_readiness",
    "evidence_need",
    "verifier_expectation",
    "termination_readiness",
)


@dataclass(frozen=True, slots=True)
class HybridLatentConfig:
    """Shape and recurrent-depth contract for the hybrid latent workspace."""

    latent_dim: int = 256
    slots: int = 9
    min_reasoning_steps: int = 2
    max_reasoning_steps: int = 6
    convergence_tolerance: float = 1e-3
    intent_anchor_strength: float = 0.98

    def normalized(self) -> "HybridLatentConfig":
        latent_dim = max(16, int(self.latent_dim))
        slots = max(len(DEFAULT_SLOT_NAMES), int(self.slots))
        minimum = max(1, int(self.min_reasoning_steps))
        maximum = max(minimum, int(self.max_reasoning_steps))
        return HybridLatentConfig(
            latent_dim=latent_dim,
            slots=slots,
            min_reasoning_steps=minimum,
            max_reasoning_steps=maximum,
            convergence_tolerance=max(0.0, float(self.convergence_tolerance)),
            intent_anchor_strength=max(0.0, min(1.0, float(self.intent_anchor_strength))),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self.normalized())


def latent_slot_names(count: int) -> tuple[str, ...]:
    resolved = max(0, int(count))
    if resolved <= len(DEFAULT_SLOT_NAMES):
        return DEFAULT_SLOT_NAMES[:resolved]
    extras = tuple(f"scratch_{index}" for index in range(resolved - len(DEFAULT_SLOT_NAMES)))
    return DEFAULT_SLOT_NAMES + extras


def _walk_mappings(value: Any, *, depth: int = 0) -> Sequence[Mapping[str, Any]]:
    if depth > 6:
        return ()
    out: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        out.append(value)
        for nested in value.values():
            out.extend(_walk_mappings(nested, depth=depth + 1))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value[:24]:
            out.extend(_walk_mappings(nested, depth=depth + 1))
    return tuple(out)


def _clip(value: Any, limit: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    half = max(1, (limit - 32) // 2)
    return text[:half] + "\n...<truncated>...\n" + text[-half:]


def structured_runtime_feedback(
    action: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Turn raw tool/verifier output into compact evidence for latent updates."""
    action_name = str(action.get("name") or "") if isinstance(action, Mapping) else ""
    raw = observation if isinstance(observation, Mapping) else {}
    mappings = _walk_mappings(raw)

    exit_code: int | None = None
    stdout = ""
    stderr = ""
    error = ""
    paths: list[str] = []
    for item in mappings:
        if exit_code is None:
            candidate = item.get("exit_code", item.get("returncode"))
            if isinstance(candidate, (int, float)):
                exit_code = int(candidate)
        if not stdout and isinstance(item.get("stdout"), str):
            stdout = _clip(item.get("stdout"))
        if not stderr and isinstance(item.get("stderr"), str):
            stderr = _clip(item.get("stderr"))
        if not error:
            for key in ("error", "exception", "message"):
                if isinstance(item.get(key), str) and item.get(key):
                    error = _clip(item.get(key), 2000)
                    break
        path = item.get("path")
        if isinstance(path, str) and path and path not in paths:
            paths.append(path)

    failed = bool(error) or (exit_code is not None and exit_code != 0)
    kind = "observation"
    if action_name in {"test.run", "verify.run"}:
        kind = "verifier_feedback"
    elif action_name == "process.exec":
        kind = "runtime_feedback"
    elif action_name in {"repo.edit", "repo.replace"}:
        kind = "edit_feedback"
    elif action_name in {"repo.read", "repo.search", "docs.fetch"}:
        kind = "evidence_feedback"

    feedback: dict[str, Any] = {
        "kind": kind,
        "action": action_name,
        "status": "failed" if failed else "ok",
    }
    if exit_code is not None:
        feedback["exit_code"] = exit_code
    if paths:
        feedback["paths"] = paths[:16]
    if stdout:
        feedback["stdout"] = stdout
    if stderr:
        feedback["stderr"] = stderr
    if error:
        feedback["error"] = error
    if len(feedback) == 3 and raw:
        rendered = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
        feedback["summary"] = _clip(rendered, 3000)
    return feedback


def hybrid_event(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize training and live events through the exact same hybrid schema."""
    normalized = normalize_event(event)
    if normalized.get("kind") != "transition":
        return normalized
    action = normalized.get("action") if isinstance(normalized.get("action"), Mapping) else None
    observation = (
        normalized.get("observation")
        if isinstance(normalized.get("observation"), Mapping)
        else {}
    )
    return {
        "kind": "transition",
        "action": action,
        "feedback": structured_runtime_feedback(action, observation),
    }


def supervised_plan_targets(
    action: str,
    *,
    stop_target: bool,
    value_target: float | None,
) -> tuple[float | None, ...]:
    name = str(action)
    repair = 1.0 if name in {"repo.replace", "repo.edit"} else 0.0
    evidence = 1.0 if name in {
        "repo.search", "repo.read", "docs.fetch", "process.exec", "test.run", "verify.run"
    } else 0.0
    verifier = None if value_target is None else max(0.0, min(1.0, float(value_target)))
    termination = 1.0 if bool(stop_target) else 0.0
    return repair, evidence, verifier, termination


def supervised_repair_plan(
    *,
    goal: str,
    action: str,
    target_inputs: Mapping[str, Any] | None,
    candidate_paths: Sequence[str] = (),
    value_target: float | None = None,
) -> Mapping[str, Any]:
    """Build the same intent/RepairIR contract used at runtime for supervision."""
    inputs = target_inputs if isinstance(target_inputs, Mapping) else {}
    target_path = str(inputs.get("path") or "") or None
    visible_paths = [str(item) for item in candidate_paths[:16] if str(item)]
    if target_path:
        visible_paths = [target_path, *[item for item in visible_paths if item != target_path]]
    contract = compile_task_intent(goal)
    grounded = ground_task_intent(
        contract,
        {"candidate_paths": visible_paths, "candidate_locations": (), "active_concepts": ()},
        pointer_index=0,
    )
    ir = compile_repair_ir(
        grounded,
        action=action,
        pointer_index=0,
        expected_verifier_success=value_target,
    )
    validation = validate_repair_ir(ir, grounded)
    if not validation.valid:
        ir = rectify_repair_ir(ir, grounded)
        validation = validate_repair_ir(ir, grounded)
    return {
        "format": HYBRID_LATENT_FORMAT,
        "task_intent": contract.to_dict(),
        "grounded_intent": grounded.to_dict(),
        "repair_ir": ir.to_dict(),
        "repair_ir_validation": validation.to_dict(),
        "repair_readiness": 1.0 if action in {"repo.replace", "repo.edit"} else 0.0,
        "expected_verifier_success": value_target,
    }


def repair_plan_from_state(
    *,
    goal: str,
    action: str,
    pointer_index: int | None,
    plan_probabilities: Sequence[float],
    cognition: Mapping[str, Any],
    slots: int,
    reasoning_depth: int,
) -> Mapping[str, Any]:
    """Compile mutable latent decisions around an immutable user intent contract."""
    probs = [float(item) for item in plan_probabilities[: len(PLAN_FIELDS)]]
    while len(probs) < len(PLAN_FIELDS):
        probs.append(0.0)
    values = dict(zip(PLAN_FIELDS, probs))
    contract = compile_task_intent(goal)
    grounded = ground_task_intent(contract, cognition, pointer_index=pointer_index)
    questions_raw = cognition.get("open_questions", ())
    questions = (
        tuple(str(item) for item in questions_raw if str(item))
        if isinstance(questions_raw, Sequence) and not isinstance(questions_raw, (str, bytes))
        else ()
    )
    ir = compile_repair_ir(
        grounded,
        action=action,
        pointer_index=pointer_index,
        expected_verifier_success=values["verifier_expectation"],
        unresolved_questions=questions[:6],
        reasoning_depth=reasoning_depth,
    )
    validation = validate_repair_ir(ir, grounded)
    rectified = False
    if not validation.valid:
        ir = rectify_repair_ir(ir, grounded)
        validation = validate_repair_ir(ir, grounded)
        rectified = True
    return {
        "format": HYBRID_LATENT_FORMAT,
        "task_intent": contract.to_dict(),
        "grounded_intent": grounded.to_dict(),
        "repair_ir": ir.to_dict(),
        "repair_ir_validation": validation.to_dict(),
        "repair_ir_rectified": rectified,
        "repair_readiness": values["repair_readiness"],
        "needs_more_evidence": values["evidence_need"],
        "expected_verifier_success": values["verifier_expectation"],
        "termination_readiness": values["termination_readiness"],
        "latent_slots": list(latent_slot_names(slots)),
        "reasoning_depth": int(reasoning_depth),
    }


def repair_prompt(
    action: str,
    goal: str,
    event: Mapping[str, Any],
    *,
    target_path: str | None,
    repair_plan: Mapping[str, Any] | None,
) -> str:
    """Compile validated semantic intent into exact action arguments.

    Semantic planning remains free-form inside latent cognition / RepairIR.  The
    renderer is deliberately narrow: it must obey the immutable user contract,
    grounded repository target, and validated RepairIR.
    """
    context = dict(repair_plan or {})
    task_intent = context.get("task_intent", {})
    grounded = context.get("grounded_intent", {})
    repair_ir = context.get("repair_ir", {})
    validation = context.get("repair_ir_validation", {})
    return (
        "LCFA constrained semantic patch compiler.\n"
        "The user intent contract is the immutable source of truth.\n"
        "The controller has already chosen the action and RepairIR target.\n"
        "Return ONLY one JSON object containing exact inputs for that action.\n"
        "Do not choose another action, target another file, or invent unrelated changes.\n"
        "For repo.replace return path, old and new. For repo.edit return path and content.\n"
        "Preserve every invariant and prohibited outcome in the intent/RepairIR.\n"
        f"action={action}\n"
        f"goal={goal}\n"
        f"target_path={target_path or ''}\n"
        f"task_intent={json.dumps(task_intent, sort_keys=True, ensure_ascii=False, default=str)}\n"
        f"grounded_intent={json.dumps(grounded, sort_keys=True, ensure_ascii=False, default=str)}\n"
        f"repair_ir={json.dumps(repair_ir, sort_keys=True, ensure_ascii=False, default=str)}\n"
        f"repair_ir_validation={json.dumps(validation, sort_keys=True, ensure_ascii=False, default=str)}\n"
        f"event={json.dumps(hybrid_event(event), sort_keys=True, ensure_ascii=False, default=str)}\n"
    )


def make_hybrid_core(
    torch: Any,
    *,
    hidden_size: int,
    config: HybridLatentConfig,
    device: str,
) -> Any:
    """Build the trainable latent workspace with a persistent z_intent slot."""
    cfg = config.normalized()
    nn = torch.nn

    class HybridLatentCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.latent_dim = int(cfg.latent_dim)
            self.slots = int(cfg.slots)
            self.min_reasoning_steps = int(cfg.min_reasoning_steps)
            self.max_reasoning_steps = int(cfg.max_reasoning_steps)
            self.convergence_tolerance = float(cfg.convergence_tolerance)
            self.intent_anchor_strength = float(cfg.intent_anchor_strength)
            self.slot_seed = nn.Parameter(torch.empty(self.slots, self.latent_dim))
            nn.init.normal_(self.slot_seed, mean=0.0, std=0.02)
            self.intent_projection = nn.Linear(self.hidden_size, self.latent_dim)
            self.evidence_projection = nn.Linear(self.hidden_size, self.latent_dim)
            self.update = nn.GRUCell(self.latent_dim * 3, self.latent_dim)
            self.norm = nn.LayerNorm(self.latent_dim)
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_size + self.latent_dim * 2, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            # Baseline-preserving residual initialization: newly added latent
            # cognition must not corrupt the feature geometry expected by a
            # pretrained RWKV controller.  At initialization fused == source;
            # the residual is learned only when supervised evidence supports it.
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)
            self.plan = nn.Linear(self.latent_dim * 2, len(PLAN_FIELDS))

        def forward(self, hidden: Any, latent: Any | None = None) -> tuple[Any, Any, Any, int]:
            source = hidden.float()
            evidence = torch.tanh(self.evidence_projection(source))
            batch = int(source.shape[0])
            if latent is None:
                intent_anchor = torch.tanh(self.intent_projection(source))
                latent = self.slot_seed.unsqueeze(0).expand(batch, -1, -1).clone()
                latent = latent + evidence.unsqueeze(1)
                latent[:, 0, :] = intent_anchor
            else:
                # Slot zero is the persistent user-intent anchor.  Tool/verifier
                # events may refine its edge, but cannot freely overwrite it.
                intent_anchor = latent[:, 0, :]

            depth = 0
            for index in range(self.max_reasoning_steps):
                reasoning = latent[:, 1:, :] if self.slots > 1 else latent
                pooled = reasoning.mean(dim=1)
                update_input = torch.cat([evidence, pooled, intent_anchor], dim=-1)
                update_input = update_input.unsqueeze(1).expand(-1, self.slots, -1)
                previous = latent
                updated = self.update(
                    update_input.reshape(batch * self.slots, self.latent_dim * 3),
                    previous.reshape(batch * self.slots, self.latent_dim),
                ).reshape(batch, self.slots, self.latent_dim)
                latent = self.norm(updated + previous)
                proposed_intent = latent[:, 0, :]
                strength = self.intent_anchor_strength
                latent[:, 0, :] = self.norm(
                    strength * intent_anchor + (1.0 - strength) * proposed_intent
                )
                depth = index + 1
                if depth >= self.min_reasoning_steps and self.convergence_tolerance > 0:
                    delta = float(
                        (latent[:, 1:, :] - previous[:, 1:, :])
                        .detach()
                        .abs()
                        .mean()
                        .item()
                    )
                    if delta <= self.convergence_tolerance:
                        break

            intent_anchor = latent[:, 0, :]
            reasoning = latent[:, 1:, :] if self.slots > 1 else latent
            pooled = reasoning.mean(dim=1)
            fused = source + self.fusion(torch.cat([source, intent_anchor, pooled], dim=-1))
            plan_logits = self.plan(torch.cat([intent_anchor, pooled], dim=-1))
            return fused, latent, plan_logits, depth

    return HybridLatentCore().to(device)


__all__ = [
    "DEFAULT_SLOT_NAMES",
    "HYBRID_CONTROLLER_FORMAT",
    "HYBRID_LATENT_FORMAT",
    "HybridLatentConfig",
    "PLAN_FIELDS",
    "hybrid_event",
    "latent_slot_names",
    "make_hybrid_core",
    "repair_plan_from_state",
    "repair_prompt",
    "structured_runtime_feedback",
    "supervised_plan_targets",
    "supervised_repair_plan",
]
