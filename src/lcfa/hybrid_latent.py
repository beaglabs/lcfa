"""Typed latent workspace for the LCFA + RWKV hybrid software agent.

The repository/semantic graph remains the source of truth.  This module owns the
small trainable working-memory state used to reason over evidence between tool
steps.  RWKV supplies temporal memory; the latent workspace supplies iterative
problem-state refinement; fixed control heads consume their fused state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence

from .recurrent_transitions import normalize_event

HYBRID_CONTROLLER_FORMAT = "lcfa.hybrid-latent-rwkv.v1"
HYBRID_LATENT_FORMAT = "lcfa.hybrid-latent-workspace.v1"

DEFAULT_SLOT_NAMES: tuple[str, ...] = (
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
    """Shape and recurrent-depth contract for the alpha latent workspace."""

    latent_dim: int = 256
    slots: int = 8
    min_reasoning_steps: int = 2
    max_reasoning_steps: int = 6
    convergence_tolerance: float = 1e-3

    def normalized(self) -> "HybridLatentConfig":
        latent_dim = max(16, int(self.latent_dim))
        slots = max(4, int(self.slots))
        minimum = max(1, int(self.min_reasoning_steps))
        maximum = max(minimum, int(self.max_reasoning_steps))
        return HybridLatentConfig(
            latent_dim=latent_dim,
            slots=slots,
            min_reasoning_steps=minimum,
            max_reasoning_steps=maximum,
            convergence_tolerance=max(0.0, float(self.convergence_tolerance)),
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
    """Turn raw tool/verifier output into compact evidence for latent updates.

    The raw world remains recorded by the semantic episode.  This projection is
    deliberately small: it captures result type, process/verifier status, useful
    failure text and edited/read paths without turning terminal transcripts into
    the reasoning state itself.
    """
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
    elif action_name in {"process.exec"}:
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
    """Cheap semantic supervision for the alpha repair-plan projection."""
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
    inputs = target_inputs if isinstance(target_inputs, Mapping) else {}
    path = str(inputs.get("path") or "") or None
    return {
        "format": HYBRID_LATENT_FORMAT,
        "goal": str(goal),
        "intended_action": str(action),
        "target_path": path,
        "candidate_paths": [str(item) for item in candidate_paths[:16]],
        "required_changes": ([f"apply {action} to {path}"] if path and action in {"repo.replace", "repo.edit"} else []),
        "invariants": ["preserve unrelated repository behavior", "satisfy the external verifier"],
        "expected_verifier_success": (
            None if value_target is None else max(0.0, min(1.0, float(value_target)))
        ),
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
    paths_raw = cognition.get("candidate_paths", ())
    paths = (
        tuple(str(item) for item in paths_raw if str(item))
        if isinstance(paths_raw, Sequence) and not isinstance(paths_raw, (str, bytes))
        else ()
    )
    locations_raw = cognition.get("candidate_locations", ())
    locations = (
        tuple(str(item) for item in locations_raw if str(item))
        if isinstance(locations_raw, Sequence) and not isinstance(locations_raw, (str, bytes))
        else ()
    )
    concepts_raw = cognition.get("active_concepts", ())
    concepts = (
        tuple(str(item) for item in concepts_raw if str(item))
        if isinstance(concepts_raw, Sequence) and not isinstance(concepts_raw, (str, bytes))
        else ()
    )
    questions_raw = cognition.get("open_questions", ())
    questions = (
        tuple(str(item) for item in questions_raw if str(item))
        if isinstance(questions_raw, Sequence) and not isinstance(questions_raw, (str, bytes))
        else ()
    )
    pointer = int(pointer_index or 0)
    target_path = paths[pointer % len(paths)] if paths else None
    probs = [float(item) for item in plan_probabilities[: len(PLAN_FIELDS)]]
    while len(probs) < len(PLAN_FIELDS):
        probs.append(0.0)
    values = dict(zip(PLAN_FIELDS, probs))
    return {
        "format": HYBRID_LATENT_FORMAT,
        "goal": str(goal),
        "target_path": target_path,
        "target_symbols": list(locations[:8]),
        "intended_action": str(action),
        "required_changes": (
            [f"apply {action} to {target_path}"]
            if target_path and action in {"repo.replace", "repo.edit"}
            else []
        ),
        "invariants": ["preserve unrelated repository behavior", "satisfy the external verifier"],
        "evidence_refs": list(concepts[:12]),
        "unresolved_questions": list(questions[:6]),
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
    """Dedicated semantic patch/action renderer prompt.

    The controller has already selected the action.  This renderer is only
    allowed to materialize exact action arguments from a reasoned repair plan
    and exact retrieved evidence.
    """
    return (
        "LCFA semantic patch renderer. The hybrid latent controller already chose the action.\n"
        "Materialize ONLY the exact JSON inputs for that action; do not choose another action.\n"
        "For repo.replace return path, old and new. For repo.edit return path and content.\n"
        "Use the target file and evidence literally; do not invent unrelated edits.\n"
        f"action={action}\n"
        f"goal={goal}\n"
        f"target_path={target_path or ''}\n"
        f"repair_plan={json.dumps(dict(repair_plan or {}), sort_keys=True, ensure_ascii=False, default=str)}\n"
        f"event={json.dumps(hybrid_event(event), sort_keys=True, ensure_ascii=False, default=str)}\n"
    )


def make_hybrid_core(
    torch: Any,
    *,
    hidden_size: int,
    config: HybridLatentConfig,
    device: str,
) -> Any:
    """Build the trainable typed latent workspace without importing torch globally."""
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
            self.slot_seed = nn.Parameter(torch.empty(self.slots, self.latent_dim))
            nn.init.normal_(self.slot_seed, mean=0.0, std=0.02)
            self.evidence_projection = nn.Linear(self.hidden_size, self.latent_dim)
            self.update = nn.GRUCell(self.latent_dim * 2, self.latent_dim)
            self.norm = nn.LayerNorm(self.latent_dim)
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_size + self.latent_dim, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.plan = nn.Linear(self.latent_dim, len(PLAN_FIELDS))

        def forward(self, hidden: Any, latent: Any | None = None) -> tuple[Any, Any, Any, int]:
            source = hidden.float()
            evidence = torch.tanh(self.evidence_projection(source))
            batch = int(source.shape[0])
            if latent is None:
                latent = self.slot_seed.unsqueeze(0).expand(batch, -1, -1)
                latent = latent + evidence.unsqueeze(1)

            depth = 0
            for index in range(self.max_reasoning_steps):
                pooled = latent.mean(dim=1)
                update_input = torch.cat([evidence, pooled], dim=-1)
                update_input = update_input.unsqueeze(1).expand(-1, self.slots, -1)
                previous = latent
                updated = self.update(
                    update_input.reshape(batch * self.slots, self.latent_dim * 2),
                    previous.reshape(batch * self.slots, self.latent_dim),
                ).reshape(batch, self.slots, self.latent_dim)
                latent = self.norm(updated + previous)
                depth = index + 1
                if depth >= self.min_reasoning_steps and self.convergence_tolerance > 0:
                    delta = float((latent - previous).detach().abs().mean().item())
                    if delta <= self.convergence_tolerance:
                        break

            pooled = latent.mean(dim=1)
            fused = source + self.fusion(torch.cat([source, pooled], dim=-1))
            plan_logits = self.plan(pooled)
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
