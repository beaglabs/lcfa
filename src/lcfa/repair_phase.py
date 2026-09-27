"""Explicit repair-phase policy for autonomous LCFA rollouts.

The learned controller remains responsible for ranking actions inside each phase,
but impossible or structurally premature actions are masked. This prevents the
common failure mode where a controller endlessly gathers evidence or verifies
before it has ever mutated the repository.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

DISCOVER = "discover"
GROUND = "ground"
VERIFY = "verify"
REPAIR = "repair"
DONE = "done"

EDIT_ACTIONS = {"repo.edit", "repo.replace"}
EVIDENCE_ACTIONS = {"repo.search", "repo.read", "docs.fetch", "process.exec"}
VERIFY_ACTIONS = {"test.run", "verify.run"}
INSPECTION_ACTIONS = {"git.status", "git.diff"}


@dataclass(frozen=True, slots=True)
class RepairPhaseState:
    phase: str = DISCOVER
    evidence_actions: int = 0
    reads: int = 0
    searches: int = 0
    mutations: int = 0
    verifications: int = 0
    verifier_passed: bool = False
    last_action: str | None = None

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def _action_name(action: Mapping[str, Any] | None) -> str:
    return str(action.get("name") or "") if isinstance(action, Mapping) else ""


def _walk_mappings(value: Any, *, depth: int = 0) -> tuple[Mapping[str, Any], ...]:
    if depth > 6:
        return ()
    out: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        out.append(value)
        for nested in value.values():
            out.extend(_walk_mappings(nested, depth=depth + 1))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value[:32]:
            out.extend(_walk_mappings(nested, depth=depth + 1))
    return tuple(out)


def verifier_outcome(
    action: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
) -> bool | None:
    """Resolve verifier/test success conservatively from nested tool output."""
    if _action_name(action) not in VERIFY_ACTIONS:
        return None
    mappings = _walk_mappings(observation if isinstance(observation, Mapping) else {})
    saw_explicit_success = False
    saw_exit_zero = False
    for item in mappings:
        for key in ("success", "passed", "ok", "resolved"):
            value = item.get(key)
            if isinstance(value, bool):
                if not value:
                    return False
                saw_explicit_success = True
        code = item.get("exit_code", item.get("returncode"))
        if isinstance(code, (int, float)):
            if int(code) != 0:
                return False
            saw_exit_zero = True
        for key in ("error", "exception"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return False
    if saw_explicit_success or saw_exit_zero:
        return True
    return None


def advance_repair_phase(
    state: RepairPhaseState,
    action: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None = None,
) -> RepairPhaseState:
    """Advance phase state after the action that actually executed."""
    name = _action_name(action)
    if not name:
        return state

    evidence = state.evidence_actions
    reads = state.reads
    searches = state.searches
    mutations = state.mutations
    verifications = state.verifications
    phase = state.phase
    passed = state.verifier_passed

    if name in EVIDENCE_ACTIONS:
        evidence += 1
        reads += int(name == "repo.read")
        searches += int(name == "repo.search")
        if phase == DISCOVER and (reads >= 1 and evidence >= 2 or evidence >= 3):
            phase = GROUND
        elif phase == REPAIR and (reads >= state.reads + 1 or evidence >= state.evidence_actions + 1):
            phase = GROUND
    elif name in EDIT_ACTIONS:
        mutations += 1
        passed = False
        phase = VERIFY
    elif name in VERIFY_ACTIONS:
        verifications += 1
        outcome = verifier_outcome(action, observation)
        if outcome is True:
            passed = True
            phase = DONE
        elif outcome is False:
            passed = False
            phase = REPAIR
        else:
            phase = VERIFY
    elif name == "stop":
        phase = DONE

    return RepairPhaseState(
        phase=phase,
        evidence_actions=evidence,
        reads=reads,
        searches=searches,
        mutations=mutations,
        verifications=verifications,
        verifier_passed=passed,
        last_action=name,
    )


def phase_action_policy(
    state: RepairPhaseState,
    action_names: Sequence[str],
    *,
    has_target: bool,
) -> tuple[tuple[str, ...], Mapping[str, float]]:
    """Return legal actions and phase-specific logit biases.

    The hard lock after repeated evidence collection is deliberate: once a
    grounded target exists, a rollout must attempt mutation instead of using
    repo.read/repo.search as a permanently safe fallback.
    """
    names = tuple(str(name) for name in action_names)
    available = set(names)
    biases: dict[str, float] = {}

    if state.phase == DONE:
        allowed = {"stop"}
    elif state.phase == VERIFY:
        allowed = VERIFY_ACTIONS | INSPECTION_ACTIONS
        biases.update({"verify.run": 4.0, "test.run": 3.5, "git.diff": 1.0})
    elif state.phase in {GROUND, REPAIR}:
        if has_target and state.evidence_actions >= 4:
            allowed = set(EDIT_ACTIONS)
        else:
            allowed = set(EVIDENCE_ACTIONS)
            if has_target:
                allowed |= EDIT_ACTIONS
        biases.update({"repo.edit": 4.0, "repo.replace": 4.0})
        if state.phase == GROUND:
            biases.update({"repo.read": -1.5, "repo.search": -2.0})
        else:
            biases.update({"repo.read": -0.5, "repo.search": -1.0})
    else:
        allowed = set(EVIDENCE_ACTIONS)
        if has_target and (state.reads >= 1 or state.evidence_actions >= 2):
            allowed |= EDIT_ACTIONS
            biases.update({"repo.edit": 1.5, "repo.replace": 1.5})

    resolved = tuple(name for name in names if name in allowed and name in available)
    if not resolved:
        # Never produce an empty mask: fall back to the vocabulary while still
        # suppressing premature stop at the caller when possible.
        resolved = names
    return resolved, biases


def stop_allowed(state: RepairPhaseState) -> bool:
    return state.phase == DONE and state.verifier_passed


__all__ = [
    "DISCOVER",
    "DONE",
    "EDIT_ACTIONS",
    "EVIDENCE_ACTIONS",
    "GROUND",
    "INSPECTION_ACTIONS",
    "REPAIR",
    "RepairPhaseState",
    "VERIFY",
    "VERIFY_ACTIONS",
    "advance_repair_phase",
    "phase_action_policy",
    "stop_allowed",
    "verifier_outcome",
]
