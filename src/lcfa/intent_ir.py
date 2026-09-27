"""Intent contracts and grounded RepairIR for LCFA software repair.

Natural-language requests are preserved as an immutable task contract, grounded
against semantic repository entities, then compiled into a typed RepairIR before
any edit is rendered.  This module is intentionally model-agnostic: learned
latent cognition proposes decisions, while these contracts keep the user request
and repository-visible constraints explicit and auditable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any, Mapping, Sequence

TASK_INTENT_FORMAT = "lcfa.task-intent.v1"
GROUNDED_INTENT_FORMAT = "lcfa.grounded-task-intent.v1"
REPAIR_IR_FORMAT = "lcfa.repair-ir.v1"

EDIT_ACTIONS = {"repo.replace", "repo.edit"}
_OPERATION_BY_ACTION = {
    "repo.replace": "replace_span",
    "repo.edit": "replace_file_content",
    "repo.read": "inspect_entity",
    "repo.search": "retrieve_evidence",
    "test.run": "run_targeted_tests",
    "verify.run": "run_verifier",
    "git.status": "inspect_worktree",
    "git.diff": "inspect_patch",
    "process.exec": "run_process",
    "docs.fetch": "retrieve_docs",
    "stop": "terminate",
}


def _clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _sequence(value: Any, *, limit: int = 32) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(_clean(item) for item in value[:limit] if _clean(item))


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in values if item))


@dataclass(frozen=True, slots=True)
class TaskIntentContract:
    request: str
    requested_outcomes: tuple[str, ...]
    preserve_constraints: tuple[str, ...]
    prohibited_outcomes: tuple[str, ...]
    acceptance_conditions: tuple[str, ...]
    unresolved_references: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ("user_request",)
    schema_version: str = TASK_INTENT_FORMAT

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroundedIntentTarget:
    entity_id: str
    kind: str
    path: str | None = None
    confidence: float = 0.0
    source: str = "semantic_retrieval"

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroundedTaskIntent:
    contract: TaskIntentContract
    targets: tuple[GroundedIntentTarget, ...]
    candidate_paths: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    unresolved_references: tuple[str, ...]
    schema_version: str = GROUNDED_INTENT_FORMAT

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract.to_dict(),
            "targets": [item.to_dict() for item in self.targets],
            "candidate_paths": list(self.candidate_paths),
            "evidence_refs": list(self.evidence_refs),
            "unresolved_references": list(self.unresolved_references),
        }


@dataclass(frozen=True, slots=True)
class RepairOperation:
    kind: str
    target_entity: str | None = None
    target_path: str | None = None
    parameters: Mapping[str, Any] | None = None

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "kind": self.kind,
            "target_entity": self.target_entity,
            "target_path": self.target_path,
            "parameters": dict(self.parameters or {}),
        }


@dataclass(frozen=True, slots=True)
class RepairIR:
    user_request: str
    intended_action: str
    semantic_draft: str
    target_path: str | None
    target_entities: tuple[str, ...]
    operations: tuple[RepairOperation, ...]
    requested_outcomes: tuple[str, ...]
    invariants: tuple[str, ...]
    prohibited_outcomes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    expected_verifier_success: float | None
    reasoning_depth: int = 0
    schema_version: str = REPAIR_IR_FORMAT

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "user_request": self.user_request,
            "intended_action": self.intended_action,
            "semantic_draft": self.semantic_draft,
            "target_path": self.target_path,
            "target_entities": list(self.target_entities),
            "operations": [item.to_dict() for item in self.operations],
            "requested_outcomes": list(self.requested_outcomes),
            "invariants": list(self.invariants),
            "prohibited_outcomes": list(self.prohibited_outcomes),
            "evidence_refs": list(self.evidence_refs),
            "unresolved_questions": list(self.unresolved_questions),
            "expected_verifier_success": self.expected_verifier_success,
            "reasoning_depth": self.reasoning_depth,
        }


@dataclass(frozen=True, slots=True)
class RepairIRValidation:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


_CONSTRAINT_PREFIXES = (
    "keep ", "preserve ", "without ", "while keeping ", "while preserving ",
)
_PROHIBITED_PREFIXES = (
    "do not ", "don't ", "must not ", "never ", "avoid ",
)


def _intent_clauses(goal: str) -> tuple[str, ...]:
    text = _clean(goal)
    if not text:
        return ()
    parts = re.split(r"(?:\n+|(?<=[.!?;])\s+)", text)
    clauses = tuple(_clean(item).rstrip(".;") for item in parts if _clean(item))
    return clauses or (text,)


def compile_task_intent(goal: str) -> TaskIntentContract:
    """Compile user language into a conservative intent contract.

    The compiler intentionally preserves wording instead of trying to invent a
    more specific requirement than the user supplied.  Ambiguity is carried
    forward for repository grounding / latent reasoning rather than silently
    resolved here.
    """
    request = _clean(goal)
    clauses = _intent_clauses(request)
    requested: list[str] = []
    preserve: list[str] = []
    prohibited: list[str] = []
    for clause in clauses:
        lowered = clause.lower()
        if lowered.startswith(_PROHIBITED_PREFIXES):
            prohibited.append(clause)
        elif lowered.startswith(_CONSTRAINT_PREFIXES):
            preserve.append(clause)
        else:
            requested.append(clause)
    if request and not requested:
        requested.append(request)
    acceptance = tuple(f"satisfy user outcome: {item}" for item in requested)
    return TaskIntentContract(
        request=request,
        requested_outcomes=_dedupe(requested),
        preserve_constraints=_dedupe(preserve),
        prohibited_outcomes=_dedupe(prohibited),
        acceptance_conditions=acceptance,
    )


def _entity_kind(entity_id: str) -> str:
    prefix = entity_id.split("://", 1)[0] if "://" in entity_id else "concept"
    return prefix or "concept"


def ground_task_intent(
    contract: TaskIntentContract,
    cognition: Mapping[str, Any] | None,
    *,
    pointer_index: int | None = None,
) -> GroundedTaskIntent:
    """Ground a task contract only against semantic entities visible now."""
    state = cognition if isinstance(cognition, Mapping) else {}
    paths = _sequence(state.get("candidate_paths"), limit=16)
    locations = _sequence(state.get("candidate_locations"), limit=16)
    concepts = _sequence(state.get("active_concepts"), limit=24)
    pointer = int(pointer_index or 0)
    selected_path = paths[pointer % len(paths)] if paths else None

    targets: list[GroundedIntentTarget] = []
    for index, entity_id in enumerate(locations):
        targets.append(GroundedIntentTarget(
            entity_id=entity_id,
            kind=_entity_kind(entity_id),
            path=selected_path if index == pointer % max(1, len(locations)) else None,
            confidence=0.95 if index == pointer % max(1, len(locations)) else 0.70,
        ))
    if not targets and selected_path:
        targets.append(GroundedIntentTarget(
            entity_id=f"file:{selected_path}",
            kind="file",
            path=selected_path,
            confidence=0.65,
            source="retrieval_path",
        ))

    unresolved = contract.unresolved_references
    if not targets and contract.request:
        unresolved = _dedupe((*unresolved, "repository target not yet grounded"))
    return GroundedTaskIntent(
        contract=contract,
        targets=tuple(targets),
        candidate_paths=paths,
        evidence_refs=_dedupe((*locations, *concepts)),
        unresolved_references=unresolved,
    )


def _semantic_draft(
    contract: TaskIntentContract,
    *,
    action: str,
    target_path: str | None,
    target_entities: Sequence[str],
) -> str:
    outcomes = "; ".join(contract.requested_outcomes) or contract.request
    target = target_entities[0] if target_entities else target_path or "the grounded repository target"
    preserve = "; ".join(contract.preserve_constraints)
    draft = f"Satisfy: {outcomes}. Apply {action} to {target}."
    if preserve:
        draft += f" Preserve: {preserve}."
    return draft


def compile_repair_ir(
    grounded: GroundedTaskIntent,
    *,
    action: str,
    pointer_index: int | None = None,
    expected_verifier_success: float | None = None,
    unresolved_questions: Sequence[str] = (),
    reasoning_depth: int = 0,
) -> RepairIR:
    """Compile grounded intent + latent control decision into typed RepairIR."""
    pointer = int(pointer_index or 0)
    path = (
        grounded.candidate_paths[pointer % len(grounded.candidate_paths)]
        if grounded.candidate_paths else None
    )
    target_entities = tuple(item.entity_id for item in grounded.targets[:8])
    target_entity = target_entities[pointer % len(target_entities)] if target_entities else None
    operation = RepairOperation(
        kind=_OPERATION_BY_ACTION.get(str(action), "unknown"),
        target_entity=target_entity,
        target_path=path,
        parameters={"action": str(action)},
    )
    invariants = _dedupe((
        *grounded.contract.preserve_constraints,
        "preserve unrelated repository behavior",
        "satisfy the external verifier",
    ))
    return RepairIR(
        user_request=grounded.contract.request,
        intended_action=str(action),
        semantic_draft=_semantic_draft(
            grounded.contract,
            action=str(action),
            target_path=path,
            target_entities=target_entities,
        ),
        target_path=path,
        target_entities=target_entities,
        operations=(operation,),
        requested_outcomes=grounded.contract.requested_outcomes,
        invariants=invariants,
        prohibited_outcomes=grounded.contract.prohibited_outcomes,
        evidence_refs=grounded.evidence_refs[:16],
        unresolved_questions=_dedupe((*grounded.unresolved_references, *tuple(unresolved_questions))),
        expected_verifier_success=(
            None if expected_verifier_success is None
            else max(0.0, min(1.0, float(expected_verifier_success)))
        ),
        reasoning_depth=max(0, int(reasoning_depth)),
    )


def validate_repair_ir(ir: RepairIR, grounded: GroundedTaskIntent) -> RepairIRValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if not ir.requested_outcomes:
        errors.append("repair IR has no user-requested outcomes")
    if not ir.operations:
        errors.append("repair IR has no operation")
    expected_kind = _OPERATION_BY_ACTION.get(ir.intended_action)
    if expected_kind and any(op.kind != expected_kind for op in ir.operations):
        errors.append("repair IR operation kind does not match intended action")
    if ir.intended_action in EDIT_ACTIONS:
        if not ir.target_path:
            errors.append("edit repair IR is not grounded to a target path")
        elif grounded.candidate_paths and ir.target_path not in grounded.candidate_paths:
            errors.append("edit target path was not visible in grounded retrieval")
    visible_entities = {item.entity_id for item in grounded.targets}
    invisible = [item for item in ir.target_entities if visible_entities and item not in visible_entities]
    if invisible:
        errors.append("repair IR references repository entities not visible during grounding")
    if not ir.target_entities:
        warnings.append("repair IR has path grounding but no semantic entity grounding")
    if ir.unresolved_questions:
        warnings.append("repair IR still contains unresolved intent questions")
    return RepairIRValidation(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))


def rectify_repair_ir(ir: RepairIR, grounded: GroundedTaskIntent) -> RepairIR:
    """Deterministically rectify visibility/schema errors without inventing facts."""
    path = ir.target_path
    if grounded.candidate_paths and path not in grounded.candidate_paths:
        path = grounded.candidate_paths[0]
    visible_entities = tuple(item.entity_id for item in grounded.targets)
    entities = tuple(item for item in ir.target_entities if item in set(visible_entities))
    if not entities:
        entities = visible_entities[:8]
    operation_kind = _OPERATION_BY_ACTION.get(ir.intended_action, "unknown")
    operation = RepairOperation(
        kind=operation_kind,
        target_entity=entities[0] if entities else None,
        target_path=path,
        parameters={"action": ir.intended_action},
    )
    return replace(
        ir,
        target_path=path,
        target_entities=entities,
        operations=(operation,),
    )


def validate_rendered_inputs(
    ir: RepairIR,
    action: str,
    inputs: Mapping[str, Any] | None,
    *,
    visible_paths: Sequence[str] = (),
) -> RepairIRValidation:
    errors: list[str] = []
    if not isinstance(inputs, Mapping):
        return RepairIRValidation(False, ("renderer did not return a mapping",), ())
    path = inputs.get("path")
    if action in EDIT_ACTIONS:
        if not isinstance(path, str) or not path:
            errors.append("rendered edit has no path")
        if ir.target_path and path != ir.target_path:
            errors.append("rendered path diverges from RepairIR target")
        if visible_paths and isinstance(path, str) and path not in set(visible_paths):
            errors.append("rendered path was not visible during grounding")
    if action == "repo.replace":
        if not isinstance(inputs.get("old"), str) or not inputs.get("old"):
            errors.append("repo.replace requires non-empty old text")
        if not isinstance(inputs.get("new"), str):
            errors.append("repo.replace requires string new text")
    if action == "repo.edit" and not isinstance(inputs.get("content"), str):
        errors.append("repo.edit requires string content")
    return RepairIRValidation(valid=not errors, errors=tuple(errors), warnings=())


__all__ = [
    "EDIT_ACTIONS",
    "GROUNDED_INTENT_FORMAT",
    "GroundedIntentTarget",
    "GroundedTaskIntent",
    "REPAIR_IR_FORMAT",
    "RepairIR",
    "RepairIROperation",
    "RepairIRValidation",
    "TASK_INTENT_FORMAT",
    "TaskIntentContract",
    "compile_repair_ir",
    "compile_task_intent",
    "ground_task_intent",
    "rectify_repair_ir",
    "validate_rendered_inputs",
    "validate_repair_ir",
]
