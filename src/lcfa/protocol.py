"""Core protocol types for LCFA.

The core deliberately models reasoning, solution state, and agentic actions as
separate typed artifacts. Neural weights are optional implementations behind
these contracts rather than prerequisites for execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EntityRef:
    id: str
    type: str
    namespace: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    id: str
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceValue:
    """A state value coupled to the evidence that supports it."""

    value: Any
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    value: Any
    subject: EntityRef | None = None
    unit: str | None = None
    confidence: float | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Declarative recommendation; this is not itself an external action."""

    kind: str
    subject: EntityRef | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True, slots=True)
class OperatorResult:
    value: Any
    evidence: tuple[EvidenceRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanNode:
    id: str
    operator: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReasoningPlan:
    id: str
    nodes: tuple[PlanNode, ...]
    outputs: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "lcfa.plan.v1"


@dataclass(frozen=True, slots=True)
class TraceStep:
    node_id: str
    operation: str
    kind: str
    status: str
    duration_ms: float
    evidence: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    plan_id: str
    steps: tuple[TraceStep, ...]
    schema_version: str = "lcfa.trace.v1"


@dataclass(frozen=True, slots=True)
class SolutionState:
    id: str
    plan_id: str
    values: Mapping[str, Any]
    findings: tuple[Finding, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    trace: ExecutionTrace | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "lcfa.solution.v1"


@dataclass(frozen=True, slots=True)
class ActionNode:
    id: str
    action: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    requires_approval: bool = False
    approval_key: str | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class ActionGraph:
    id: str
    source_solution_id: str
    nodes: tuple[ActionNode, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "lcfa.action_graph.v1"


@dataclass(frozen=True, slots=True)
class ActionResult:
    value: Any = None
    observations: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionRun:
    graph_id: str
    source_solution_id: str
    results: Mapping[str, ActionResult]
    observations: Mapping[str, Any]
    trace: ExecutionTrace


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    state: Mapping[str, Any] = field(default_factory=dict)
    capabilities: frozenset[str] = frozenset()
    approvals: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)
