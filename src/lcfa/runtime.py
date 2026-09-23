"""Deterministic reasoning and governed action execution for LCFA-Zero."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .protocol import (
    ActionGraph,
    ActionResult,
    ActionRun,
    EvidenceRef,
    EvidenceValue,
    ExecutionContext,
    ExecutionTrace,
    Finding,
    OperatorResult,
    ReasoningPlan,
    Recommendation,
    SolutionState,
    TraceStep,
)
from .registry import ActionRegistry, OperatorRegistry


class PlanError(RuntimeError):
    pass


class ActionPolicyError(RuntimeError):
    pass


def _dedupe_evidence(items: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    seen: set[str] = set()
    out: list[EvidenceRef] = []
    for item in items:
        if item.id not in seen:
            seen.add(item.id)
            out.append(item)
    return tuple(out)


def _deep_get(value: Any, path: str) -> Any:
    if not path:
        return value
    current = value
    for segment in path.split("."):
        if isinstance(current, Mapping):
            current = current[segment]
        else:
            current = getattr(current, segment)
    return current


def _resolve_reasoning_binding(
    binding: Any,
    context: ExecutionContext,
    results: Mapping[str, OperatorResult],
) -> tuple[Any, tuple[EvidenceRef, ...]]:
    if not isinstance(binding, str) or not binding.startswith("$"):
        return binding, ()

    if binding.startswith("$state."):
        raw = _deep_get(context.state, binding.removeprefix("$state."))
        if isinstance(raw, EvidenceValue):
            return raw.value, raw.evidence
        return raw, ()

    if binding.startswith("$node."):
        ref = binding.removeprefix("$node.")
        node_id, _, path = ref.partition(".")
        if node_id not in results:
            raise PlanError(f"node binding references unavailable result: {node_id}")
        result = results[node_id]
        return _deep_get(result.value, path), result.evidence

    raise PlanError(f"unsupported reasoning binding: {binding}")


def _resolve_action_binding(
    binding: Any,
    solution: SolutionState,
    results: Mapping[str, ActionResult],
) -> Any:
    if not isinstance(binding, str) or not binding.startswith("$"):
        return binding

    if binding.startswith("$solution."):
        return _deep_get(solution, binding.removeprefix("$solution."))

    if binding.startswith("$action."):
        ref = binding.removeprefix("$action.")
        node_id, _, path = ref.partition(".")
        if node_id not in results:
            raise PlanError(f"action binding references unavailable result: {node_id}")
        return _deep_get(results[node_id].value, path)

    raise PlanError(f"unsupported action binding: {binding}")


def _topological_ids(nodes: Iterable[Any]) -> tuple[str, ...]:
    by_id = {node.id: node for node in nodes}
    if len(by_id) != len(tuple(nodes)):
        raise PlanError("graph contains duplicate node ids")

    remaining = set(by_id)
    resolved: set[str] = set()
    ordered: list[str] = []

    while remaining:
        ready = sorted(
            node_id
            for node_id in remaining
            if set(by_id[node_id].depends_on) <= resolved
        )
        if not ready:
            unresolved = {node_id: by_id[node_id].depends_on for node_id in remaining}
            raise PlanError(f"graph has a cycle or missing dependency: {unresolved}")
        for node_id in ready:
            missing = set(by_id[node_id].depends_on) - set(by_id)
            if missing:
                raise PlanError(f"node {node_id} references missing dependencies: {sorted(missing)}")
            ordered.append(node_id)
            resolved.add(node_id)
            remaining.remove(node_id)

    return tuple(ordered)


class ReasoningExecutor:
    def __init__(self, operators: OperatorRegistry) -> None:
        self.operators = operators

    def execute(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        by_id = {node.id: node for node in plan.nodes}
        order = _topological_ids(plan.nodes)
        results: dict[str, OperatorResult] = {}
        steps: list[TraceStep] = []

        for node_id in order:
            node = by_id[node_id]
            spec = self.operators.get(node.operator)
            started = perf_counter()
            try:
                resolved_inputs: dict[str, Any] = {}
                inherited_evidence: list[EvidenceRef] = []
                for name, binding in node.inputs.items():
                    value, evidence = _resolve_reasoning_binding(binding, context, results)
                    resolved_inputs[name] = value
                    inherited_evidence.extend(evidence)

                result = spec.handler(context, resolved_inputs, node.params)
                if spec.evidence_preserving:
                    result = replace(
                        result,
                        evidence=_dedupe_evidence((*inherited_evidence, *result.evidence)),
                    )
                results[node_id] = result
                steps.append(
                    TraceStep(
                        node_id=node_id,
                        operation=node.operator,
                        kind="compute",
                        status="ok",
                        duration_ms=(perf_counter() - started) * 1000.0,
                        evidence=tuple(ref.id for ref in result.evidence),
                    )
                )
            except Exception as exc:
                steps.append(
                    TraceStep(
                        node_id=node_id,
                        operation=node.operator,
                        kind="compute",
                        status="error",
                        duration_ms=(perf_counter() - started) * 1000.0,
                        error=str(exc),
                    )
                )
                raise

        missing_outputs = set(plan.outputs) - set(results)
        if missing_outputs:
            raise PlanError(f"plan outputs do not exist: {sorted(missing_outputs)}")

        output_values = {node_id: results[node_id].value for node_id in plan.outputs}
        all_evidence = _dedupe_evidence(
            ref for node_id in plan.outputs for ref in results[node_id].evidence
        )

        findings: list[Finding] = []
        recommendations: list[Recommendation] = []
        for value in output_values.values():
            values = value if isinstance(value, (list, tuple)) else (value,)
            for item in values:
                if isinstance(item, Finding):
                    findings.append(item)
                elif isinstance(item, Recommendation):
                    recommendations.append(item)

        return SolutionState(
            id=f"solution:{uuid4()}",
            plan_id=plan.id,
            values=output_values,
            findings=tuple(findings),
            recommendations=tuple(recommendations),
            evidence=all_evidence,
            trace=ExecutionTrace(plan_id=plan.id, steps=tuple(steps)),
        )


class ActionExecutor:
    """Executes only explicitly registered actions after capability/approval checks."""

    def __init__(self, actions: ActionRegistry) -> None:
        self.actions = actions

    def execute(
        self,
        graph: ActionGraph,
        solution: SolutionState,
        context: ExecutionContext,
    ) -> ActionRun:
        if graph.source_solution_id != solution.id:
            raise ActionPolicyError("action graph does not belong to supplied SolutionState")

        by_id = {node.id: node for node in graph.nodes}
        order = _topological_ids(graph.nodes)
        results: dict[str, ActionResult] = {}
        observations: dict[str, Any] = {}
        steps: list[TraceStep] = []

        for node_id in order:
            node = by_id[node_id]
            spec = self.actions.get(node.action)
            missing_caps = set(node.required_capabilities) - set(context.capabilities)
            if missing_caps:
                raise ActionPolicyError(
                    f"action {node.id} missing capabilities: {sorted(missing_caps)}"
                )
            if node.requires_approval:
                approval_key = node.approval_key or node.id
                if approval_key not in context.approvals:
                    raise ActionPolicyError(
                        f"action {node.id} requires approval: {approval_key}"
                    )

            started = perf_counter()
            try:
                resolved_inputs = {
                    name: _resolve_action_binding(binding, solution, results)
                    for name, binding in node.inputs.items()
                }
                result = spec.handler(context, resolved_inputs)
                results[node_id] = result
                for key, value in result.observations.items():
                    observations[f"{node_id}.{key}"] = value
                steps.append(
                    TraceStep(
                        node_id=node_id,
                        operation=node.action,
                        kind="action",
                        status="ok",
                        duration_ms=(perf_counter() - started) * 1000.0,
                        evidence=tuple(ref.id for ref in result.evidence),
                    )
                )
            except Exception as exc:
                steps.append(
                    TraceStep(
                        node_id=node_id,
                        operation=node.action,
                        kind="action",
                        status="error",
                        duration_ms=(perf_counter() - started) * 1000.0,
                        error=str(exc),
                    )
                )
                raise

        return ActionRun(
            graph_id=graph.id,
            source_solution_id=solution.id,
            results=results,
            observations=observations,
            trace=ExecutionTrace(plan_id=graph.id, steps=tuple(steps)),
        )
