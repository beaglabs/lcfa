"""Generic constraint checking and filtering operators."""

from __future__ import annotations

from typing import Any, Mapping

from ..protocol import ExecutionContext, OperatorResult
from ..registry import OperatorRegistry, OperatorSpec
from ._util import get_path, sequence


class ConstraintViolation(ValueError):
    pass


def _compare(lhs: Any, rhs: Any, op: str) -> bool:
    if op == "eq":
        return lhs == rhs
    if op == "ne":
        return lhs != rhs
    if op == "gt":
        return lhs > rhs
    if op == "gte":
        return lhs >= rhs
    if op == "lt":
        return lhs < rhs
    if op == "lte":
        return lhs <= rhs
    if op == "in":
        return lhs in rhs
    if op == "not_in":
        return lhs not in rhs
    if op == "contains":
        return rhs in lhs
    raise ValueError(f"unsupported constraint operator: {op}")


def _check(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    op = str(params.get("op", "eq"))
    lhs = inputs.get("lhs")
    rhs = inputs.get("rhs", params.get("rhs"))
    passed = _compare(lhs, rhs, op)
    return OperatorResult(value={"passed": passed, "lhs": lhs, "rhs": rhs, "op": op})


def _require(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    op = str(params.get("op", "eq"))
    lhs = inputs.get("lhs")
    rhs = inputs.get("rhs", params.get("rhs"))
    if not _compare(lhs, rhs, op):
        message = str(params.get("message", f"constraint failed: {lhs!r} {op} {rhs!r}"))
        raise ConstraintViolation(message)
    return OperatorResult(value={"passed": True, "lhs": lhs, "rhs": rhs, "op": op})


def _filter(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    items = sequence(inputs["items"], "constraints.filter.items")
    field = str(params.get("field", ""))
    op = str(params.get("op", "eq"))
    rhs = inputs.get("value", params.get("value"))
    selected = [item for item in items if _compare(get_path(item, field), rhs, op)]
    return OperatorResult(value=selected, metadata={"selected": len(selected), "total": len(items)})


def _all(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    conditions = sequence(inputs["conditions"], "constraints.all.conditions")
    passed = all(bool(condition) for condition in conditions)
    if bool(params.get("raise_on_fail", False)) and not passed:
        raise ConstraintViolation(str(params.get("message", "one or more constraints failed")))
    return OperatorResult(value=passed)


def register_constraint_operators(registry: OperatorRegistry) -> OperatorRegistry:
    registry.register(OperatorSpec("constraints.check", _check, description="Evaluate a typed comparison without side effects."))
    registry.register(OperatorSpec("constraints.require", _require, description="Require a typed comparison to hold or abort reasoning."))
    registry.register(OperatorSpec("constraints.filter", _filter, description="Filter records using a field comparison."))
    registry.register(OperatorSpec("constraints.all", _all, description="Combine boolean constraints with logical AND."))
    return registry
