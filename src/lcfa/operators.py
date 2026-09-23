"""Training-free operator registration used by LCFA-Zero and examples."""

from __future__ import annotations

from statistics import fmean
from typing import Mapping

from .operator_lib import register_reasoning_operators
from .protocol import ExecutionContext, Finding, OperatorResult, Recommendation
from .registry import OperatorRegistry, OperatorSpec


def _identity(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    return OperatorResult(value=inputs.get("value"))


def _mean(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    values = inputs["values"]
    if not isinstance(values, (list, tuple)):
        raise TypeError("stats.mean expects a list or tuple")
    return OperatorResult(value=fmean(values))


def _delta(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    return OperatorResult(value=float(inputs["current"]) - float(inputs["baseline"]))


def _finding(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    finding = Finding(
        kind=str(params["kind"]),
        value=inputs.get("value"),
        unit=str(params["unit"]) if "unit" in params else None,
    )
    return OperatorResult(value=finding)


def _recommend(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    recommendation = Recommendation(
        kind=str(params["kind"]),
        parameters={"value": inputs.get("value"), **dict(params.get("parameters", {}))},
    )
    return OperatorResult(value=recommendation)


def register_core_operators(registry: OperatorRegistry) -> OperatorRegistry:
    registry.register(OperatorSpec("core.identity", _identity))
    registry.register(OperatorSpec("stats.mean", _mean))
    registry.register(OperatorSpec("stats.delta", _delta))
    registry.register(OperatorSpec("core.finding", _finding))
    registry.register(OperatorSpec("core.recommend", _recommend))
    register_reasoning_operators(registry)
    return registry
