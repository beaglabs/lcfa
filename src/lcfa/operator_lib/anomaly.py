"""Deterministic anomaly and smoothing operators."""

from __future__ import annotations

from math import inf
from statistics import median
from typing import Mapping

from ..protocol import ExecutionContext, OperatorResult
from ..registry import OperatorRegistry, OperatorSpec
from ._util import numbers


def _robust_scores(values: list[float]) -> list[float]:
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations)
    if mad == 0:
        return [0.0 if value == center else (inf if value > center else -inf) for value in values]
    scale = 1.4826 * mad
    return [(value - center) / scale for value in values]


def _robust_zscores(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    values = numbers(inputs["values"], "anomaly.robust_zscores.values")
    scores = _robust_scores(values)
    return OperatorResult(value=scores, metadata={"median": median(values), "method": "MAD"})


def _outliers(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    values = numbers(inputs["values"], "anomaly.outliers.values")
    threshold = float(params.get("threshold", 3.5))
    if threshold <= 0:
        raise ValueError("anomaly.outliers threshold must be > 0")
    scores = _robust_scores(values)
    result = [
        {"index": index, "value": value, "score": score}
        for index, (value, score) in enumerate(zip(values, scores))
        if abs(score) >= threshold
    ]
    return OperatorResult(value=result, metadata={"threshold": threshold, "method": "MAD"})


def _ewma(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    values = numbers(inputs["values"], "anomaly.ewma.values")
    alpha = float(params.get("alpha", 0.3))
    if not 0.0 < alpha <= 1.0:
        raise ValueError("anomaly.ewma alpha must be in (0, 1]")
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return OperatorResult(value=smoothed, metadata={"alpha": alpha})


def register_anomaly_operators(registry: OperatorRegistry) -> OperatorRegistry:
    registry.register(OperatorSpec("anomaly.robust_zscores", _robust_zscores, description="Compute median/MAD robust z-scores."))
    registry.register(OperatorSpec("anomaly.outliers", _outliers, description="Locate robust MAD outliers above a configurable threshold."))
    registry.register(OperatorSpec("anomaly.ewma", _ewma, description="Compute an exponentially weighted moving average."))
    return registry
