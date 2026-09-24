"""Training-free cohort comparison operators."""

from __future__ import annotations

from math import sqrt
from statistics import fmean, median
from typing import Mapping

from ..protocol import ExecutionContext, OperatorResult
from ..registry import OperatorRegistry, OperatorSpec
from ._util import mapping, numbers


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _compare(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    current = numbers(inputs["current"], "cohort.compare.current")
    baseline = numbers(inputs["baseline"], "cohort.compare.baseline")
    current_mean = fmean(current)
    baseline_mean = fmean(baseline)
    delta = current_mean - baseline_mean
    percent_delta = None if baseline_mean == 0 else (delta / baseline_mean) * 100.0

    pooled_denominator = len(current) + len(baseline) - 2
    effect_size = None
    if pooled_denominator > 0:
        pooled_variance = (
            (len(current) - 1) * _sample_variance(current)
            + (len(baseline) - 1) * _sample_variance(baseline)
        ) / pooled_denominator
        if pooled_variance > 0:
            effect_size = delta / sqrt(pooled_variance)

    return OperatorResult(
        value={
            "current_count": len(current),
            "baseline_count": len(baseline),
            "current_mean": current_mean,
            "baseline_mean": baseline_mean,
            "delta": delta,
            "percent_delta": percent_delta,
            "effect_size": effect_size,
        }
    )


def _percentile(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    cohort = numbers(inputs["cohort"], "cohort.percentile.cohort")
    value = float(inputs["value"])
    below = sum(item < value for item in cohort)
    equal = sum(item == value for item in cohort)
    percentile = ((below + 0.5 * equal) / len(cohort)) * 100.0
    return OperatorResult(value=percentile)


def _summarize(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    groups = mapping(inputs["groups"], "cohort.summarize.groups")
    summary: dict[str, object] = {}
    for name, raw_values in groups.items():
        values = numbers(raw_values, f"cohort.summarize.groups.{name}")
        summary[str(name)] = {
            "count": len(values),
            "mean": fmean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
    return OperatorResult(value=summary)


def register_cohort_operators(registry: OperatorRegistry) -> OperatorRegistry:
    registry.register(OperatorSpec("cohort.compare", _compare, description="Compare two numeric cohorts and report mean shift and effect size."))
    registry.register(OperatorSpec("cohort.percentile", _percentile, description="Compute a midpoint percentile rank within a cohort."))
    registry.register(OperatorSpec("cohort.summarize", _summarize, description="Produce deterministic descriptive summaries for named cohorts."))
    return registry
