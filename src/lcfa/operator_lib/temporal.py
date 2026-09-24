"""Training-free temporal reasoning operators."""

from __future__ import annotations

from statistics import fmean
from typing import Mapping

from ..protocol import ExecutionContext, OperatorResult
from ..registry import OperatorRegistry, OperatorSpec
from ._util import get_path, numbers, parse_time, sequence


def _window(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    items = sequence(inputs["items"], "temporal.window.items")
    field = str(params.get("timestamp_field", "timestamp"))
    start_raw = inputs.get("start", params.get("start"))
    end_raw = inputs.get("end", params.get("end"))
    start = parse_time(start_raw) if start_raw is not None else None
    end = parse_time(end_raw) if end_raw is not None else None
    inclusive = bool(params.get("inclusive", True))

    selected: list[object] = []
    for item in items:
        timestamp = parse_time(get_path(item, field))
        after_start = start is None or (timestamp >= start if inclusive else timestamp > start)
        before_end = end is None or (timestamp <= end if inclusive else timestamp < end)
        if after_start and before_end:
            selected.append(item)
    return OperatorResult(value=selected, metadata={"count": len(selected)})


def _slope(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    ys = numbers(inputs["values"], "temporal.slope.values")
    if len(ys) < 2:
        raise ValueError("temporal.slope requires at least two values")

    raw_times = inputs.get("times")
    if raw_times is None:
        xs = [float(index) for index in range(len(ys))]
    else:
        times = sequence(raw_times, "temporal.slope.times")
        if len(times) != len(ys):
            raise ValueError("temporal.slope times and values must have equal length")
        parsed = [parse_time(value).timestamp() for value in times]
        origin = parsed[0]
        xs = [value - origin for value in parsed]

    x_mean = fmean(xs)
    y_mean = fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("temporal.slope requires distinct time coordinates")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    return OperatorResult(value=slope)


def _rolling_mean(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    values = numbers(inputs["values"], "temporal.rolling_mean.values")
    width = int(params.get("window", 3))
    if width < 1:
        raise ValueError("temporal.rolling_mean window must be >= 1")
    if width > len(values):
        return OperatorResult(value=[])
    result = [fmean(values[index - width + 1 : index + 1]) for index in range(width - 1, len(values))]
    return OperatorResult(value=result, metadata={"window": width})


def _change_point(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    values = numbers(inputs["values"], "temporal.change_point.values")
    min_segment = int(params.get("min_segment", 2))
    if min_segment < 1:
        raise ValueError("temporal.change_point min_segment must be >= 1")
    if len(values) < min_segment * 2:
        raise ValueError("temporal.change_point requires two complete segments")

    candidates: list[tuple[float, int, float, float]] = []
    for split in range(min_segment, len(values) - min_segment + 1):
        before = fmean(values[:split])
        after = fmean(values[split:])
        delta = after - before
        candidates.append((abs(delta), split, before, after))

    score, split, before, after = max(candidates, key=lambda item: item[0])
    return OperatorResult(
        value={
            "index": split,
            "before_mean": before,
            "after_mean": after,
            "delta": after - before,
            "score": score,
        }
    )


def register_temporal_operators(registry: OperatorRegistry) -> OperatorRegistry:
    registry.register(OperatorSpec("temporal.window", _window, description="Filter records by an inclusive or exclusive time window."))
    registry.register(OperatorSpec("temporal.slope", _slope, description="Compute least-squares temporal slope."))
    registry.register(OperatorSpec("temporal.rolling_mean", _rolling_mean, description="Compute fixed-width rolling means."))
    registry.register(OperatorSpec("temporal.change_point", _change_point, description="Locate the strongest mean-shift split point."))
    return registry
