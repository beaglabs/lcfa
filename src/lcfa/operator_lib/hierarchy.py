"""Domain-neutral hierarchy and relationship-graph operators."""

from __future__ import annotations

from collections import deque
from statistics import fmean
from typing import Any, Mapping

from ..protocol import ExecutionContext, OperatorResult
from ..registry import OperatorRegistry, OperatorSpec
from ._util import mapping, sequence


def _relations(value: object) -> list[tuple[str, str, str | None]]:
    items = sequence(value, "hierarchy.relations")
    out: list[tuple[str, str, str | None]] = []
    for item in items:
        rel = mapping(item, "hierarchy relation")
        source = str(rel["source"])
        target = str(rel["target"])
        predicate = str(rel["predicate"]) if "predicate" in rel and rel["predicate"] is not None else None
        out.append((source, target, predicate))
    return out


def _root_ids(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in sequence(value, "hierarchy.roots")]


def _neighbors(
    relations: list[tuple[str, str, str | None]],
    node: str,
    *,
    direction: str,
    predicate: str | None,
) -> list[str]:
    found: list[str] = []
    for source, target, edge_predicate in relations:
        if predicate is not None and edge_predicate != predicate:
            continue
        if direction == "children" and target == node:
            found.append(source)
        elif direction == "parents" and source == node:
            found.append(target)
        elif direction == "forward" and source == node:
            found.append(target)
        elif direction == "reverse" and target == node:
            found.append(source)
        elif direction == "both":
            if source == node:
                found.append(target)
            if target == node:
                found.append(source)
    return found


def _expand(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    relations = _relations(inputs["relations"])
    roots = _root_ids(inputs["roots"])
    direction = str(params.get("direction", "children"))
    if direction not in {"children", "parents", "forward", "reverse", "both"}:
        raise ValueError(f"unsupported hierarchy direction: {direction}")
    predicate = str(params["predicate"]) if params.get("predicate") is not None else None
    max_depth = int(params.get("max_depth", 1))
    if max_depth < 0:
        raise ValueError("hierarchy.expand max_depth must be >= 0")

    depths: dict[str, int] = {root: 0 for root in roots}
    queue = deque(roots)
    order: list[str] = list(roots)
    while queue:
        current = queue.popleft()
        depth = depths[current]
        if depth >= max_depth:
            continue
        for neighbor in _neighbors(relations, current, direction=direction, predicate=predicate):
            if neighbor in depths:
                continue
            depths[neighbor] = depth + 1
            order.append(neighbor)
            queue.append(neighbor)

    return OperatorResult(value={"nodes": order, "depth": depths})


def _path(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    relations = _relations(inputs["relations"])
    source = str(inputs["source"])
    target = str(inputs["target"])
    direction = str(params.get("direction", "both"))
    predicate = str(params["predicate"]) if params.get("predicate") is not None else None
    max_depth = int(params.get("max_depth", 32))

    queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
    visited = {source}
    while queue:
        current, path = queue.popleft()
        if current == target:
            return OperatorResult(value=path)
        if len(path) - 1 >= max_depth:
            continue
        for neighbor in _neighbors(relations, current, direction=direction, predicate=predicate):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
    return OperatorResult(value=None)


def _aggregate(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    raw_values = mapping(inputs["values"], "hierarchy.aggregate.values")
    nodes = _root_ids(inputs["nodes"])
    selected: list[float] = []
    for node in nodes:
        if node in raw_values:
            selected.append(float(raw_values[node]))

    op = str(params.get("op", "mean"))
    if op == "count":
        value: Any = len(selected)
    elif not selected:
        raise ValueError("hierarchy.aggregate selected no numeric values")
    elif op == "mean":
        value = fmean(selected)
    elif op == "sum":
        value = sum(selected)
    elif op == "min":
        value = min(selected)
    elif op == "max":
        value = max(selected)
    else:
        raise ValueError(f"unsupported hierarchy aggregate op: {op}")
    return OperatorResult(value=value, metadata={"selected": len(selected), "op": op})


def register_hierarchy_operators(registry: OperatorRegistry) -> OperatorRegistry:
    registry.register(OperatorSpec("hierarchy.expand", _expand, description="Traverse typed relationships from one or more roots."))
    registry.register(OperatorSpec("hierarchy.path", _path, description="Find a shortest path through a relationship graph."))
    registry.register(OperatorSpec("hierarchy.aggregate", _aggregate, description="Aggregate values over an explicitly selected node set."))
    return registry
