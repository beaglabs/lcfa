"""Stable JSON serialization for LCFA protocol artifacts."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from . import protocol as p


IR_FORMAT = "lcfa.ir.v1"
_TYPE_KEY = "$lcfa_type"


class SerializationError(ValueError):
    pass


_PROTOCOL_TYPES = {
    cls.__name__: cls
    for cls in (
        p.EntityRef,
        p.EvidenceRef,
        p.EvidenceValue,
        p.Finding,
        p.Recommendation,
        p.OperatorResult,
        p.PlanNode,
        p.ReasoningPlan,
        p.TraceStep,
        p.ExecutionTrace,
        p.SolutionState,
        p.ActionNode,
        p.ActionGraph,
        p.ActionResult,
        p.ActionRun,
        p.ExecutionContext,
    )
}

_KIND_BY_TYPE = {
    p.ReasoningPlan: "reasoning_plan",
    p.SolutionState: "solution_state",
    p.ActionGraph: "action_graph",
    p.ExecutionTrace: "execution_trace",
}


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        cls = type(value)
        if cls.__name__ not in _PROTOCOL_TYPES:
            raise SerializationError(f"unsupported dataclass type: {cls.__name__}")
        body = {_TYPE_KEY: cls.__name__}
        for item in fields(value):
            body[item.name] = _encode(getattr(value, item.name))
        return body
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, frozenset):
        return {_TYPE_KEY: "frozenset", "items": [_encode(item) for item in sorted(value)]}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise SerializationError(f"unsupported IR value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value

    tag = value.get(_TYPE_KEY)
    if tag == "tuple":
        return tuple(_decode(item) for item in value.get("items", []))
    if tag == "frozenset":
        return frozenset(_decode(item) for item in value.get("items", []))
    if tag in _PROTOCOL_TYPES:
        cls = _PROTOCOL_TYPES[tag]
        kwargs = {
            key: _decode(item)
            for key, item in value.items()
            if key != _TYPE_KEY
        }
        return cls(**kwargs)
    if tag is not None:
        raise SerializationError(f"unknown LCFA IR type: {tag}")
    return {key: _decode(item) for key, item in value.items()}


def to_ir_dict(value: Any) -> dict[str, Any]:
    kind = _KIND_BY_TYPE.get(type(value))
    if kind is None:
        raise SerializationError(f"unsupported top-level IR artifact: {type(value).__name__}")
    return {
        "format": IR_FORMAT,
        "kind": kind,
        "payload": _encode(value),
    }


def from_ir_dict(data: Mapping[str, Any]) -> Any:
    if data.get("format") != IR_FORMAT:
        raise SerializationError(f"unsupported IR format: {data.get('format')!r}")
    decoded = _decode(data.get("payload"))
    expected_kind = _KIND_BY_TYPE.get(type(decoded))
    if expected_kind != data.get("kind"):
        raise SerializationError(
            f"IR kind mismatch: envelope={data.get('kind')!r}, payload={expected_kind!r}"
        )
    return decoded


def dumps_ir(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(to_ir_dict(value), indent=indent, sort_keys=True)


def loads_ir(text: str) -> Any:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise SerializationError("IR document must be a JSON object")
    return from_ir_dict(raw)


def dump_ir(value: Any, path: str | Path, *, indent: int | None = 2) -> None:
    Path(path).write_text(dumps_ir(value, indent=indent) + "\n", encoding="utf-8")


def load_ir(path: str | Path) -> Any:
    return loads_ir(Path(path).read_text(encoding="utf-8"))
