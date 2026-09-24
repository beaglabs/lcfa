"""Shared validation helpers for deterministic operator libraries."""

from __future__ import annotations

from datetime import datetime, timezone
from numbers import Real
from typing import Any, Mapping, Sequence


def sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} expects a sequence")
    return value


def numbers(value: object, name: str) -> list[float]:
    items = sequence(value, name)
    out: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{name} expects numeric values")
        out.append(float(item))
    if not out:
        raise ValueError(f"{name} requires at least one value")
    return out


def mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} expects a mapping")
    return value


def get_path(value: object, path: str) -> Any:
    current: Any = value
    if not path:
        return current
    for segment in path.split("."):
        if isinstance(current, Mapping):
            current = current[segment]
        else:
            current = getattr(current, segment)
    return current


def parse_time(value: object) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, Real) and not isinstance(value, bool):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    else:
        raise TypeError(f"unsupported timestamp value: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def canonical_key(value: object) -> str:
    if isinstance(value, Mapping):
        return repr(tuple(sorted((str(k), canonical_key(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return repr(tuple(canonical_key(v) for v in value))
    return repr(value)
