"""Semantic identity and immutable state snapshot contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
import json
from threading import RLock
from typing import Any, Mapping, Protocol

from blake3 import blake3


class StateStoreError(RuntimeError):
    """Base error for state-store operations."""


class StateNotFoundError(StateStoreError):
    pass


class StateConflictError(StateStoreError):
    pass


@dataclass(frozen=True, slots=True)
class StateIdentity:
    """Stable logical identity plus immutable snapshot identity."""

    semantic_id: str
    version: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    identity: StateIdentity
    payload: Any
    effective_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class StateStore(Protocol):
    def put(
        self,
        semantic_id: str,
        payload: Any,
        *,
        effective_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> StateSnapshot:
        ...

    def get(self, semantic_id: str, version: int | None = None) -> StateSnapshot:
        ...

    def history(self, semantic_id: str) -> tuple[StateSnapshot, ...]:
        ...


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes used for immutable content identity."""

    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return f"b3:{blake3(canonical_bytes(value)).hexdigest()}"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    raise TypeError(f"state payload is not canonically JSON serializable: {type(value).__name__}")


class MemoryStateStore:
    """Reference in-memory implementation of semantic UUID -> version -> hash state."""

    def __init__(self) -> None:
        self._items: dict[str, list[StateSnapshot]] = {}
        self._lock = RLock()

    def put(
        self,
        semantic_id: str,
        payload: Any,
        *,
        effective_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> StateSnapshot:
        if not semantic_id:
            raise ValueError("semantic_id must not be empty")

        digest = content_hash(payload)
        with self._lock:
            snapshots = self._items.setdefault(semantic_id, [])
            current_version = snapshots[-1].identity.version if snapshots else 0
            if expected_version is not None and expected_version != current_version:
                raise StateConflictError(
                    f"expected version {expected_version} for {semantic_id}, found {current_version}"
                )

            snapshot = StateSnapshot(
                identity=StateIdentity(
                    semantic_id=semantic_id,
                    version=current_version + 1,
                    content_hash=digest,
                ),
                payload=payload,
                effective_at=effective_at,
                metadata=dict(metadata or {}),
            )
            snapshots.append(snapshot)
            return snapshot

    def get(self, semantic_id: str, version: int | None = None) -> StateSnapshot:
        with self._lock:
            snapshots = self._items.get(semantic_id)
            if not snapshots:
                raise StateNotFoundError(semantic_id)
            if version is None:
                return snapshots[-1]
            if version < 1 or version > len(snapshots):
                raise StateNotFoundError(f"{semantic_id}@{version}")
            return snapshots[version - 1]

    def history(self, semantic_id: str) -> tuple[StateSnapshot, ...]:
        with self._lock:
            snapshots = self._items.get(semantic_id)
            if not snapshots:
                raise StateNotFoundError(semantic_id)
            return tuple(snapshots)
