"""Optional LCFA actions backed by the Addressable Memory Specification."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .addressable_memory import BASE_OVERLAY, SQLiteAddressableMemory
from .protocol import ActionResult, ExecutionContext
from .registry import ActionRegistry, ActionSpec


class MemoryActionError(RuntimeError):
    pass


def _require(context: ExecutionContext, capability: str) -> None:
    if capability not in context.capabilities:
        raise MemoryActionError(f"{capability} capability is required")


def _path(context: ExecutionContext) -> Path:
    raw = context.metadata.get("memory_db")
    if not raw:
        raise MemoryActionError("memory_db is required in ExecutionContext.metadata")
    return Path(str(raw))


def _overlay(context: ExecutionContext) -> str:
    return str(context.metadata.get("memory_overlay") or BASE_OVERLAY)


def _memory_put(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    _require(context, "memory.write")
    address = str(inputs.get("address") or "")
    kind = str(inputs.get("kind") or "working")
    if not address:
        raise MemoryActionError("memory.put requires address")
    metadata = inputs.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise MemoryActionError("memory.put metadata must be an object")
    with SQLiteAddressableMemory(_path(context)) as store:
        ref = store.put(
            address,
            inputs.get("value"),
            kind=kind,
            media_type=str(inputs.get("media_type") or "application/json"),
            metadata=dict(metadata or {}),
            overlay_id=_overlay(context),
            scope=str(inputs.get("scope") or "session"),
        )
    value = {"address": ref.address, "object_hash": ref.object_hash, "overlay_id": ref.overlay_id}
    return ActionResult(value=value, observations={"memory": value})


def _memory_get(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    _require(context, "memory.read")
    address = str(inputs.get("address") or "")
    if not address:
        raise MemoryActionError("memory.get requires address")
    with SQLiteAddressableMemory(_path(context)) as store:
        ref = store.resolve_ref(address, overlay_id=_overlay(context))
        obj = store.get_object(ref.object_hash)
    value = {
        "address": ref.address,
        "object_hash": obj.hash,
        "kind": obj.kind,
        "media_type": obj.media_type,
        "value": obj.value,
        "metadata": dict(obj.metadata),
    }
    return ActionResult(value=value, observations={"memory": value})


def _memory_link(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    _require(context, "memory.write")
    source, relation, target = (str(inputs.get(key) or "") for key in ("source", "relation", "target"))
    if not source or not relation or not target:
        raise MemoryActionError("memory.link requires source, relation, and target")
    metadata = inputs.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise MemoryActionError("memory.link metadata must be an object")
    with SQLiteAddressableMemory(_path(context)) as store:
        edge = store.link(
            source,
            relation,
            target,
            evidence_hash=(str(inputs["evidence_hash"]) if inputs.get("evidence_hash") else None),
            confidence=float(inputs.get("confidence", 1.0) or 1.0),
            metadata=dict(metadata or {}),
        )
    value = {"source": edge.source, "relation": edge.relation, "target": edge.target, "confidence": edge.confidence}
    return ActionResult(value=value, observations={"memory_relation": value})


def _memory_search(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    _require(context, "memory.read")
    query = str(inputs.get("query") or "").strip()
    if not query:
        raise MemoryActionError("memory.search requires query")
    kinds = inputs.get("kinds")
    if kinds is not None and not isinstance(kinds, (list, tuple)):
        raise MemoryActionError("memory.search kinds must be an array")
    with SQLiteAddressableMemory(_path(context)) as store:
        hits = store.search_text(query, kinds=[str(item) for item in (kinds or ())], limit=int(inputs.get("limit", 20) or 20))
    value = {
        "query": query,
        "hits": [
            {"object_hash": item.hash, "kind": item.kind, "media_type": item.media_type, "metadata": dict(item.metadata)}
            for item in hits
        ],
    }
    return ActionResult(value=value, observations={"memory_search": value})


def _memory_snapshot(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    _require(context, "memory.read")
    metadata = inputs.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise MemoryActionError("memory.snapshot metadata must be an object")
    with SQLiteAddressableMemory(_path(context)) as store:
        snapshot = store.snapshot(overlay_id=_overlay(context), metadata=dict(metadata or {}))
    value = {"id": snapshot.id, "overlay_id": snapshot.overlay_id, "root_hash": snapshot.root_hash, "parent_id": snapshot.parent_id}
    return ActionResult(value=value, observations={"memory_snapshot": value})


def _memory_branch(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    _require(context, "memory.write")
    metadata = inputs.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise MemoryActionError("memory.branch metadata must be an object")
    with SQLiteAddressableMemory(_path(context)) as store:
        overlay_id = store.create_overlay(
            parent_id=str(inputs.get("parent_id") or _overlay(context)),
            overlay_id=(str(inputs["overlay_id"]) if inputs.get("overlay_id") else None),
            metadata=dict(metadata or {}),
        )
    value = {"overlay_id": overlay_id}
    return ActionResult(value=value, observations={"memory_branch": value})


def _memory_commit(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    _require(context, "memory.write")
    overlay_id = str(inputs.get("overlay_id") or _overlay(context))
    with SQLiteAddressableMemory(_path(context)) as store:
        parent_id = store.commit_overlay(overlay_id)
    value = {"overlay_id": overlay_id, "parent_id": parent_id}
    return ActionResult(value=value, observations={"memory_commit": value})


def register_memory_actions(registry: ActionRegistry | None = None) -> ActionRegistry:
    registry = registry or ActionRegistry()
    for spec in (
        ActionSpec("memory.put", _memory_put, effects=("memory.write",), description="Bind an immutable memory object to an address."),
        ActionSpec("memory.get", _memory_get, description="Resolve an address in the current memory overlay."),
        ActionSpec("memory.link", _memory_link, effects=("memory.write",), description="Create or update a typed memory relation."),
        ActionSpec("memory.search", _memory_search, description="Search canonical memory using the lexical index."),
        ActionSpec("memory.snapshot", _memory_snapshot, description="Snapshot the current visible memory namespace."),
        ActionSpec("memory.branch", _memory_branch, effects=("memory.write",), description="Create a copy-on-write memory overlay."),
        ActionSpec("memory.commit", _memory_commit, effects=("memory.write",), description="Commit an overlay into its parent."),
    ):
        registry.register(spec)
    return registry


__all__ = ["MemoryActionError", "register_memory_actions"]
