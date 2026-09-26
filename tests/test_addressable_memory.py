from __future__ import annotations

import pytest

from lcfa.addressable_memory import (
    BASE_OVERLAY,
    MEMORY_SPEC_FORMAT,
    MemoryConflictError,
    MemoryNotFoundError,
    SQLiteAddressableMemory,
    canonical_address,
    code_address,
)


def test_content_addressed_objects_refs_and_text_search(tmp_path) -> None:
    with SQLiteAddressableMemory(tmp_path / "memory.db") as store:
        first = store.put_object(
            {"symbol": "train_rwkv_heads", "identifier": "trust_remote_code"},
            kind="semantic",
            metadata={"path": "src/lcfa/recurrent_train.py"},
        )
        second = store.put_object(
            {"symbol": "train_rwkv_heads", "identifier": "trust_remote_code"},
            kind="semantic",
            metadata={"path": "src/lcfa/recurrent_train.py"},
        )
        assert first.hash == second.hash

        address = canonical_address("mem", "repos/lcfa/symbols/train_rwkv_heads")
        ref = store.bind(address, first.hash)
        assert ref.overlay_id == BASE_OVERLAY
        assert store.resolve(address).value["identifier"] == "trust_remote_code"

        hits = store.search_text("trust_remote_code")
        assert [item.hash for item in hits] == [first.hash]
        assert store.stats()["objects"] == 1


def test_code_addresses_and_typed_relations(tmp_path) -> None:
    with SQLiteAddressableMemory(tmp_path / "memory.db") as store:
        source = code_address(
            "symbol", "python/function/lcfa.recurrent_train/train_rwkv_heads"
        )
        target = code_address("identifier", "python/trust_remote_code")
        evidence = store.put_object(
            {"path": "src/lcfa/recurrent_train.py", "line": 83},
            kind="observation",
        )
        edge = store.link(
            source,
            "references",
            target,
            evidence_hash=evidence.hash,
            metadata={"source": "python-ast"},
        )
        assert edge.confidence == 1.0
        neighbors = store.neighbors(source, direction="out")
        assert len(neighbors) == 1
        assert neighbors[0].target == target
        assert neighbors[0].metadata["source"] == "python-ast"


def test_overlay_copy_on_write_whiteout_and_commit(tmp_path) -> None:
    goal = canonical_address("mem", "working/goal")
    obsolete = canonical_address("mem", "working/obsolete")
    with SQLiteAddressableMemory(tmp_path / "memory.db") as store:
        store.put(goal, {"text": "base"}, kind="working")
        store.put(obsolete, {"value": 1}, kind="working")
        overlay = store.create_overlay(overlay_id="session:test")

        assert store.resolve(goal, overlay_id=overlay).value["text"] == "base"
        store.put(goal, {"text": "branch"}, kind="working", overlay_id=overlay)
        store.unbind(obsolete, overlay_id=overlay)

        assert store.resolve(goal, overlay_id=overlay).value["text"] == "branch"
        with pytest.raises(MemoryNotFoundError):
            store.resolve(obsolete, overlay_id=overlay)
        assert store.resolve(goal).value["text"] == "base"

        assert store.commit_overlay(overlay) == BASE_OVERLAY
        assert store.resolve(goal).value["text"] == "branch"
        with pytest.raises(MemoryNotFoundError):
            store.resolve(obsolete)


def test_episodic_events_snapshots_and_vector_bindings(tmp_path) -> None:
    with SQLiteAddressableMemory(tmp_path / "memory.db") as store:
        address = canonical_address("mem", "tasks/example/goal")
        ref = store.put(address, {"goal": "fix loader"}, kind="working")
        first = store.append_event("episode:1", {"action": "repo.search"})
        second = store.append_event(
            "episode:1", {"action": "repo.read"}, state_hash="b3:state"
        )
        assert (first.sequence, second.sequence) == (1, 2)
        assert [event.event_hash for event in store.events("episode:1")] == [
            first.event_hash,
            second.event_hash,
        ]

        snapshot = store.snapshot(metadata={"phase": "before-edit"})
        assert snapshot.root_hash.startswith("b3:")
        assert snapshot.metadata["phase"] == "before-edit"

        binding = store.bind_vector(ref.object_hash, "code-embed-v1", 384)
        assert binding.vector_id > 0
        assert store.vector_binding_by_id(binding.vector_id) == binding
        with pytest.raises(MemoryConflictError):
            store.bind_vector(ref.object_hash, "code-embed-v1", 768)

        assert MEMORY_SPEC_FORMAT.startswith("lcfa.addressable-memory")
