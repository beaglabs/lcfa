from __future__ import annotations

from lcfa.memory_actions import register_memory_actions
from lcfa.protocol import ExecutionContext


def test_memory_actions_round_trip(tmp_path) -> None:
    db = tmp_path / "memory.db"
    registry = register_memory_actions()
    context = ExecutionContext(
        capabilities=frozenset({"memory.read", "memory.write"}),
        metadata={"memory_db": str(db)},
    )

    put = registry.get("memory.put").handler(
        context,
        {"address": "mem://working/goal", "kind": "working", "value": {"goal": "fix loader"}},
    )
    assert put.value["object_hash"].startswith("b3:")

    get = registry.get("memory.get").handler(context, {"address": "mem://working/goal"})
    assert get.value["value"] == {"goal": "fix loader"}

    search = registry.get("memory.search").handler(context, {"query": "loader"})
    assert len(search.value["hits"]) == 1

    branch = registry.get("memory.branch").handler(context, {"overlay_id": "session:test"})
    assert branch.value["overlay_id"] == "session:test"
