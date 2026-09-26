from __future__ import annotations

import json
from pathlib import Path

from lcfa import load_transitions, prepare_transition_file


def test_prepare_directory_skips_collection_manifest(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "task-001.json").write_text(
        json.dumps({
            "schema_version": "lcfa.semantic-trajectory.v1",
            "id": "semantic-episode:one",
            "goal": "fix x",
            "steps": [
                {
                    "index": 1,
                    "solution_id": "solution:1",
                    "hypothesis": None,
                    "action": {"name": "repo.search", "inputs": {"query": "x"}},
                    "observation": {},
                    "terminal": False,
                },
                {
                    "index": 2,
                    "solution_id": "solution:2",
                    "hypothesis": None,
                    "action": None,
                    "observation": {},
                    "terminal": True,
                },
            ],
            "final_solution_id": "solution:2",
            "patch": "",
            "success": True,
        }),
        encoding="utf-8",
    )
    (episodes / "collection.json").write_text(
        json.dumps({"format": "lcfa.recurrent-collection.v1", "tasks": 1}),
        encoding="utf-8",
    )
    output = tmp_path / "transitions.jsonl"
    count = prepare_transition_file([episodes], output)
    assert count == 2
    rows = load_transitions(output)
    assert [row.target_action for row in rows] == ["repo.search", "stop"]
    assert all(row.value_target == 1.0 for row in rows)
