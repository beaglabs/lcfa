from __future__ import annotations

import json
from pathlib import Path

from lcfa.recurrent_transitions import (
    RECURRENT_TRANSITION_FORMAT,
    episode_to_transitions,
    load_transitions,
)


def _episode() -> dict:
    return {
        "id": "semantic-episode:pointer",
        "goal": "Fix Widget.render",
        "metadata": {
            "retrieval": {
                "goal": "Fix Widget.render",
                "queries": ["Widget.render"],
                "candidates": [
                    {
                        "concept_id": "method://pkg/widget.py:Widget.render",
                        "path": "pkg/widget.py",
                        "kind": "method",
                        "label": "Widget.render",
                        "score": 10.0,
                        "evidence": ["exact:Widget.render"],
                    },
                    {
                        "concept_id": "test://tests/test_widget.py:test_render",
                        "path": "tests/test_widget.py",
                        "kind": "function",
                        "label": "test_render",
                        "score": 5.0,
                        "evidence": [],
                    },
                ],
            }
        },
        "steps": [
            {
                "index": 1,
                "action": {"name": "repo.read", "inputs": {"path": "pkg/widget.py"}},
                "observation": {},
                "terminal": False,
            }
        ],
        "schema_version": "lcfa.semantic-trajectory.v1",
    }


def test_episode_transitions_keep_aligned_semantic_candidate_ids() -> None:
    row = episode_to_transitions(_episode())[0]
    assert row.candidate_paths == ("pkg/widget.py", "tests/test_widget.py")
    assert row.candidate_entities == (
        "method://pkg/widget.py:Widget.render",
        "test://tests/test_widget.py:test_render",
    )
    assert row.target_pointer == 0
    assert row.event["retrieval"]["entities"][0] == "method://pkg/widget.py:Widget.render"


def test_v4_round_trip_and_v3_compatibility(tmp_path: Path) -> None:
    row = episode_to_transitions(_episode())[0]
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row.to_dict()) + "\n", encoding="utf-8")
    loaded = load_transitions(path)[0]
    assert loaded.candidate_entities == row.candidate_entities

    legacy = tmp_path / "legacy.jsonl"
    payload = dict(row.to_dict())
    payload["schema_version"] = "lcfa.recurrent-transition.v3"
    payload.pop("candidate_entities", None)
    legacy.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    legacy_row = load_transitions(legacy)[0]
    assert legacy_row.candidate_entities == ()
    assert RECURRENT_TRANSITION_FORMAT == "lcfa.recurrent-transition.v4"
