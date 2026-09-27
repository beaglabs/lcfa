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


def test_episode_transitions_keep_aligned_semantic_candidate_priors() -> None:
    row = episode_to_transitions(_episode())[0]
    assert row.candidate_paths == ("pkg/widget.py", "tests/test_widget.py")
    assert row.candidate_entities == (
        "method://pkg/widget.py:Widget.render",
        "test://tests/test_widget.py:test_render",
    )
    assert row.candidate_scores == (10.0, 5.0)
    assert row.candidate_evidence == (("exact:Widget.render",), ())
    assert row.target_pointer == 0
    assert row.event["retrieval"]["entities"][0] == "method://pkg/widget.py:Widget.render"


def test_v5_round_trip_and_v3_v4_compatibility(tmp_path: Path) -> None:
    row = episode_to_transitions(_episode())[0]
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row.to_dict()) + "\n", encoding="utf-8")
    loaded = load_transitions(path)[0]
    assert loaded.candidate_entities == row.candidate_entities
    assert loaded.candidate_scores == row.candidate_scores
    assert loaded.candidate_evidence == row.candidate_evidence

    legacy_v4 = tmp_path / "legacy-v4.jsonl"
    v4_payload = dict(row.to_dict())
    v4_payload["schema_version"] = "lcfa.recurrent-transition.v4"
    v4_payload.pop("candidate_scores", None)
    v4_payload.pop("candidate_evidence", None)
    legacy_v4.write_text(json.dumps(v4_payload) + "\n", encoding="utf-8")
    v4_row = load_transitions(legacy_v4)[0]
    assert v4_row.candidate_entities == row.candidate_entities
    assert v4_row.candidate_scores == ()
    assert v4_row.candidate_evidence == ()

    legacy_v3 = tmp_path / "legacy-v3.jsonl"
    v3_payload = dict(v4_payload)
    v3_payload["schema_version"] = "lcfa.recurrent-transition.v3"
    v3_payload.pop("candidate_entities", None)
    legacy_v3.write_text(json.dumps(v3_payload) + "\n", encoding="utf-8")
    v3_row = load_transitions(legacy_v3)[0]
    assert v3_row.candidate_entities == ()
    assert v3_row.candidate_scores == ()
    assert RECURRENT_TRANSITION_FORMAT == "lcfa.recurrent-transition.v5"
