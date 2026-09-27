from __future__ import annotations

import json

from lcfa.recurrent_transitions import (
    RECURRENT_TRANSITION_FORMAT,
    episode_to_transitions,
    load_transitions,
)


def _episode() -> dict:
    return {
        "id": "semantic-episode:retrieval-prior",
        "goal": "Fix Widget.render",
        "schema_version": "lcfa.semantic-trajectory.v1",
        "metadata": {
            "retrieval": {
                "goal": "Fix Widget.render",
                "queries": ["Widget.render"],
                "candidates": [
                    {
                        "concept_id": "function://pkg/widget.py:Widget.render",
                        "path": "pkg/widget.py",
                        "kind": "function",
                        "label": "Widget.render",
                        "score": 42.5,
                        "evidence": ["exact:Widget.render", "path-token:widget"],
                    },
                    {
                        "concept_id": "file://tests/test_widget.py",
                        "path": "tests/test_widget.py",
                        "kind": "file",
                        "label": "test_widget.py",
                        "score": 8.0,
                        "evidence": ["lexical:Widget"],
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
            },
            {
                "index": 2,
                "action": None,
                "observation": {},
                "terminal": True,
            },
        ],
    }


def test_v5_transitions_preserve_aligned_retrieval_prior(tmp_path) -> None:
    rows = episode_to_transitions(_episode())
    assert RECURRENT_TRANSITION_FORMAT == "lcfa.recurrent-transition.v5"
    assert rows[0].candidate_paths == ("pkg/widget.py", "tests/test_widget.py")
    assert rows[0].candidate_entities == (
        "function://pkg/widget.py:Widget.render",
        "file://tests/test_widget.py",
    )
    assert rows[0].candidate_scores == (42.5, 8.0)
    assert rows[0].candidate_evidence == (
        ("exact:Widget.render", "path-token:widget"),
        ("lexical:Widget",),
    )
    assert rows[0].target_pointer == 0

    path = tmp_path / "v5.jsonl"
    path.write_text(
        "\n".join(json.dumps(row.to_dict()) for row in rows) + "\n",
        encoding="utf-8",
    )
    loaded = load_transitions(path)
    assert loaded[0].candidate_scores == rows[0].candidate_scores
    assert loaded[0].candidate_evidence == rows[0].candidate_evidence
