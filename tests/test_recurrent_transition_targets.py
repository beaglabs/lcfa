from __future__ import annotations

from lcfa.recurrent_transitions import episode_to_transitions


def test_transition_targets_preserve_arguments_and_pointer_labels() -> None:
    episode = {
        "schema_version": "lcfa.semantic-trajectory.v1",
        "id": "semantic-episode:test",
        "goal": "Fix trust_remote_code in loader.py",
        "success": True,
        "metadata": {
            "retrieval": {
                "schema_version": "lcfa.repo-retrieval.v1",
                "goal": "Fix trust_remote_code in loader.py",
                "queries": ["trust_remote_code", "loader"],
                "candidates": [
                    {
                        "concept_id": "symbol://repo/loader:load",
                        "path": "loader.py",
                        "kind": "function",
                        "label": "load",
                        "score": 12.0,
                        "evidence": ["exact:trust_remote_code:references"],
                    }
                ],
            }
        },
        "steps": [
            {
                "index": 1,
                "solution_id": "s1",
                "hypothesis": None,
                "action": {"name": "repo.search", "inputs": {"query": "trust_remote_code"}},
                "observation": {"observations": {"search": {"query": "trust_remote_code", "hits": []}}},
                "terminal": False,
            },
            {
                "index": 2,
                "solution_id": "s2",
                "hypothesis": None,
                "action": {"name": "repo.read", "inputs": {"path": "loader.py"}},
                "observation": {"observations": {"file": {"path": "loader.py", "text": "broken"}}},
                "terminal": False,
            },
            {
                "index": 3,
                "solution_id": "s3",
                "hypothesis": None,
                "action": {
                    "name": "repo.replace",
                    "inputs": {"path": "loader.py", "old": "broken", "new": "fixed"},
                },
                "observation": {"observations": {"edit": {"path": "loader.py", "replacements": 1}}},
                "terminal": False,
            },
            {
                "index": 4,
                "solution_id": "s4",
                "hypothesis": None,
                "action": None,
                "observation": {},
                "terminal": True,
            },
        ],
        "final_solution_id": "s4",
        "patch": "",
    }

    rows = episode_to_transitions(episode)
    assert len(rows) == 4
    assert rows[0].event["retrieval"]["queries"][0] == "trust_remote_code"
    assert rows[0].event["retrieval"]["paths"] == ["loader.py"]
    assert rows[0].target_pointer == 0
    assert rows[1].target_pointer == 0
    assert rows[2].target_pointer == 0
    assert rows[2].target_inputs == {
        "path": "loader.py",
        "old": "broken",
        "new": "fixed",
    }
    # The next recurrent event fingerprints the write rather than replaying
    # source text, while the separate supervised target keeps the exact span.
    assert "old" not in rows[3].event["action"]["inputs"]
    assert "new" not in rows[3].event["action"]["inputs"]
    assert "old_sha256" in rows[3].event["action"]["inputs"]
    assert rows[3].candidate_paths == ("loader.py",)
