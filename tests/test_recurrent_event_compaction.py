from __future__ import annotations

from lcfa.recurrent_transitions import normalize_event


def test_repo_edit_history_fingerprints_content_without_replaying_source() -> None:
    content = "VALUE = 'fixed'\n" * 5000
    event = normalize_event({
        "kind": "transition",
        "action": {
            "name": "repo.edit",
            "inputs": {"path": "pkg/models.py", "content": content},
        },
        "observation": {
            "edit": {"path": "pkg/models.py", "bytes_after": len(content.encode())}
        },
    })
    inputs = event["action"]["inputs"]
    assert inputs["path"] == "pkg/models.py"
    assert "content" not in inputs
    assert inputs["content_bytes"] == len(content.encode())
    assert len(inputs["content_sha256"]) == 64
    assert event["observation"]["edit"]["path"] == "pkg/models.py"


def test_repo_replace_history_fingerprints_old_and_new_text() -> None:
    event = normalize_event({
        "kind": "transition",
        "action": {
            "name": "repo.replace",
            "inputs": {
                "path": "pkg/models.py",
                "old": "return value.strip()",
                "new": "return value.strip() if value is not None else None",
            },
        },
        "observation": {"edit": {"replacements": 1}},
    })
    inputs = event["action"]["inputs"]
    assert inputs["path"] == "pkg/models.py"
    assert "old" not in inputs
    assert "new" not in inputs
    assert inputs["old_bytes"] > 0
    assert inputs["new_bytes"] > inputs["old_bytes"]
    assert len(inputs["old_sha256"]) == 64
    assert len(inputs["new_sha256"]) == 64


def test_repo_read_history_keeps_read_target_and_observation_evidence() -> None:
    source = "def normalize_name(value):\n    return value\n"
    event = normalize_event({
        "kind": "transition",
        "action": {"name": "repo.read", "inputs": {"path": "pkg/models.py"}},
        "observation": {"source": source},
    })
    assert event["action"] == {
        "name": "repo.read",
        "inputs": {"path": "pkg/models.py"},
    }
    assert event["observation"]["source"] == source
