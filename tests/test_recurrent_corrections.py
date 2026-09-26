from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from lcfa.recurrent_collect import task_from_dict
from lcfa.recurrent_corrections import corrective_transitions_for_episode


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_failed_rollout_states_get_corrective_not_failed_action_targets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "lcfa@example.invalid")
    _git(repo, "config", "user.name", "LCFA Test")
    (repo / "value.py").write_text("VALUE = 'broken'\n", encoding="utf-8")
    _git(repo, "add", "value.py")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "value.py").write_text("VALUE = 'fixed'\n", encoding="utf-8")
    _git(repo, "add", "value.py")
    _git(repo, "commit", "-m", "fix")
    fix = _git(repo, "rev-parse", "HEAD")

    task = task_from_dict({
        "id": "repair-value",
        "repo": str(repo),
        "base_ref": base,
        "fix_ref": fix,
        "goal": "Fix VALUE so it equals fixed",
        "verify_argv": [
            sys.executable,
            "-c",
            "from value import VALUE; assert VALUE == 'fixed'",
        ],
    })
    retrieval = {
        "schema_version": "lcfa.repo-retrieval.v1",
        "goal": task.goal,
        "queries": ["VALUE"],
        "candidates": [
            {
                "concept_id": "file://repo/value.py",
                "path": "value.py",
                "kind": "file",
                "label": "value.py",
                "score": 10.0,
                "evidence": ["lexical:VALUE"],
            }
        ],
    }
    episode = {
        "schema_version": "lcfa.semantic-trajectory.v1",
        "id": "semantic-episode:failed",
        "goal": task.goal,
        "success": False,
        "metadata": {
            "task_id": task.id,
            "base_commit": base,
            "fix_commit": fix,
            "retrieval": retrieval,
        },
        "steps": [
            {
                "index": 1,
                "solution_id": "s1",
                "hypothesis": None,
                "action": {"name": "repo.search", "inputs": {"query": "VALUE"}},
                "observation": {"observations": {"search": {"query": "VALUE", "hits": []}}},
                "terminal": False,
            },
            {
                "index": 2,
                "solution_id": "s2",
                "hypothesis": None,
                "action": {"name": "repo.read", "inputs": {"path": "value.py"}},
                "observation": {"observations": {"file": {"path": "value.py", "text": "VALUE = 'broken'"}}},
                "terminal": False,
            },
            {
                "index": 3,
                "solution_id": "s3",
                "hypothesis": None,
                "action": {"name": "verify.run", "inputs": {}},
                "observation": {"observations": {"process": {"exit_code": 1}}},
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

    rows = corrective_transitions_for_episode(episode, task)
    assert [row.target_action for row in rows[:3]] == [
        "repo.search",
        "repo.read",
        "repo.replace",
    ]
    assert rows[2].target_inputs["path"] == "value.py"
    assert rows[2].target_inputs["old"] == "VALUE = 'broken'\n"
    assert rows[2].target_inputs["new"] == "VALUE = 'fixed'\n"
    assert rows[2].target_pointer == 0
    assert rows[2].metadata["corrective"] is True
    assert rows[2].metadata["source_success"] is False
    # Crucially, the failed verify action at step 3 is not copied as the target.
    assert rows[2].target_action != episode["steps"][2]["action"]["name"]
