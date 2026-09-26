from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from lcfa.recurrent_collect import collect_trajectories, load_tasks, task_from_dict


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "lcfa@example.invalid")
    _git(repo, "config", "user.name", "LCFA Test")
    (repo / "ok.txt").write_text("ok\n", encoding="utf-8")
    _git(repo, "add", "ok.txt")
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_task_schema_requires_argv_arrays() -> None:
    with pytest.raises(ValueError, match="verify_argv must be an argv array"):
        task_from_dict({
            "id": "bad",
            "repo": ".",
            "goal": "do something",
            "verify_argv": "pytest -q",
        })


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    row = {"id": "same", "repo": ".", "goal": "x", "verify_argv": ["true"]}
    tasks.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(tasks)


def test_collector_uses_detached_worktree_and_skips_already_passing_task(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    artifact = tmp_path / "dummy-artifact"
    artifact.mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({
            "schema_version": "lcfa.recurrent-task.v1",
            "id": "already-passes",
            "repo": str(repo),
            "base_ref": "HEAD",
            "goal": "Pretend ok.txt is broken",
            "verify_argv": [
                sys.executable,
                "-c",
                "from pathlib import Path; assert Path('ok.txt').read_text().strip() == 'ok'",
            ],
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "episodes"
    worktrees = tmp_path / "worktrees"
    summary = collect_trajectories(
        tasks,
        artifact=artifact,
        output_dir=output,
        worktree_root=worktrees,
        max_steps=2,
    )
    assert summary["tasks"] == 1
    assert summary["collected"] == 0
    assert summary["skipped_baseline_pass"] == 1
    assert summary["errors"] == 0
    assert summary["results"][0]["status"] == "baseline-already-passes"
    assert not (worktrees / "already-passes").exists()
    assert (output / "collection.json").exists()
