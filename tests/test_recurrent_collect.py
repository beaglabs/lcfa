from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from lcfa.recurrent_collect import collect_trajectories, load_tasks, task_from_dict


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _make_repo(root: Path) -> tuple[Path, str, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "lcfa@example.invalid")
    _git(repo, "config", "user.name", "LCFA Test")
    (repo / "value.py").write_text("VALUE = 'broken'\n", encoding="utf-8")
    _git(repo, "add", "value.py")
    _git(repo, "commit", "-m", "buggy")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "value.py").write_text("VALUE = 'fixed'\n", encoding="utf-8")
    _git(repo, "add", "value.py")
    _git(repo, "commit", "-m", "fix")
    fix = _git(repo, "rev-parse", "HEAD")
    return repo, base, fix


def test_task_schema_requires_argv_arrays() -> None:
    with pytest.raises(ValueError, match="verify_argv must be an argv array"):
        task_from_dict({
            "id": "bad",
            "repo": ".",
            "goal": "do something",
            "verify_argv": "pytest -q",
        })


def test_load_tasks_reports_missing_file_cleanly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="task file does not exist"):
        load_tasks(tmp_path / "missing.jsonl")


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    row = {"id": "same", "repo": ".", "goal": "x", "verify_argv": ["true"]}
    tasks.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(tasks)


def test_oracle_collection_replays_known_fix_without_model_teacher(tmp_path: Path) -> None:
    repo, base, fix = _make_repo(tmp_path)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({
            "schema_version": "lcfa.recurrent-task.v2",
            "id": "known-fix",
            "repo": str(repo),
            "base_ref": base,
            "fix_ref": fix,
            "goal": "Fix VALUE so it equals fixed",
            "verify_argv": [
                sys.executable,
                "-c",
                "from value import VALUE; assert VALUE == 'fixed'",
            ],
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "episodes"
    worktrees = tmp_path / "worktrees"
    summary = collect_trajectories(
        tasks,
        mode="oracle",
        output_dir=output,
        worktree_root=worktrees,
    )
    assert summary["mode"] == "oracle"
    assert summary["tasks"] == 1
    assert summary["collected"] == 1
    assert summary["successful"] == 1
    assert summary["errors"] == 0
    assert summary["runtime"] is None
    assert not (worktrees / "known-fix").exists()

    episode = json.loads((output / "known-fix.json").read_text(encoding="utf-8"))
    assert episode["success"] is True
    assert episode["metadata"]["collection_mode"] == "oracle"
    actions = [
        step["action"]["name"] if step.get("action") else "stop"
        for step in episode["steps"]
    ]
    assert actions[0] == "repo.search"
    assert "repo.read" in actions
    assert "repo.edit" in actions
    assert "verify.run" in actions
    assert actions[-1] == "stop"
    assert "VALUE = 'fixed'" in episode["patch"]


def test_oracle_collection_skips_already_passing_task(tmp_path: Path) -> None:
    repo, base, fix = _make_repo(tmp_path)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({
            "schema_version": "lcfa.recurrent-task.v2",
            "id": "already-passes",
            "repo": str(repo),
            "base_ref": base,
            "fix_ref": fix,
            "goal": "Pretend value.py is broken",
            "verify_argv": [sys.executable, "-c", "raise SystemExit(0)"],
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "episodes"
    worktrees = tmp_path / "worktrees"
    summary = collect_trajectories(
        tasks,
        mode="oracle",
        output_dir=output,
        worktree_root=worktrees,
    )
    assert summary["collected"] == 0
    assert summary["skipped_baseline_pass"] == 1
    assert summary["errors"] == 0
    assert summary["results"][0]["status"] == "baseline-already-passes"
    assert not (worktrees / "already-passes").exists()
    assert (output / "collection.json").exists()


def test_rollout_collection_requires_controller(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({
            "id": "x",
            "repo": ".",
            "goal": "x",
            "verify_argv": ["true"],
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="rollout mode requires --controller"):
        collect_trajectories(
            tasks,
            mode="rollout",
            output_dir=tmp_path / "episodes",
            worktree_root=tmp_path / "worktrees",
        )
