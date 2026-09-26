from __future__ import annotations

from pathlib import Path
import subprocess

from lcfa.recurrent_collect import load_tasks


def test_bundled_recurrent_starter_tasks_are_valid_and_pass_on_main() -> None:
    root = Path(__file__).resolve().parents[1]
    tasks = load_tasks(root / "data" / "recurrent" / "tasks.jsonl")
    assert len(tasks) == 3
    assert len({task.id for task in tasks}) == 3
    for task in tasks:
        assert task.metadata and task.metadata.get("source") == "lcfa-regression"
        result = subprocess.run(
            list(task.verify_argv),
            cwd=root,
            text=True,
            capture_output=True,
            timeout=task.timeout_seconds,
        )
        assert result.returncode == 0, (
            f"starter verifier failed for {task.id}:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
