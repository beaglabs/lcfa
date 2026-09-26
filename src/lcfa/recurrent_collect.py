"""Collect real LCFA semantic-agent trajectories in isolated git worktrees.

Task files are JSONL. Each task must provide a local git repository, a goal, and
an argv-style verifier. The collector checks the verifier before the agent,
runs the teacher inside a detached worktree, then runs the verifier again and
attaches the resulting success label to the semantic episode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

from .repo_index import PythonRepoIndexer
from .semantic_agent import SemanticWorkspaceAgent
from .semantic_graph import SQLiteSemanticGraph


COLLECTION_FORMAT = "lcfa.recurrent-collection.v1"
TASK_FORMAT = "lcfa.recurrent-task.v1"


class CollectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecurrentCollectionTask:
    id: str
    repo: str
    goal: str
    verify_argv: tuple[str, ...]
    base_ref: str = "HEAD"
    setup_argv: tuple[str, ...] = ()
    timeout_seconds: int = 300
    metadata: Mapping[str, Any] | None = None
    schema_version: str = TASK_FORMAT


@dataclass(frozen=True, slots=True)
class VerificationResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True, slots=True)
class CollectionTaskResult:
    task_id: str
    status: str
    episode_path: str | None
    success: bool | None
    baseline_passed: bool | None
    patch_bytes: int
    steps: int
    elapsed_seconds: float
    error: str | None = None


def _as_argv(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an argv array")
    argv = tuple(str(item) for item in value)
    if not argv:
        raise ValueError(f"{field} must not be empty")
    return argv


def task_from_dict(raw: Mapping[str, Any]) -> RecurrentCollectionTask:
    if raw.get("schema_version") not in (None, TASK_FORMAT):
        raise ValueError(f"unsupported task schema: {raw.get('schema_version')}")
    task_id = str(raw.get("id") or "").strip()
    repo = str(raw.get("repo") or "").strip()
    goal = str(raw.get("goal") or "").strip()
    if not task_id or not repo or not goal:
        raise ValueError("task requires id, repo, and goal")
    verify_argv = _as_argv(raw.get("verify_argv"), field="verify_argv")
    setup_raw = raw.get("setup_argv")
    setup_argv = () if setup_raw in (None, []) else _as_argv(setup_raw, field="setup_argv")
    timeout_seconds = max(1, int(raw.get("timeout_seconds", 300)))
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
    return RecurrentCollectionTask(
        id=task_id,
        repo=repo,
        goal=goal,
        verify_argv=verify_argv,
        base_ref=str(raw.get("base_ref") or "HEAD"),
        setup_argv=setup_argv,
        timeout_seconds=timeout_seconds,
        metadata=dict(metadata),
    )


def load_tasks(path: str | Path) -> tuple[RecurrentCollectionTask, ...]:
    tasks: list[RecurrentCollectionTask] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, Mapping):
                raise ValueError(f"task line {line_number} is not an object")
            task = task_from_dict(value)
            if task.id in seen:
                raise ValueError(f"duplicate task id: {task.id}")
            seen.add(task.id)
            tasks.append(task)
    return tuple(tasks)


def _git(repo: Path, *args: str, timeout: int = 60) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise CollectionError(
            f"git {' '.join(args)} failed in {repo}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _verify_repo(repo: Path) -> None:
    if not repo.exists() or not repo.is_dir():
        raise CollectionError(f"repository does not exist: {repo}")
    if _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise CollectionError(f"not a git worktree: {repo}")


def _run(argv: Sequence[str], cwd: Path, timeout_seconds: int) -> VerificationResult:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(argv), cwd=cwd, text=True, capture_output=True, timeout=timeout_seconds,
        )
        return VerificationResult(
            tuple(argv), int(result.returncode), result.stdout, result.stderr,
            time.monotonic() - started, False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return VerificationResult(
            tuple(argv), 124, stdout, stderr, time.monotonic() - started, True,
        )


def _clip(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...<truncated>"


def _verification_dict(value: VerificationResult) -> Mapping[str, Any]:
    raw = asdict(value)
    raw["stdout"] = _clip(value.stdout)
    raw["stderr"] = _clip(value.stderr)
    raw["passed"] = value.passed
    return raw


def _safe_id(value: str) -> str:
    rendered = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)
    return rendered.strip("-.") or "task"


def _add_worktree(source_repo: Path, worktree: Path, ref: str) -> str:
    if worktree.exists():
        shutil.rmtree(worktree)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    resolved = _git(source_repo, "rev-parse", ref)
    _git(source_repo, "worktree", "add", "--detach", str(worktree), resolved, timeout=120)
    return resolved


def _remove_worktree(source_repo: Path, worktree: Path) -> None:
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=source_repo, text=True, capture_output=True, timeout=60,
        )
    finally:
        if worktree.exists():
            shutil.rmtree(worktree, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"], cwd=source_repo,
            text=True, capture_output=True, timeout=30,
        )


def _collect_one(
    task: RecurrentCollectionTask,
    *,
    artifact: Path,
    output_dir: Path,
    worktree_root: Path,
    max_steps: int,
    allow_docs: bool,
    keep_worktrees: bool,
    require_baseline_failure: bool,
) -> CollectionTaskResult:
    started = time.monotonic()
    source_repo = Path(task.repo).expanduser().resolve()
    _verify_repo(source_repo)
    task_name = _safe_id(task.id)
    worktree = worktree_root / task_name
    episode_path = output_dir / f"{task_name}.json"
    base_commit = _add_worktree(source_repo, worktree, task.base_ref)
    try:
        if task.setup_argv:
            setup = _run(task.setup_argv, worktree, task.timeout_seconds)
            if not setup.passed:
                return CollectionTaskResult(
                    task.id, "setup-failed", None, None, None, 0, 0,
                    time.monotonic() - started,
                    error=f"setup failed: {_clip(setup.stderr or setup.stdout, 2000)}",
                )

        baseline = _run(task.verify_argv, worktree, task.timeout_seconds)
        if require_baseline_failure and baseline.passed:
            return CollectionTaskResult(
                task.id, "baseline-already-passes", None, None, True, 0, 0,
                time.monotonic() - started,
            )

        db_path = worktree / ".lcfa" / "semantic.db"
        with SQLiteSemanticGraph(db_path) as graph:
            PythonRepoIndexer(graph, worktree).index()
            agent = SemanticWorkspaceAgent.from_artifact(
                graph,
                worktree,
                artifact,
                max_steps=max_steps,
                allow_docs=allow_docs,
            )
            episode = agent.run(task.goal, auto_approve=True)

        verification = _run(task.verify_argv, worktree, task.timeout_seconds)
        episode_raw = dict(episode.to_dict())
        episode_raw["success"] = verification.passed
        episode_raw["metadata"] = {
            "collection_format": COLLECTION_FORMAT,
            "task_id": task.id,
            "task_metadata": dict(task.metadata or {}),
            "repo": str(source_repo),
            "base_ref": task.base_ref,
            "base_commit": base_commit,
            "baseline": _verification_dict(baseline),
            "verification": _verification_dict(verification),
            "teacher_artifact": str(artifact),
            "max_steps": max_steps,
            "allow_docs": allow_docs,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        episode_path.write_text(
            json.dumps(episode_raw, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return CollectionTaskResult(
            task.id,
            "collected",
            str(episode_path),
            verification.passed,
            baseline.passed,
            len(episode.patch.encode("utf-8")),
            len(episode.steps),
            time.monotonic() - started,
        )
    except Exception as exc:
        return CollectionTaskResult(
            task.id, "error", None, None, None, 0, 0,
            time.monotonic() - started, error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if not keep_worktrees:
            _remove_worktree(source_repo, worktree)


def collect_trajectories(
    tasks_path: str | Path,
    *,
    artifact: str | Path,
    output_dir: str | Path,
    worktree_root: str | Path,
    max_steps: int = 12,
    allow_docs: bool = False,
    keep_worktrees: bool = False,
    require_baseline_failure: bool = True,
    max_tasks: int | None = None,
) -> Mapping[str, Any]:
    tasks = list(load_tasks(tasks_path))
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    artifact_path = Path(artifact).expanduser().resolve()
    if not artifact_path.exists():
        raise CollectionError(f"teacher artifact does not exist: {artifact_path}")
    output = Path(output_dir).expanduser().resolve()
    worktrees = Path(worktree_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    worktrees.mkdir(parents=True, exist_ok=True)

    results = [
        _collect_one(
            task,
            artifact=artifact_path,
            output_dir=output,
            worktree_root=worktrees,
            max_steps=max_steps,
            allow_docs=allow_docs,
            keep_worktrees=keep_worktrees,
            require_baseline_failure=require_baseline_failure,
        )
        for task in tasks
    ]
    collected = [item for item in results if item.status == "collected"]
    successful = [item for item in collected if item.success]
    summary = {
        "format": COLLECTION_FORMAT,
        "tasks": len(tasks),
        "collected": len(collected),
        "successful": len(successful),
        "failed_repairs": len(collected) - len(successful),
        "skipped_baseline_pass": sum(item.status == "baseline-already-passes" for item in results),
        "errors": sum(item.status in {"error", "setup-failed"} for item in results),
        "success_rate": (len(successful) / len(collected) if collected else None),
        "episode_paths": [item.episode_path for item in collected if item.episode_path],
        "results": [asdict(item) for item in results],
    }
    (output / "collection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "COLLECTION_FORMAT",
    "TASK_FORMAT",
    "CollectionError",
    "CollectionTaskResult",
    "RecurrentCollectionTask",
    "VerificationResult",
    "collect_trajectories",
    "load_tasks",
    "task_from_dict",
]
