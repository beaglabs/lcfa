"""Collect LCFA recurrent trajectories without a language-model teacher.

Two collection modes are supported:

``oracle``
    Bootstrap from a historical buggy commit plus a known fixed commit. LCFA
    deterministically replays typed search/read/edit/verify actions and only
    saves the episode when the external verifier passes.

``rollout``
    Let a trained RWKV recurrent controller act in the isolated worktree, then
    attach verifier success/failure as the external value label.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from .engine import LCFA
from .protocol import ActionGraph, ActionNode, ExecutionContext, SolutionState
from .repo_index import PythonRepoIndexer
from .rwkv_controller import RWKVRecurrentPolicy, load_rwkv_policy
from .rwkv_semantic import RWKVSemanticBackbone
from .semantic_agent import SemanticAgentEpisode, SemanticAgentStep, SemanticWorkspaceAgent
from .semantic_graph import SQLiteSemanticGraph
from .workspace_actions import register_workspace_actions


COLLECTION_FORMAT = "lcfa.recurrent-collection.v2"
TASK_FORMAT = "lcfa.recurrent-task.v2"
LEGACY_TASK_FORMAT = "lcfa.recurrent-task.v1"
COLLECTION_MODES = ("oracle", "rollout")


class CollectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecurrentCollectionTask:
    id: str
    repo: str
    goal: str
    verify_argv: tuple[str, ...]
    base_ref: str = "HEAD"
    fix_ref: str | None = None
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


class _RWKVPolicyCache:
    """Load one RWKV model/controller and reset recurrent state per episode."""

    def __init__(
        self,
        controller: Path,
        *,
        model_id: str | None,
        device: str | None,
        dtype: str,
    ) -> None:
        self.controller = controller
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._policy: RWKVRecurrentPolicy | None = None

    def get(self) -> RWKVRecurrentPolicy:
        if self._policy is None:
            self._policy = load_rwkv_policy(
                self.controller,
                model_id=self.model_id,
                device=self.device,
                dtype=self.dtype,
            )
        return self._policy


def _as_argv(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an argv array")
    argv = tuple(str(item) for item in value)
    if not argv:
        raise ValueError(f"{field} must not be empty")
    return argv


def task_from_dict(raw: Mapping[str, Any]) -> RecurrentCollectionTask:
    schema = raw.get("schema_version")
    if schema not in (None, TASK_FORMAT, LEGACY_TASK_FORMAT):
        raise ValueError(f"unsupported task schema: {schema}")
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
    fix_raw = raw.get("fix_ref")
    return RecurrentCollectionTask(
        id=task_id,
        repo=repo,
        goal=goal,
        verify_argv=verify_argv,
        base_ref=str(raw.get("base_ref") or "HEAD"),
        fix_ref=(str(fix_raw).strip() if fix_raw else None),
        setup_argv=setup_argv,
        timeout_seconds=timeout_seconds,
        metadata=dict(metadata),
    )


def load_tasks(path: str | Path) -> tuple[RecurrentCollectionTask, ...]:
    source = Path(path)
    if not source.is_file():
        raise CollectionError(
            f"task file does not exist: {source}; provide a JSONL recurrent task corpus"
        )
    tasks: list[RecurrentCollectionTask] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
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
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise CollectionError(
            f"git {' '.join(args)} failed in {repo}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _git_show(repo: Path, ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise CollectionError(
            f"git show {ref}:{path} failed: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CollectionError(f"oracle trajectory only supports UTF-8 text files: {path}") from exc


def _verify_repo(repo: Path) -> None:
    if not repo.exists() or not repo.is_dir():
        raise CollectionError(f"repository does not exist: {repo}")
    if _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise CollectionError(f"not a git worktree: {repo}")


def _run(argv: Sequence[str], cwd: Path, timeout_seconds: int) -> VerificationResult:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return VerificationResult(
            tuple(argv),
            int(result.returncode),
            result.stdout,
            result.stderr,
            time.monotonic() - started,
            False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return VerificationResult(
            tuple(argv),
            124,
            stdout,
            stderr,
            time.monotonic() - started,
            True,
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
            cwd=source_repo,
            text=True,
            capture_output=True,
            timeout=60,
        )
    finally:
        if worktree.exists():
            shutil.rmtree(worktree, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=source_repo,
            text=True,
            capture_output=True,
            timeout=30,
        )


def _changed_text_paths(repo: Path, base_commit: str, fix_commit: str) -> tuple[str, ...]:
    raw = _git(
        repo,
        "diff",
        "--name-status",
        "--diff-filter=ACM",
        base_commit,
        fix_commit,
        "--",
    )
    paths: list[str] = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] not in {"A", "C", "M"}:
            continue
        paths.append(parts[1])
    return tuple(paths)


def _has_unsupported_oracle_changes(repo: Path, base_commit: str, fix_commit: str) -> bool:
    raw = _git(repo, "diff", "--name-status", base_commit, fix_commit, "--")
    return any(line and line[0] in {"D", "R", "T", "U"} for line in raw.splitlines())


def _oracle_search_query(goal: str, paths: Sequence[str], worktree: Path) -> str:
    stop = {
        "this", "that", "with", "from", "into", "when", "then", "make", "support",
        "enable", "using", "should", "file", "model", "loader", "recurrent", "controller",
    }
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", goal)
    candidates = [item for item in tokens if item.casefold() not in stop]
    for token in candidates:
        needle = token.casefold()
        for raw_path in paths:
            path = worktree / raw_path
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore").casefold()
                if needle in text:
                    return token
    if paths:
        return Path(paths[0]).stem
    return candidates[0] if candidates else goal[:80]


def _oracle_action(
    executor: Any,
    worktree: Path,
    solution: SolutionState,
    *,
    action: str,
    inputs: Mapping[str, Any],
    verify_argv: Sequence[str],
    verify_timeout: int,
) -> Mapping[str, Any]:
    node_id = f"oracle-action-{uuid4().hex[:10]}"
    graph = ActionGraph(
        id=f"oracle-graph:{uuid4()}",
        source_solution_id=solution.id,
        nodes=(ActionNode(id=node_id, action=action, inputs=dict(inputs)),),
    )
    run = executor.execute(
        graph,
        solution,
        ExecutionContext(
            capabilities=frozenset({"workspace.read", "workspace.write", "process.exec"}),
            metadata={
                "workspace_root": str(worktree),
                "verify_argv": list(verify_argv),
                "verify_timeout": verify_timeout,
            },
        ),
    )
    return {
        "graph_id": run.graph_id,
        "results": {key: value.value for key, value in run.results.items()},
        "observations": dict(run.observations),
    }


def _oracle_episode(
    task: RecurrentCollectionTask,
    *,
    source_repo: Path,
    worktree: Path,
    base_commit: str,
    fix_commit: str,
) -> tuple[SemanticAgentEpisode, VerificationResult]:
    if _has_unsupported_oracle_changes(source_repo, base_commit, fix_commit):
        raise CollectionError(
            "oracle bootstrap currently supports added/modified text files only; "
            "delete/rename/type-change diffs must use rollout mode or be normalized first"
        )
    paths = _changed_text_paths(source_repo, base_commit, fix_commit)
    if not paths:
        raise CollectionError("oracle fix_ref contains no added/modified files")

    executor = LCFA(actions=register_workspace_actions()).agentic
    solution = SolutionState(
        id=f"solution:oracle:{_safe_id(task.id)}",
        plan_id="recurrent-oracle",
        values={"cognition": {}},
    )
    steps: list[SemanticAgentStep] = []

    def append(action: str | None, inputs: Mapping[str, Any], observation: Mapping[str, Any], claim: str, *, terminal: bool = False) -> None:
        index = len(steps) + 1
        steps.append(
            SemanticAgentStep(
                index=index,
                solution_id=f"{solution.id}:{index}",
                hypothesis={"claim": claim, "confidence": 1.0},
                action=({"name": action, "inputs": dict(inputs)} if action else None),
                observation=dict(observation),
                terminal=terminal,
            )
        )

    query = _oracle_search_query(task.goal, paths, worktree)
    inputs = {"query": query}
    append(
        "repo.search",
        inputs,
        _oracle_action(
            executor,
            worktree,
            solution,
            action="repo.search",
            inputs=inputs,
            verify_argv=task.verify_argv,
            verify_timeout=task.timeout_seconds,
        ),
        f"Locate repository evidence related to {query}",
    )

    for path in paths:
        existing = worktree / path
        if existing.is_file():
            inputs = {"path": path}
            append(
                "repo.read",
                inputs,
                _oracle_action(
                    executor,
                    worktree,
                    solution,
                    action="repo.read",
                    inputs=inputs,
                    verify_argv=task.verify_argv,
                    verify_timeout=task.timeout_seconds,
                ),
                f"Inspect affected file {path}",
            )

    for path in paths:
        fixed = _git_show(source_repo, fix_commit, path)
        inputs = {"path": path, "content": fixed}
        append(
            "repo.edit",
            inputs,
            _oracle_action(
                executor,
                worktree,
                solution,
                action="repo.edit",
                inputs=inputs,
                verify_argv=task.verify_argv,
                verify_timeout=task.timeout_seconds,
            ),
            f"Apply the externally verified historical repair to {path}",
        )

    verify_inputs: Mapping[str, Any] = {}
    verify_observation = _oracle_action(
        executor,
        worktree,
        solution,
        action="verify.run",
        inputs=verify_inputs,
        verify_argv=task.verify_argv,
        verify_timeout=task.timeout_seconds,
    )
    append(
        "verify.run",
        verify_inputs,
        verify_observation,
        "Run the task verifier after applying the repair",
    )
    verification = _run(task.verify_argv, worktree, task.timeout_seconds)
    append(
        None,
        {},
        {},
        "Stop only after the external verifier confirms the repair",
        terminal=True,
    )
    patch = _git(worktree, "diff", "--no-ext-diff")
    return (
        SemanticAgentEpisode(
            id=f"semantic-episode:oracle:{_safe_id(task.id)}:{uuid4().hex[:12]}",
            goal=task.goal,
            steps=tuple(steps),
            final_solution_id=steps[-1].solution_id,
            patch=patch,
        ),
        verification,
    )


def _collect_one_oracle(
    task: RecurrentCollectionTask,
    *,
    output_dir: Path,
    worktree_root: Path,
    keep_worktrees: bool,
    require_baseline_failure: bool,
) -> CollectionTaskResult:
    started = time.monotonic()
    source_repo = Path(task.repo).expanduser().resolve()
    _verify_repo(source_repo)
    if not task.fix_ref:
        return CollectionTaskResult(
            task.id,
            "missing-fix-ref",
            None,
            None,
            None,
            0,
            0,
            time.monotonic() - started,
            error="oracle mode requires fix_ref",
        )
    task_name = _safe_id(task.id)
    worktree = worktree_root / task_name
    episode_path = output_dir / f"{task_name}.json"
    base_commit = _add_worktree(source_repo, worktree, task.base_ref)
    try:
        fix_commit = _git(source_repo, "rev-parse", task.fix_ref)
        if task.setup_argv:
            setup = _run(task.setup_argv, worktree, task.timeout_seconds)
            if not setup.passed:
                return CollectionTaskResult(
                    task.id,
                    "setup-failed",
                    None,
                    None,
                    None,
                    0,
                    0,
                    time.monotonic() - started,
                    error=f"setup failed: {_clip(setup.stderr or setup.stdout, 2000)}",
                )
        baseline = _run(task.verify_argv, worktree, task.timeout_seconds)
        if require_baseline_failure and baseline.passed:
            return CollectionTaskResult(
                task.id,
                "baseline-already-passes",
                None,
                None,
                True,
                0,
                0,
                time.monotonic() - started,
            )
        episode, verification = _oracle_episode(
            task,
            source_repo=source_repo,
            worktree=worktree,
            base_commit=base_commit,
            fix_commit=fix_commit,
        )
        if not verification.passed:
            return CollectionTaskResult(
                task.id,
                "oracle-verification-failed",
                None,
                False,
                baseline.passed,
                len(episode.patch.encode("utf-8")),
                len(episode.steps),
                time.monotonic() - started,
                error=_clip(verification.stderr or verification.stdout, 2000),
            )
        episode_raw = dict(episode.to_dict())
        episode_raw["success"] = True
        episode_raw["metadata"] = {
            "collection_format": COLLECTION_FORMAT,
            "collection_mode": "oracle",
            "task_id": task.id,
            "task_metadata": dict(task.metadata or {}),
            "repo": str(source_repo),
            "base_ref": task.base_ref,
            "base_commit": base_commit,
            "fix_ref": task.fix_ref,
            "fix_commit": fix_commit,
            "baseline": _verification_dict(baseline),
            "verification": _verification_dict(verification),
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
            True,
            baseline.passed,
            len(episode.patch.encode("utf-8")),
            len(episode.steps),
            time.monotonic() - started,
        )
    except Exception as exc:
        return CollectionTaskResult(
            task.id,
            "error",
            None,
            None,
            None,
            0,
            0,
            time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if not keep_worktrees:
            _remove_worktree(source_repo, worktree)


def _collect_one_rollout(
    task: RecurrentCollectionTask,
    *,
    policy: RWKVRecurrentPolicy,
    controller: Path,
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
                    task.id,
                    "setup-failed",
                    None,
                    None,
                    None,
                    0,
                    0,
                    time.monotonic() - started,
                    error=f"setup failed: {_clip(setup.stderr or setup.stdout, 2000)}",
                )
        baseline = _run(task.verify_argv, worktree, task.timeout_seconds)
        if require_baseline_failure and baseline.passed:
            return CollectionTaskResult(
                task.id,
                "baseline-already-passes",
                None,
                None,
                True,
                0,
                0,
                time.monotonic() - started,
            )

        db_path = worktree / ".lcfa" / "semantic.db"
        with SQLiteSemanticGraph(db_path) as graph:
            PythonRepoIndexer(graph, worktree).index()
            agent = SemanticWorkspaceAgent(
                graph,
                worktree,
                RWKVSemanticBackbone(policy, graph),
                max_steps=max_steps,
                allow_docs=allow_docs,
                verify_argv=task.verify_argv,
                verify_timeout=task.timeout_seconds,
            )
            episode = agent.run(task.goal, auto_approve=True)

        verification = _run(task.verify_argv, worktree, task.timeout_seconds)
        episode_raw = dict(episode.to_dict())
        episode_raw["success"] = verification.passed
        episode_raw["metadata"] = {
            "collection_format": COLLECTION_FORMAT,
            "collection_mode": "rollout",
            "task_id": task.id,
            "task_metadata": dict(task.metadata or {}),
            "repo": str(source_repo),
            "base_ref": task.base_ref,
            "base_commit": base_commit,
            "baseline": _verification_dict(baseline),
            "verification": _verification_dict(verification),
            "controller": str(controller),
            "controller_runtime": dict(policy.metadata),
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
            task.id,
            "error",
            None,
            None,
            None,
            0,
            0,
            time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if not keep_worktrees:
            _remove_worktree(source_repo, worktree)


def collect_trajectories(
    tasks_path: str | Path,
    *,
    mode: str = "oracle",
    controller: str | Path | None = None,
    model_id: str | None = None,
    device: str | None = "auto",
    dtype: str = "auto",
    output_dir: str | Path,
    worktree_root: str | Path,
    max_steps: int = 12,
    allow_docs: bool = False,
    keep_worktrees: bool = False,
    require_baseline_failure: bool = True,
    max_tasks: int | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in COLLECTION_MODES:
        raise ValueError(f"unknown collection mode: {mode}")
    tasks = list(load_tasks(tasks_path))
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    safe_ids = [_safe_id(task.id) for task in tasks]
    if len(set(safe_ids)) != len(safe_ids):
        raise ValueError("task ids collide after filesystem-safe normalization")

    output = Path(output_dir).expanduser().resolve()
    worktrees = Path(worktree_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    worktrees.mkdir(parents=True, exist_ok=True)

    controller_path: Path | None = None
    policy: RWKVRecurrentPolicy | None = None
    if resolved_mode == "rollout":
        if controller is None:
            raise CollectionError("rollout mode requires --controller")
        controller_path = Path(controller).expanduser().resolve()
        cache = _RWKVPolicyCache(
            controller_path,
            model_id=model_id,
            device=device,
            dtype=dtype,
        )
        policy = cache.get()

    results: list[CollectionTaskResult] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if progress is not None:
            progress({
                "event": "task-start",
                "index": index,
                "total": total,
                "task_id": task.id,
                "mode": resolved_mode,
            })
        if resolved_mode == "oracle":
            result = _collect_one_oracle(
                task,
                output_dir=output,
                worktree_root=worktrees,
                keep_worktrees=keep_worktrees,
                require_baseline_failure=require_baseline_failure,
            )
        else:
            assert policy is not None and controller_path is not None
            result = _collect_one_rollout(
                task,
                policy=policy,
                controller=controller_path,
                output_dir=output,
                worktree_root=worktrees,
                max_steps=max_steps,
                allow_docs=allow_docs,
                keep_worktrees=keep_worktrees,
                require_baseline_failure=require_baseline_failure,
            )
        results.append(result)
        if progress is not None:
            progress({
                "event": "task-done",
                "index": index,
                "total": total,
                "task_id": task.id,
                "mode": resolved_mode,
                "status": result.status,
                "success": result.success,
                "steps": result.steps,
                "elapsed_seconds": result.elapsed_seconds,
            })

    collected = [item for item in results if item.status == "collected"]
    successful = [item for item in collected if item.success]
    error_statuses = {
        "error",
        "setup-failed",
        "missing-fix-ref",
        "oracle-verification-failed",
    }
    summary = {
        "format": COLLECTION_FORMAT,
        "mode": resolved_mode,
        "tasks": len(tasks),
        "collected": len(collected),
        "successful": len(successful),
        "failed_repairs": len(collected) - len(successful),
        "skipped_baseline_pass": sum(
            item.status == "baseline-already-passes" for item in results
        ),
        "errors": sum(item.status in error_statuses for item in results),
        "success_rate": (len(successful) / len(collected) if collected else None),
        "episode_paths": [item.episode_path for item in collected if item.episode_path],
        "controller": (str(controller_path) if controller_path else None),
        "runtime": (dict(policy.metadata) if policy is not None else None),
        "results": [asdict(item) for item in results],
    }
    (output / "collection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "COLLECTION_FORMAT",
    "COLLECTION_MODES",
    "TASK_FORMAT",
    "CollectionError",
    "CollectionTaskResult",
    "RecurrentCollectionTask",
    "VerificationResult",
    "collect_trajectories",
    "load_tasks",
    "task_from_dict",
]
