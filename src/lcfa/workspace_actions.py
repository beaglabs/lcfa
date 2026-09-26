"""Governed repository, terminal, test, git, and documentation actions."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .protocol import ActionGraph, ActionNode, ActionResult, ExecutionContext, SolutionState
from .registry import ActionRegistry, ActionSpec


class WorkspaceActionError(RuntimeError):
    pass


def _root(context: ExecutionContext) -> Path:
    raw = context.metadata.get("workspace_root")
    if not raw:
        raise WorkspaceActionError("workspace_root is required in ExecutionContext.metadata")
    root = Path(str(raw)).resolve()
    if not root.exists() or not root.is_dir():
        raise WorkspaceActionError(f"workspace_root is not a directory: {root}")
    return root


def _within(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkspaceActionError(f"path escapes workspace: {path}") from exc
    return path


def _run(root: Path, argv: list[str], *, cwd: str | Path | None = None, timeout: int = 120) -> ActionResult:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise WorkspaceActionError("argv must be a non-empty list of strings")
    workdir = _within(root, cwd or ".")
    proc = subprocess.run(
        argv, cwd=workdir, text=True, capture_output=True,
        timeout=max(1, min(int(timeout), 1800)), check=False,
    )
    observation = {
        "argv": argv, "cwd": str(workdir), "exit_code": proc.returncode,
        "stdout": proc.stdout, "stderr": proc.stderr,
    }
    return ActionResult(value=observation, observations={"process": observation})


def _repo_read(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    root = _root(context)
    path = _within(root, str(inputs.get("path", "")))
    if not path.is_file():
        raise WorkspaceActionError(f"not a file: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    start = max(1, int(inputs.get("start_line", 1) or 1))
    end_raw = inputs.get("end_line")
    lines = text.splitlines()
    end = len(lines) if end_raw is None else min(len(lines), int(end_raw))
    selected = "\n".join(lines[start - 1:end])
    value = {"path": str(path.relative_to(root)), "start_line": start, "end_line": end, "text": selected}
    return ActionResult(value=value, observations={"file": value})


def _repo_search(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    root = _root(context)
    query = str(inputs.get("query", "")).strip()
    if not query:
        raise WorkspaceActionError("repo.search requires query")
    limit = max(1, min(int(inputs.get("limit", 50) or 50), 500))
    hits = []
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
    for path in root.rglob("*"):
        if len(hits) >= limit:
            break
        if not path.is_file() or any(part in skip for part in path.relative_to(root).parts):
            continue
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if query.lower() in line.lower():
                    hits.append({"path": str(path.relative_to(root)), "line": number, "text": line[:1000]})
                    if len(hits) >= limit:
                        break
        except OSError:
            continue
    value = {"query": query, "hits": hits}
    return ActionResult(value=value, observations={"search": value})


def _repo_edit(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    if "workspace.write" not in context.capabilities:
        raise WorkspaceActionError("repo.edit requires workspace.write capability")
    root = _root(context)
    path = _within(root, str(inputs.get("path", "")))
    content = inputs.get("content")
    if not isinstance(content, str):
        raise WorkspaceActionError("repo.edit requires string content")
    before = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    value = {"path": str(path.relative_to(root)), "bytes_before": len(before.encode()), "bytes_after": len(content.encode())}
    return ActionResult(value=value, observations={"edit": value})


def _process_exec(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    if "process.exec" not in context.capabilities:
        raise WorkspaceActionError("process.exec requires process.exec capability")
    raw = inputs.get("argv")
    if not isinstance(raw, (list, tuple)):
        raise WorkspaceActionError("process.exec requires argv array; shell strings are intentionally unsupported")
    return _run(_root(context), [str(item) for item in raw], cwd=inputs.get("cwd"), timeout=int(inputs.get("timeout", 120) or 120))


def _test_run(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    if "process.exec" not in context.capabilities:
        raise WorkspaceActionError("test.run requires process.exec capability")
    target = str(inputs.get("target", "")).strip()
    argv = ["python", "-m", "pytest"]
    if target:
        argv.append(target)
    argv.extend([str(item) for item in inputs.get("args", [])] if isinstance(inputs.get("args", []), (list, tuple)) else [])
    return _run(_root(context), argv, cwd=inputs.get("cwd"), timeout=int(inputs.get("timeout", 600) or 600))


def _git_status(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    del inputs
    return _run(_root(context), ["git", "status", "--short"], timeout=30)


def _git_diff(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    argv = ["git", "diff", "--no-ext-diff"]
    if inputs.get("cached"):
        argv.append("--cached")
    return _run(_root(context), argv, timeout=30)


def _docs_fetch(context: ExecutionContext, inputs: Mapping[str, object]) -> ActionResult:
    if "network.docs" not in context.capabilities:
        raise WorkspaceActionError("docs.fetch requires network.docs capability")
    url = str(inputs.get("url", "")).strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise WorkspaceActionError("docs.fetch permits HTTPS URLs only")
    allow = {str(item).lower() for item in context.metadata.get("docs_allow_domains", ())}
    host = parsed.hostname.lower()
    if not allow or not any(host == domain or host.endswith("." + domain) for domain in allow):
        raise WorkspaceActionError(f"documentation host is not allowlisted: {host}")
    forbidden = tuple(str(item).lower() for item in context.metadata.get(
        "docs_forbidden_substrings", ("swe-bench", "/pull/", "/commit/", "gold.patch", "test.patch")
    ))
    if any(item and item in url.lower() for item in forbidden):
        raise WorkspaceActionError("documentation URL blocked by benchmark contamination policy")
    request = Request(url, headers={"User-Agent": "LCFA-Semantic-Runtime/0.1"})
    with urlopen(request, timeout=max(1, min(int(inputs.get("timeout", 15) or 15), 60))) as response:
        max_bytes = max(1024, min(int(inputs.get("max_bytes", 2_000_000) or 2_000_000), 8_000_000))
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise WorkspaceActionError("documentation response exceeds max_bytes")
        charset = response.headers.get_content_charset() or "utf-8"
        text = data.decode(charset, errors="replace")
    value = {"url": url, "host": host, "content_type": response.headers.get_content_type(), "text": text}
    return ActionResult(value=value, observations={"documentation": value})


def register_workspace_actions(registry: ActionRegistry | None = None) -> ActionRegistry:
    registry = registry or ActionRegistry()
    specs = (
        ActionSpec("repo.read", _repo_read, description="Read a bounded file range from the workspace."),
        ActionSpec("repo.search", _repo_search, description="Search workspace text without invoking a shell."),
        ActionSpec("repo.edit", _repo_edit, effects=("filesystem.write",), description="Replace one workspace file."),
        ActionSpec("process.exec", _process_exec, effects=("process.exec",), description="Execute argv directly without a shell."),
        ActionSpec("test.run", _test_run, effects=("process.exec", "filesystem.write"), description="Run pytest in the workspace."),
        ActionSpec("git.status", _git_status, description="Read repository status."),
        ActionSpec("git.diff", _git_diff, description="Read repository diff."),
        ActionSpec("docs.fetch", _docs_fetch, effects=("network.read",), description="Fetch allowlisted versioned documentation."),
    )
    for spec in specs:
        registry.register(spec)
    return registry


_CAPABILITIES = {
    "repo.read": ("workspace.read",),
    "repo.search": ("workspace.read",),
    "git.status": ("workspace.read",),
    "git.diff": ("workspace.read",),
    "repo.edit": ("workspace.write",),
    "process.exec": ("process.exec",),
    "test.run": ("process.exec",),
    "docs.fetch": ("network.docs",),
}


def compile_cognitive_actions(solution: SolutionState) -> ActionGraph:
    cognition = solution.values.get("cognition", {}) if isinstance(solution.values, Mapping) else {}
    raw_actions = cognition.get("next_actions", ()) if isinstance(cognition, Mapping) else ()
    nodes = []
    for index, item in enumerate(raw_actions):
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("action", ""))
        if not action:
            continue
        requires_approval = action in {"repo.edit", "process.exec"}
        node_id = f"semantic-action-{index + 1}"
        nodes.append(ActionNode(
            id=node_id,
            action=action,
            inputs=dict(item.get("inputs", {})) if isinstance(item.get("inputs", {}), Mapping) else {},
            effects=(),
            required_capabilities=_CAPABILITIES.get(action, ()),
            requires_approval=requires_approval,
            approval_key=(f"approve:{node_id}" if requires_approval else None),
            idempotency_key=f"{solution.id}:{index}:{action}",
        ))
    return ActionGraph(
        id=f"action-graph:{uuid4()}", source_solution_id=solution.id, nodes=tuple(nodes),
        metadata={"source": "semantic-runtime", "schema": "lcfa.semantic-actions.v1"},
    )


__all__ = ["WorkspaceActionError", "register_workspace_actions", "compile_cognitive_actions"]
