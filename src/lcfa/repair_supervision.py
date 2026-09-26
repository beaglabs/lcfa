"""Historical repair targets used only for supervised edit/correction labels.

Gold commits are intentionally isolated here.  Repository localization never
calls this module; fix refs are used only after base-repository retrieval has
chosen what the agent can observe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping


class RepairSupervisionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RepairTarget:
    path: str
    action: str
    inputs: Mapping[str, Any]
    before_sha256: str
    after_sha256: str

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RepairSupervisionError(
            f"git {' '.join(args)} failed: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _text(repo: Path, ref: str, path: str, *, missing_ok: bool = False) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        if missing_ok:
            return ""
        raise RepairSupervisionError(
            f"git show {ref}:{path} failed: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairSupervisionError(f"repair target is not UTF-8 text: {path}") from exc


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _minimal_span(before: str, after: str) -> tuple[str, str] | None:
    """Return one exact old/new span covering all changed lines when safe."""
    if before == after:
        return None
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    prefix = 0
    shared = min(len(before_lines), len(after_lines))
    while prefix < shared and before_lines[prefix] == after_lines[prefix]:
        prefix += 1

    suffix = 0
    before_remaining = len(before_lines) - prefix
    after_remaining = len(after_lines) - prefix
    while (
        suffix < before_remaining
        and suffix < after_remaining
        and before_lines[len(before_lines) - 1 - suffix]
        == after_lines[len(after_lines) - 1 - suffix]
    ):
        suffix += 1

    before_end = len(before_lines) - suffix if suffix else len(before_lines)
    after_end = len(after_lines) - suffix if suffix else len(after_lines)
    old = "".join(before_lines[prefix:before_end])
    new = "".join(after_lines[prefix:after_end])
    if not old or before.count(old) != 1:
        return None
    return old, new


def historical_repair_targets(
    repo: str | Path,
    base_ref: str,
    fix_ref: str,
) -> tuple[RepairTarget, ...]:
    root = Path(repo).expanduser().resolve()
    status = _git(root, "diff", "--name-status", base_ref, fix_ref, "--").decode(
        "utf-8", errors="replace"
    )
    targets: list[RepairTarget] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        change = parts[0]
        if change not in {"A", "M"} or len(parts) != 2:
            raise RepairSupervisionError(
                "historical supervision currently supports added/modified UTF-8 files only; "
                f"unsupported diff entry: {line}"
            )
        path = parts[1]
        before = _text(root, base_ref, path, missing_ok=(change == "A"))
        after = _text(root, fix_ref, path)
        span = _minimal_span(before, after)
        if span is not None:
            old, new = span
            action = "repo.replace"
            inputs: Mapping[str, Any] = {"path": path, "old": old, "new": new}
        else:
            action = "repo.edit"
            inputs = {"path": path, "content": after}
        targets.append(RepairTarget(
            path=path,
            action=action,
            inputs=inputs,
            before_sha256=_hash(before),
            after_sha256=_hash(after),
        ))
    if not targets:
        raise RepairSupervisionError("fix ref contains no supported text repair targets")
    return tuple(targets)


__all__ = [
    "RepairSupervisionError",
    "RepairTarget",
    "historical_repair_targets",
]
