"""Verifier-backed corrective supervision for failed recurrent rollouts.

Failed actions are never copied back into the imitation target.  Historical
fix refs are used only to determine the corrective edit label after the live
rollout has produced an off-policy state. Retrieval candidates remain the ones
computed from the base repository during rollout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .recurrent_collect import RecurrentCollectionTask, load_tasks
from .recurrent_transitions import (
    RecurrentTransition,
    dump_transitions,
    episode_to_transitions,
    load_episode,
)
from .repair_supervision import historical_repair_targets
from .repo_retrieval import pointer_for_action, retrieval_from_mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_map(tasks_path: str | Path) -> Mapping[str, RecurrentCollectionTask]:
    return {task.id: task for task in load_tasks(tasks_path)}


def _action(step: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = step.get("action")
    return raw if isinstance(raw, Mapping) else None


def corrective_transitions_for_episode(
    episode: Mapping[str, Any],
    task: RecurrentCollectionTask,
) -> tuple[RecurrentTransition, ...]:
    """Relabel every visited failed state with the next verified repair action."""
    if bool(episode.get("success")):
        return episode_to_transitions(episode)
    if not task.fix_ref:
        return ()

    metadata = _mapping(episode.get("metadata"))
    retrieval = retrieval_from_mapping(_mapping(metadata.get("retrieval")))
    if retrieval is None:
        return ()
    base_ref = str(metadata.get("base_commit") or task.base_ref)
    fix_ref = str(metadata.get("fix_commit") or task.fix_ref)
    repairs = historical_repair_targets(task.repo, base_ref, fix_ref)
    repair_by_path = {repair.path: repair for repair in repairs}
    ranked_gold_paths = [
        path for path in retrieval.candidate_paths if path in repair_by_path
    ]

    source_rows = episode_to_transitions(episode)
    steps_raw = episode.get("steps", ())
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)):
        return ()

    seen_search = False
    read_paths: set[str] = set()
    edited_paths: set[str] = set()
    rows: list[RecurrentTransition] = []

    for source_row, raw_step in zip(source_rows, steps_raw):
        # Choose the correction for the state *before* the rollout action in
        # raw_step executes. This is DAgger-style off-policy state relabeling.
        target_action: str
        target_inputs: Mapping[str, Any]
        retrieval_miss = False
        if not seen_search:
            target_action = "repo.search"
            target_inputs = {
                "query": retrieval.queries[0] if retrieval.queries else task.goal
            }
        else:
            unread = [path for path in ranked_gold_paths if path not in read_paths]
            if unread:
                target_action = "repo.read"
                target_inputs = {"path": unread[0]}
            else:
                pending_repairs = [
                    repair for repair in repairs if repair.path not in edited_paths
                ]
                if pending_repairs and all(
                    repair.path in retrieval.candidate_paths
                    or not (Path(task.repo) / repair.path).exists()
                    for repair in pending_repairs
                ):
                    repair = pending_repairs[0]
                    target_action = repair.action
                    target_inputs = dict(repair.inputs)
                elif pending_repairs:
                    # The base retriever failed to expose the gold location.
                    # Do not leak that path into the recurrent candidate state.
                    target_action = "repo.search"
                    target_inputs = {
                        "query": retrieval.queries[0] if retrieval.queries else task.goal
                    }
                    retrieval_miss = True
                else:
                    target_action = "verify.run"
                    target_inputs = {}

        rows.append(RecurrentTransition(
            episode_id=f"{source_row.episode_id}:corrective",
            step_index=source_row.step_index,
            goal=source_row.goal,
            event=source_row.event,
            target_action=target_action,
            stop_target=False,
            value_target=1.0,
            metadata={
                **dict(source_row.metadata or {}),
                "corrective": True,
                "source_success": False,
                "retrieval_miss": retrieval_miss,
                "task_id": task.id,
            },
            target_inputs=dict(target_inputs),
            target_pointer=pointer_for_action(target_action, target_inputs, retrieval),
            candidate_queries=source_row.candidate_queries,
            candidate_paths=source_row.candidate_paths,
        ))

        executed = _action(_mapping(raw_step))
        if executed is not None:
            name = str(executed.get("name") or "")
            inputs = _mapping(executed.get("inputs"))
            if name == "repo.search":
                seen_search = True
            elif name == "repo.read" and inputs.get("path"):
                read_paths.add(str(inputs["path"]))
            elif name in {"repo.replace", "repo.edit"} and inputs.get("path"):
                edited_paths.add(str(inputs["path"]))

    return tuple(rows)


def prepare_corrective_transition_file(
    episodes: Sequence[str | Path],
    tasks_path: str | Path,
    output: str | Path,
) -> int:
    tasks = _task_map(tasks_path)
    rows: list[RecurrentTransition] = []
    for raw_path in episodes:
        path = Path(raw_path)
        children = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for child in children:
            episode = load_episode(child)
            if str(episode.get("schema_version", "")) != "lcfa.semantic-trajectory.v1":
                continue
            metadata = _mapping(episode.get("metadata"))
            task_id = str(metadata.get("task_id") or "")
            task = tasks.get(task_id)
            if task is None:
                continue
            if bool(episode.get("success")):
                continue
            rows.extend(corrective_transitions_for_episode(episode, task))
    return dump_transitions(rows, output)


def prepare_successful_rollout_transition_file(
    episodes: Sequence[str | Path],
    output: str | Path,
) -> int:
    """Imitate only verified successful self-rollouts, never failed behavior."""
    rows: list[RecurrentTransition] = []
    for raw_path in episodes:
        path = Path(raw_path)
        children = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for child in children:
            episode = load_episode(child)
            if str(episode.get("schema_version", "")) != "lcfa.semantic-trajectory.v1":
                continue
            if not bool(episode.get("success")):
                continue
            rows.extend(episode_to_transitions(episode))
    return dump_transitions(rows, output)


__all__ = [
    "corrective_transitions_for_episode",
    "prepare_corrective_transition_file",
    "prepare_successful_rollout_transition_file",
]
