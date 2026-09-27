"""Rollout collection for the LCFA hybrid latent + RWKV controller."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .hybrid_semantic_controller import load_hybrid_policy
from .phase_runtime import PhaseMutationRuntimePolicy, attach_controller_trace
from .recurrent_collect import (
    COLLECTION_FORMAT,
    CollectionTaskResult,
    _collect_one_rollout,
    _safe_id,
    load_tasks,
)

HYBRID_COLLECTION_FORMAT = "lcfa.hybrid-recurrent-collection.v1"
ProgressCallback = Callable[[Mapping[str, Any]], None]


def collect_hybrid_trajectories(
    tasks_path: str | Path,
    *,
    controller: str | Path,
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
    progress: ProgressCallback | None = None,
) -> Mapping[str, Any]:
    tasks = list(load_tasks(tasks_path))
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    safe_ids = [_safe_id(task.id) for task in tasks]
    if len(set(safe_ids)) != len(safe_ids):
        raise ValueError("task ids collide after filesystem-safe normalization")

    output = Path(output_dir).expanduser().resolve()
    worktrees = Path(worktree_root).expanduser().resolve()
    controller_path = Path(controller).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    worktrees.mkdir(parents=True, exist_ok=True)
    policy = PhaseMutationRuntimePolicy(
        load_hybrid_policy(
            controller_path,
            model_id=model_id,
            device=device,
            dtype=dtype,
        )
    )

    results: list[CollectionTaskResult] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if progress is not None:
            progress({
                "event": "task-start",
                "index": index,
                "total": total,
                "task_id": task.id,
                "mode": "hybrid-rollout",
            })
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
        trace = policy.consume_controller_trace()
        if result.episode_path and trace:
            attach_controller_trace(result.episode_path, trace)
        results.append(result)
        if progress is not None:
            progress({
                "event": "task-done",
                "index": index,
                "total": total,
                "task_id": task.id,
                "mode": "hybrid-rollout",
                "status": result.status,
                "success": result.success,
                "steps": result.steps,
                "elapsed_seconds": result.elapsed_seconds,
            })

    collected = [item for item in results if item.status == "collected"]
    successful = [item for item in collected if item.success]
    error_statuses = {
        "error", "setup-failed", "missing-fix-ref", "oracle-verification-failed",
    }
    summary = {
        "format": HYBRID_COLLECTION_FORMAT,
        "episode_collection_format": COLLECTION_FORMAT,
        "mode": "hybrid-rollout",
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
        "controller": str(controller_path),
        "runtime": dict(policy.metadata),
        "results": [asdict(item) for item in results],
    }
    (output / "collection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["HYBRID_COLLECTION_FORMAT", "collect_hybrid_trajectories"]
