"""Verifier-driven closed-loop improvement for the LCFA hybrid controller."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

from .hybrid_collect import collect_hybrid_trajectories
from .hybrid_semantic_train import train_hybrid_controller
from .recurrent_corrections import (
    prepare_corrective_transition_file,
    prepare_successful_rollout_transition_file,
)
from .recurrent_transitions import load_transitions, merge_transition_files

HYBRID_IMPROVEMENT_FORMAT = "lcfa.hybrid-recurrent-improvement.v1"
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _emit(progress: ProgressCallback | None, event: str, **values: Any) -> None:
    if progress is not None:
        progress({"event": event, **values})


def _transition_count(path: Path) -> int:
    return len(load_transitions(path)) if path.is_file() else 0


def improve_hybrid_controller(
    tasks_path: str | Path,
    *,
    base_transitions: str | Path,
    controller: str | Path,
    output_root: str | Path,
    rounds: int = 1,
    model_id: str | None = None,
    device: str | None = "auto",
    dtype: str = "auto",
    max_steps: int = 12,
    allow_docs: bool = False,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 20260925,
    backbone_mode: str = "frozen",
    training_scope: str = "joint",
    backbone_learning_rate: float = 5e-6,
    pointer_loss_weight: float = 0.5,
    plan_loss_weight: float = 0.25,
    argument_loss_weight: float = 0.5,
    max_argument_chars: int = 8192,
    latent_dim: int = 256,
    latent_slots: int = 9,
    min_reasoning_steps: int = 2,
    max_reasoning_steps: int = 6,
    convergence_tolerance: float = 1e-3,
    progress: ProgressCallback | None = None,
) -> Mapping[str, Any]:
    round_count = max(1, int(rounds))
    tasks = Path(tasks_path).expanduser().resolve()
    accumulated = Path(base_transitions).expanduser().resolve()
    current_controller = Path(controller).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not accumulated.is_file():
        raise ValueError(f"base transitions do not exist: {accumulated}")
    if not current_controller.exists():
        raise ValueError(f"controller does not exist: {current_controller}")

    history: list[Mapping[str, Any]] = []
    for round_index in range(1, round_count + 1):
        round_dir = root / f"round-{round_index:03d}"
        episodes = round_dir / "episodes"
        worktrees = round_dir / "worktrees"
        successful_path = round_dir / "successful.jsonl"
        corrective_path = round_dir / "corrective.jsonl"
        merged_path = round_dir / "transitions.jsonl"
        next_controller = round_dir / "controller"
        round_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(episodes, ignore_errors=True)
        shutil.rmtree(worktrees, ignore_errors=True)

        _emit(
            progress,
            "round-start",
            round=round_index,
            rounds=round_count,
            controller=str(current_controller),
            transitions=str(accumulated),
        )
        rollout = collect_hybrid_trajectories(
            tasks,
            controller=current_controller,
            model_id=model_id,
            device=device,
            dtype=dtype,
            output_dir=episodes,
            worktree_root=worktrees,
            max_steps=max_steps,
            allow_docs=allow_docs,
            keep_worktrees=False,
            require_baseline_failure=True,
            progress=(
                (lambda event, r=round_index: _emit(
                    progress, "rollout-progress", round=r, detail=dict(event)
                ))
                if progress is not None else None
            ),
        )

        successful_count = prepare_successful_rollout_transition_file(
            [episodes], successful_path
        )
        corrective_count = prepare_corrective_transition_file(
            [episodes], tasks, corrective_path
        )
        merge_inputs: list[Path] = [accumulated]
        if successful_count:
            merge_inputs.append(successful_path)
        else:
            successful_path.unlink(missing_ok=True)
        if corrective_count:
            merge_inputs.append(corrective_path)
        else:
            corrective_path.unlink(missing_ok=True)
        merged_count = merge_transition_files(merge_inputs, merged_path)

        _emit(
            progress,
            "round-data",
            round=round_index,
            successful_transitions=successful_count,
            corrective_transitions=corrective_count,
            accumulated_transitions=merged_count,
            rollout_successful=rollout.get("successful"),
            rollout_collected=rollout.get("collected"),
            rollout_errors=rollout.get("errors"),
        )
        resolved_model = str(
            model_id
            or rollout.get("runtime", {}).get("model_id")
            or "RWKV/RWKV7-G1j-1.5B-20260831"
        )
        training = train_hybrid_controller(
            merged_path,
            next_controller,
            model_id=resolved_model,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
            validation_fraction=validation_fraction,
            seed=seed + round_index,
            backbone_mode=backbone_mode,
            training_scope=training_scope,
            backbone_learning_rate=backbone_learning_rate,
            pointer_loss_weight=pointer_loss_weight,
            plan_loss_weight=plan_loss_weight,
            argument_loss_weight=argument_loss_weight,
            max_argument_chars=max_argument_chars,
            latent_dim=latent_dim,
            latent_slots=latent_slots,
            min_reasoning_steps=min_reasoning_steps,
            max_reasoning_steps=max_reasoning_steps,
            convergence_tolerance=convergence_tolerance,
            init_controller=current_controller,
            progress=(
                (lambda event, r=round_index: _emit(
                    progress, "training-progress", round=r, detail=dict(event)
                ))
                if progress is not None else None
            ),
        )
        round_summary = {
            "round": round_index,
            "input_controller": str(current_controller),
            "output_controller": str(next_controller),
            "input_transitions": str(accumulated),
            "output_transitions": str(merged_path),
            "input_transition_count": _transition_count(accumulated),
            "successful_transitions": successful_count,
            "corrective_transitions": corrective_count,
            "output_transition_count": merged_count,
            "rollout": {
                key: rollout.get(key)
                for key in (
                    "tasks", "collected", "successful", "failed_repairs",
                    "errors", "success_rate"
                )
            },
            "training": {
                key: training.get(key)
                for key in (
                    "backbone_mode", "training_scope", "init_controller",
                    "init_head_tensors_loaded", "init_hybrid_tensors_loaded",
                    "init_backbone_loaded", "semantic_pointer_init_tensors_loaded",
                    "semantic_pointer_examples", "pointer_residual_gate",
                    "train_retrieval_prior_accuracy", "validation_retrieval_prior_accuracy",
                    "train_action_accuracy", "train_pointer_accuracy",
                    "validation_action_accuracy", "validation_pointer_accuracy",
                    "validation_exact_episode_accuracy", "argument_examples",
                    "plan_examples", "pointer_examples", "elapsed_seconds"
                )
            },
        }
        (round_dir / "improvement.json").write_text(
            json.dumps(round_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        history.append(round_summary)
        accumulated = merged_path
        current_controller = next_controller
        _emit(progress, "round-done", **round_summary)

    summary = {
        "format": HYBRID_IMPROVEMENT_FORMAT,
        "rounds": round_count,
        "tasks": str(tasks),
        "final_controller": str(current_controller),
        "final_transitions": str(accumulated),
        "final_transition_count": _transition_count(accumulated),
        "history": history,
    }
    (root / "improvement.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["HYBRID_IMPROVEMENT_FORMAT", "improve_hybrid_controller"]
