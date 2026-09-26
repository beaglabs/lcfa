"""Training for the first LCFA recurrent-controller experiment.

Phase 1 deliberately freezes RWKV-7 and trains only action/stop/value heads.
This isolates whether pretrained recurrent state already contains useful
control information before recurrent-backbone fine-tuning.

Because the backbone is frozen, recurrent hidden states are computed once and
cached. Replaying the 1.5B model on every optimization epoch would be identical
work and is intentionally avoided.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping

from .recurrent_eval import evaluate_loaded_heads, group_episodes, split_transitions
from .recurrent_transitions import ACTION_VOCAB, RecurrentTransition, load_transitions
from .rwkv_controller import DEFAULT_RWKV_MODEL, RWKV_CONTROLLER_FORMAT, RWKVControllerError
from .torch_runtime import resolve_device, resolve_dtype


TRAINING_FORMAT = "lcfa.rwkv-controller-training.v1"
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _event_text(row: RecurrentTransition) -> str:
    return json.dumps(row.event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def _emit(progress: ProgressCallback | None, event: str, **values: Any) -> None:
    if progress is not None:
        progress({"event": event, **values})


def train_rwkv_heads(
    transitions_path: str | Path,
    output_dir: str | Path,
    *,
    model_id: str = DEFAULT_RWKV_MODEL,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device: str | None = "auto",
    dtype: str = "auto",
    validation_fraction: float = 0.2,
    seed: int = 20260925,
    progress: ProgressCallback | None = None,
) -> Mapping[str, Any]:
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RWKVControllerError(
            "RWKV training requires `pip install -e '.[rwkv]'`"
        ) from exc

    started = time.monotonic()
    rows = load_transitions(transitions_path)
    if not rows:
        raise ValueError("transition dataset is empty")
    for row in rows:
        if row.target_action not in ACTION_VOCAB:
            raise ValueError(f"unknown action target: {row.target_action}")
    train_rows, validation_rows = split_transitions(
        rows,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if not train_rows:
        raise ValueError("training split is empty")

    random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = resolve_device(torch, device)
    try:
        resolved_dtype_name, resolved_dtype = resolve_dtype(torch, resolved_device, dtype)
    except ValueError as exc:
        raise RWKVControllerError(str(exc)) from exc

    epoch_count = max(1, int(epochs))
    train_episode_groups = [list(episode) for episode in group_episodes(train_rows)]
    _emit(
        progress,
        "model-load-start",
        model_id=model_id,
        device=resolved_device,
        dtype=resolved_dtype_name,
        transitions=len(rows),
        train_transitions=len(train_rows),
        validation_transitions=len(validation_rows),
        train_episodes=len(train_episode_groups),
        epochs=epoch_count,
    )
    load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=resolved_dtype,
        trust_remote_code=True,
    ).to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _emit(
        progress,
        "model-loaded",
        elapsed_seconds=time.monotonic() - load_started,
        device=resolved_device,
        dtype=resolved_dtype_name,
    )

    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RWKVControllerError("RWKV model config does not expose hidden_size")
    heads = torch.nn.ModuleDict({
        "action": torch.nn.Linear(hidden_size, len(ACTION_VOCAB)),
        "stop": torch.nn.Linear(hidden_size, 1),
        "value": torch.nn.Linear(hidden_size, 1),
    }).to(resolved_device)
    heads.train()
    optimizer = torch.optim.AdamW(heads.parameters(), lr=float(learning_rate))
    action_index = {name: index for index, name in enumerate(ACTION_VOCAB)}

    # Frozen RWKV means these recurrent features are invariant across epochs.
    # Compute each trajectory once, preserving recurrent state within an episode.
    feature_started = time.monotonic()
    _emit(
        progress,
        "feature-cache-start",
        episodes=len(train_episode_groups),
        transitions=len(train_rows),
    )
    cached_episodes: list[list[tuple[RecurrentTransition, Any]]] = []
    cached_tokens = 0
    max_event_tokens = 0
    cached_transitions = 0
    for episode_index, episode in enumerate(train_episode_groups, start=1):
        state: Any = None
        cached_episode: list[tuple[RecurrentTransition, Any]] = []
        for row in episode:
            encoded = tokenizer(_event_text(row), return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(resolved_device)
            token_count = int(input_ids.shape[-1])
            cached_tokens += token_count
            max_event_tokens = max(max_event_tokens, token_count)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "use_cache": True,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if state is not None:
                kwargs["state"] = state
            with torch.inference_mode():
                outputs = model(**kwargs)
            hidden_states = getattr(outputs, "hidden_states", None)
            state = getattr(outputs, "state", None)
            if not hidden_states or state is None:
                raise RWKVControllerError(
                    "RWKV forward must return hidden_states and recurrent state"
                )
            hidden = hidden_states[-1][:, -1, :].detach().float()
            cached_episode.append((row, hidden))
            cached_transitions += 1
        cached_episodes.append(cached_episode)
        _emit(
            progress,
            "feature-cache-progress",
            episode=episode_index,
            episodes=len(train_episode_groups),
            transitions=cached_transitions,
            total_transitions=len(train_rows),
            tokens=cached_tokens,
            max_event_tokens=max_event_tokens,
            elapsed_seconds=time.monotonic() - feature_started,
        )
    _emit(
        progress,
        "feature-cache-done",
        transitions=cached_transitions,
        tokens=cached_tokens,
        max_event_tokens=max_event_tokens,
        elapsed_seconds=time.monotonic() - feature_started,
    )

    total_updates = 0
    last_loss = 0.0
    optimization_action_correct = optimization_action_total = 0
    optimization_stop_correct = optimization_stop_total = 0
    value_examples = sum(row.value_target is not None for row in train_rows)
    epoch_history: list[Mapping[str, Any]] = []

    for epoch_index in range(1, epoch_count + 1):
        epoch_started = time.monotonic()
        epoch_loss = 0.0
        epoch_updates = 0
        epoch_action_correct = 0
        epoch_stop_correct = 0
        random.shuffle(cached_episodes)
        _emit(
            progress,
            "epoch-start",
            epoch=epoch_index,
            epochs=epoch_count,
            transitions=len(train_rows),
        )
        for episode in cached_episodes:
            for row, hidden in episode:
                action_logits = heads["action"](hidden)
                stop_logit = heads["stop"](hidden).squeeze(-1)
                value_logit = heads["value"](hidden).squeeze(-1)
                target_action = torch.tensor(
                    [action_index[row.target_action]],
                    device=resolved_device,
                    dtype=torch.long,
                )
                target_stop = torch.tensor(
                    [float(row.stop_target)],
                    device=resolved_device,
                    dtype=torch.float32,
                )
                loss = torch.nn.functional.cross_entropy(action_logits, target_action)
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    stop_logit.float(), target_stop
                )
                if row.value_target is not None:
                    target_value = torch.tensor(
                        [float(row.value_target)],
                        device=resolved_device,
                        dtype=torch.float32,
                    )
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        value_logit.float(), target_value
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                optimizer.step()
                total_updates += 1
                epoch_updates += 1
                last_loss = float(loss.detach().cpu().item())
                epoch_loss += last_loss

                predicted_action = int(torch.argmax(action_logits.detach(), dim=-1)[0].item())
                action_correct = int(predicted_action == action_index[row.target_action])
                epoch_action_correct += action_correct
                optimization_action_correct += action_correct
                optimization_action_total += 1
                predicted_stop = bool(
                    torch.sigmoid(stop_logit.detach().float())[0].item() >= 0.5
                )
                stop_correct = int(predicted_stop == row.stop_target)
                epoch_stop_correct += stop_correct
                optimization_stop_correct += stop_correct
                optimization_stop_total += 1

        epoch_summary = {
            "epoch": epoch_index,
            "loss": epoch_loss / epoch_updates if epoch_updates else 0.0,
            "action_accuracy": epoch_action_correct / epoch_updates if epoch_updates else 0.0,
            "stop_accuracy": epoch_stop_correct / epoch_updates if epoch_updates else 0.0,
            "updates": epoch_updates,
            "elapsed_seconds": time.monotonic() - epoch_started,
        }
        epoch_history.append(epoch_summary)
        _emit(progress, "epoch-done", epochs=epoch_count, **epoch_summary)

    heads.eval()
    evaluation_started = time.monotonic()
    _emit(
        progress,
        "evaluation-start",
        train_transitions=len(train_rows),
        validation_transitions=len(validation_rows),
    )
    train_evaluation = evaluate_loaded_heads(
        train_rows,
        model=model,
        tokenizer=tokenizer,
        heads=heads,
        device=resolved_device,
        action_names=ACTION_VOCAB,
    )
    validation_evaluation = (
        evaluate_loaded_heads(
            validation_rows,
            model=model,
            tokenizer=tokenizer,
            heads=heads,
            device=resolved_device,
            action_names=ACTION_VOCAB,
        )
        if validation_rows
        else {"summary": None, "predictions": []}
    )
    _emit(
        progress,
        "evaluation-done",
        elapsed_seconds=time.monotonic() - evaluation_started,
        train_action_accuracy=train_evaluation["summary"]["action_accuracy"],
        validation_action_accuracy=(
            validation_evaluation["summary"]["action_accuracy"]
            if validation_evaluation.get("summary")
            else None
        ),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "heads.safetensors"
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in heads.state_dict().items()
        },
        str(weights_path),
    )
    evaluation_payload = {
        "train": train_evaluation,
        "validation": validation_evaluation,
        "epochs": epoch_history,
    }
    (output / "evaluation.json").write_text(
        json.dumps(evaluation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    train_summary = train_evaluation["summary"]
    validation_summary = validation_evaluation.get("summary")
    summary = {
        "format": TRAINING_FORMAT,
        "controller_format": RWKV_CONTROLLER_FORMAT,
        "model_id": model_id,
        "hidden_size": hidden_size,
        "action_vocab": list(ACTION_VOCAB),
        "device": resolved_device,
        "dtype": resolved_dtype_name,
        "epochs": epoch_count,
        "learning_rate": float(learning_rate),
        "transitions": len(rows),
        "episodes": len(group_episodes(rows)),
        "train_transitions": len(train_rows),
        "train_episodes": len(group_episodes(train_rows)),
        "validation_transitions": len(validation_rows),
        "validation_episodes": len(group_episodes(validation_rows)),
        "validation_fraction": float(validation_fraction),
        "updates": total_updates,
        "final_loss": last_loss,
        "optimization_action_accuracy": (
            optimization_action_correct / optimization_action_total
            if optimization_action_total
            else 0.0
        ),
        "optimization_stop_accuracy": (
            optimization_stop_correct / optimization_stop_total
            if optimization_stop_total
            else 0.0
        ),
        "train_action_accuracy": train_summary["action_accuracy"],
        "train_stop_accuracy": train_summary["stop_accuracy"],
        "train_exact_episode_accuracy": train_summary["exact_episode_accuracy"],
        "post_train_action_accuracy": train_summary["action_accuracy"],
        "post_train_stop_accuracy": train_summary["stop_accuracy"],
        "validation_action_accuracy": (
            validation_summary["action_accuracy"] if validation_summary else None
        ),
        "validation_stop_accuracy": (
            validation_summary["stop_accuracy"] if validation_summary else None
        ),
        "validation_exact_episode_accuracy": (
            validation_summary["exact_episode_accuracy"] if validation_summary else None
        ),
        "value_examples": value_examples,
        "feature_cache_tokens": cached_tokens,
        "feature_cache_max_event_tokens": max_event_tokens,
        "epoch_history": epoch_history,
        "elapsed_seconds": time.monotonic() - started,
        "weights": "heads.safetensors",
        "evaluation": "evaluation.json",
        "backbone_frozen": True,
        "backbone_features_cached": True,
        "state_api": "rwkv7.state",
        "loader": "transformers-remote-code",
        "seed": seed,
    }
    (output / "controller.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit(
        progress,
        "training-done",
        elapsed_seconds=summary["elapsed_seconds"],
        updates=total_updates,
        output=str(output),
    )
    return summary


__all__ = ["TRAINING_FORMAT", "train_rwkv_heads"]
