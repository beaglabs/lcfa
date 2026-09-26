"""Training for LCFA recurrent RWKV controllers.

Fast ``frozen`` mode caches RWKV recurrent features and trains action, pointer,
stop, and value heads. ``full`` mode updates RWKV itself from the same
closed-loop event schema and adds causal-LM supervision for repo.replace/edit
arguments, allowing verifier-derived corrective trajectories to improve both
control and source-edit rendering.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

from .recurrent_eval import (
    evaluate_cached_heads,
    evaluate_loaded_heads,
    group_episodes,
    split_transitions,
)
from .recurrent_transitions import ACTION_VOCAB, RecurrentTransition, load_transitions
from .rwkv_controller import (
    DEFAULT_POINTER_SLOTS,
    DEFAULT_RWKV_MODEL,
    RWKV_CONTROLLER_FORMAT,
    RWKVControllerError,
    argument_prompt,
)
from .torch_runtime import resolve_device, resolve_dtype


TRAINING_FORMAT = "lcfa.rwkv-controller-training.v2"
BACKBONE_MODES = ("frozen", "full")
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _event_text(row: RecurrentTransition) -> str:
    return json.dumps(row.event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def _emit(progress: ProgressCallback | None, event: str, **values: Any) -> None:
    if progress is not None:
        progress({"event": event, **values})


def _detach_state(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_state(item) for item in value)
    if isinstance(value, list):
        return [_detach_state(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _detach_state(item) for key, item in value.items()}
    return value


def _make_heads(torch: Any, hidden_size: int, pointer_slots: int, device: str) -> Any:
    modules: dict[str, Any] = {
        "action": torch.nn.Linear(hidden_size, len(ACTION_VOCAB)),
        "stop": torch.nn.Linear(hidden_size, 1),
        "value": torch.nn.Linear(hidden_size, 1),
    }
    if pointer_slots > 0:
        modules["pointer"] = torch.nn.Linear(hidden_size, pointer_slots)
    return torch.nn.ModuleDict(modules).to(device)


def _control_loss(
    torch: Any,
    row: RecurrentTransition,
    hidden: Any,
    *,
    heads: Any,
    action_index: Mapping[str, int],
    device: str,
    pointer_loss_weight: float,
) -> tuple[Any, Mapping[str, Any]]:
    action_logits = heads["action"](hidden)
    stop_logit = heads["stop"](hidden).squeeze(-1)
    value_logit = heads["value"](hidden).squeeze(-1)
    target_action = torch.tensor(
        [action_index[row.target_action]], device=device, dtype=torch.long
    )
    target_stop = torch.tensor(
        [float(row.stop_target)], device=device, dtype=torch.float32
    )
    loss = torch.nn.functional.cross_entropy(action_logits, target_action)
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        stop_logit.float(), target_stop
    )
    if row.value_target is not None:
        target_value = torch.tensor(
            [float(row.value_target)], device=device, dtype=torch.float32
        )
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            value_logit.float(), target_value
        )

    pointer_logits = None
    pointer_correct = None
    if (
        row.target_pointer is not None
        and "pointer" in heads
        and 0 <= int(row.target_pointer) < int(heads["pointer"].out_features)
    ):
        pointer_logits = heads["pointer"](hidden)
        target_pointer = torch.tensor(
            [int(row.target_pointer)], device=device, dtype=torch.long
        )
        loss = loss + float(pointer_loss_weight) * torch.nn.functional.cross_entropy(
            pointer_logits, target_pointer
        )
        predicted_pointer = int(torch.argmax(pointer_logits.detach(), dim=-1)[0].item())
        pointer_correct = int(predicted_pointer == int(row.target_pointer))

    predicted_action = int(torch.argmax(action_logits.detach(), dim=-1)[0].item())
    predicted_stop = bool(torch.sigmoid(stop_logit.detach().float())[0].item() >= 0.5)
    return loss, {
        "action_correct": int(predicted_action == action_index[row.target_action]),
        "stop_correct": int(predicted_stop == row.stop_target),
        "pointer_correct": pointer_correct,
    }


def _argument_loss(
    torch: Any,
    model: Any,
    tokenizer: Any,
    row: RecurrentTransition,
    *,
    device: str,
    max_argument_chars: int,
) -> Any | None:
    if row.target_action not in {"repo.replace", "repo.edit"}:
        return None
    if not isinstance(row.target_inputs, Mapping) or not row.target_inputs:
        return None
    target_text = json.dumps(
        dict(row.target_inputs), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"
    if len(target_text) > max(256, int(max_argument_chars)):
        return None
    target_path = str(row.target_inputs.get("path") or "") or None
    prompt = argument_prompt(
        row.target_action,
        row.goal,
        row.event,
        target_path=target_path,
    )
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    target_ids = tokenizer(
        target_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    if target_ids.shape[-1] == 0:
        return None
    input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
    outputs = model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:].clone()
    prompt_length = int(prompt_ids.shape[-1])
    if prompt_length > 1:
        labels[:, : prompt_length - 1] = -100
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


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
    backbone_mode: str = "frozen",
    backbone_learning_rate: float = 5e-6,
    pointer_loss_weight: float = 0.5,
    argument_loss_weight: float = 0.25,
    max_argument_chars: int = 8192,
    progress: ProgressCallback | None = None,
) -> Mapping[str, Any]:
    try:
        import torch
        from safetensors.torch import save_file, save_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RWKVControllerError(
            "RWKV training requires `pip install -e '.[rwkv]'`"
        ) from exc

    resolved_backbone_mode = str(backbone_mode).strip().lower()
    if resolved_backbone_mode not in BACKBONE_MODES:
        raise ValueError(f"backbone_mode must be one of {BACKBONE_MODES}")

    started = time.monotonic()
    rows = load_transitions(transitions_path)
    if not rows:
        raise ValueError("transition dataset is empty")
    for row in rows:
        if row.target_action not in ACTION_VOCAB:
            raise ValueError(f"unknown action target: {row.target_action}")
    train_rows, validation_rows = split_transitions(
        rows, validation_fraction=validation_fraction, seed=seed
    )
    if not train_rows:
        raise ValueError("training split is empty")

    pointer_examples = [
        row for row in rows
        if row.target_pointer is not None and int(row.target_pointer) >= 0
    ]
    max_pointer = max((int(row.target_pointer) for row in pointer_examples), default=-1)
    pointer_slots = (
        max(DEFAULT_POINTER_SLOTS, max_pointer + 1) if pointer_examples else 0
    )

    random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = resolve_device(torch, device)
    try:
        resolved_dtype_name, resolved_dtype = resolve_dtype(
            torch, resolved_device, dtype
        )
    except ValueError as exc:
        raise RWKVControllerError(str(exc)) from exc

    epoch_count = max(1, int(epochs))
    all_episode_groups = [list(episode) for episode in group_episodes(rows)]
    train_ids = {row.episode_id for row in train_rows}
    validation_ids = {row.episode_id for row in validation_rows}
    _emit(
        progress,
        "model-load-start",
        model_id=model_id,
        device=resolved_device,
        dtype=resolved_dtype_name,
        transitions=len(rows),
        train_transitions=len(train_rows),
        validation_transitions=len(validation_rows),
        train_episodes=len(group_episodes(train_rows)),
        epochs=epoch_count,
        backbone_mode=resolved_backbone_mode,
        pointer_slots=pointer_slots,
    )
    load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=resolved_dtype,
        trust_remote_code=True,
    ).to(resolved_device)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RWKVControllerError("RWKV model config does not expose hidden_size")
    heads = _make_heads(torch, hidden_size, pointer_slots, resolved_device)
    action_index = {name: index for index, name in enumerate(ACTION_VOCAB)}

    frozen = resolved_backbone_mode == "frozen"
    model.eval() if frozen else model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(not frozen)
    heads.train()
    _emit(
        progress,
        "model-loaded",
        elapsed_seconds=time.monotonic() - load_started,
        device=resolved_device,
        dtype=resolved_dtype_name,
        backbone_mode=resolved_backbone_mode,
    )

    cached_episodes: list[list[tuple[RecurrentTransition, Any]]] = []
    cached_tokens = 0
    max_event_tokens = 0
    cached_train: list[list[tuple[RecurrentTransition, Any]]] = []
    cached_validation: list[list[tuple[RecurrentTransition, Any]]] = []

    if frozen:
        feature_started = time.monotonic()
        _emit(
            progress,
            "feature-cache-start",
            episodes=len(all_episode_groups),
            transitions=len(rows),
        )
        cached_transitions = 0
        for episode_index, episode in enumerate(all_episode_groups, start=1):
            state: Any = None
            cached_episode: list[tuple[RecurrentTransition, Any]] = []
            for row in episode:
                encoded = tokenizer(
                    _event_text(row), return_tensors="pt", add_special_tokens=False
                )
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
                episodes=len(all_episode_groups),
                transitions=cached_transitions,
                total_transitions=len(rows),
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
        cached_train = [
            episode for episode in cached_episodes
            if episode and episode[0][0].episode_id in train_ids
        ]
        cached_validation = [
            episode for episode in cached_episodes
            if episode and episode[0][0].episode_id in validation_ids
        ]
        optimizer = torch.optim.AdamW(heads.parameters(), lr=float(learning_rate))
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": list(heads.parameters()), "lr": float(learning_rate)},
                {"params": list(model.parameters()), "lr": float(backbone_learning_rate)},
            ]
        )

    total_updates = 0
    last_loss = 0.0
    optimization_action_correct = optimization_action_total = 0
    optimization_stop_correct = optimization_stop_total = 0
    optimization_pointer_correct = optimization_pointer_total = 0
    argument_examples = 0
    value_examples = sum(row.value_target is not None for row in train_rows)
    epoch_history: list[Mapping[str, Any]] = []

    for epoch_index in range(1, epoch_count + 1):
        epoch_started = time.monotonic()
        epoch_loss = 0.0
        epoch_updates = 0
        epoch_action_correct = 0
        epoch_stop_correct = 0
        epoch_pointer_correct = 0
        epoch_pointer_total = 0
        epoch_argument_examples = 0
        _emit(
            progress,
            "epoch-start",
            epoch=epoch_index,
            epochs=epoch_count,
            transitions=len(train_rows),
        )

        if frozen:
            random.shuffle(cached_train)
            iterable: Sequence[Any] = cached_train
        else:
            train_groups = [
                list(episode) for episode in group_episodes(train_rows)
            ]
            random.shuffle(train_groups)
            iterable = train_groups

        for episode in iterable:
            state: Any = None
            for item in episode:
                if frozen:
                    row, hidden = item
                else:
                    row = item
                    encoded = tokenizer(
                        _event_text(row), return_tensors="pt", add_special_tokens=False
                    )
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
                        kwargs["state"] = _detach_state(state)
                    outputs = model(**kwargs)
                    hidden_states = getattr(outputs, "hidden_states", None)
                    state = getattr(outputs, "state", None)
                    if not hidden_states or state is None:
                        raise RWKVControllerError(
                            "RWKV forward must return hidden_states and recurrent state"
                        )
                    hidden = hidden_states[-1][:, -1, :].float()

                loss, accuracy = _control_loss(
                    torch,
                    row,
                    hidden,
                    heads=heads,
                    action_index=action_index,
                    device=resolved_device,
                    pointer_loss_weight=pointer_loss_weight,
                )
                if not frozen and float(argument_loss_weight) > 0:
                    arg_loss = _argument_loss(
                        torch,
                        model,
                        tokenizer,
                        row,
                        device=resolved_device,
                        max_argument_chars=max_argument_chars,
                    )
                    if arg_loss is not None:
                        loss = loss + float(argument_loss_weight) * arg_loss
                        epoch_argument_examples += 1
                        argument_examples += 1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = list(heads.parameters())
                if not frozen:
                    parameters.extend(parameter for parameter in model.parameters() if parameter.requires_grad)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                total_updates += 1
                epoch_updates += 1
                last_loss = float(loss.detach().cpu().item())
                epoch_loss += last_loss
                epoch_action_correct += int(accuracy["action_correct"])
                epoch_stop_correct += int(accuracy["stop_correct"])
                optimization_action_correct += int(accuracy["action_correct"])
                optimization_action_total += 1
                optimization_stop_correct += int(accuracy["stop_correct"])
                optimization_stop_total += 1
                if accuracy["pointer_correct"] is not None:
                    epoch_pointer_correct += int(accuracy["pointer_correct"])
                    epoch_pointer_total += 1
                    optimization_pointer_correct += int(accuracy["pointer_correct"])
                    optimization_pointer_total += 1

        epoch_summary = {
            "epoch": epoch_index,
            "loss": epoch_loss / epoch_updates if epoch_updates else 0.0,
            "action_accuracy": epoch_action_correct / epoch_updates if epoch_updates else 0.0,
            "stop_accuracy": epoch_stop_correct / epoch_updates if epoch_updates else 0.0,
            "pointer_accuracy": (
                epoch_pointer_correct / epoch_pointer_total if epoch_pointer_total else None
            ),
            "pointer_examples": epoch_pointer_total,
            "argument_examples": epoch_argument_examples,
            "updates": epoch_updates,
            "elapsed_seconds": time.monotonic() - epoch_started,
        }
        epoch_history.append(epoch_summary)
        _emit(progress, "epoch-done", epochs=epoch_count, **epoch_summary)

    heads.eval()
    model.eval()
    evaluation_started = time.monotonic()
    _emit(
        progress,
        "evaluation-start",
        train_transitions=len(train_rows),
        validation_transitions=len(validation_rows),
    )
    if frozen:
        train_evaluation = evaluate_cached_heads(
            cached_train, heads=heads, action_names=ACTION_VOCAB
        )
        validation_evaluation = (
            evaluate_cached_heads(
                cached_validation, heads=heads, action_names=ACTION_VOCAB
            )
            if cached_validation
            else {"summary": None, "predictions": []}
        )
    else:
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
            if validation_evaluation.get("summary") else None
        ),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "heads.safetensors"
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in heads.state_dict().items()},
        str(weights_path),
    )
    backbone_weights: str | None = None
    if not frozen:
        backbone_path = output / "backbone.safetensors"
        save_model(model, str(backbone_path))
        backbone_weights = backbone_path.name

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
        "pointer_slots": pointer_slots,
        "device": resolved_device,
        "dtype": resolved_dtype_name,
        "epochs": epoch_count,
        "learning_rate": float(learning_rate),
        "backbone_learning_rate": float(backbone_learning_rate),
        "backbone_mode": resolved_backbone_mode,
        "pointer_loss_weight": float(pointer_loss_weight),
        "argument_loss_weight": float(argument_loss_weight),
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
            if optimization_action_total else 0.0
        ),
        "optimization_stop_accuracy": (
            optimization_stop_correct / optimization_stop_total
            if optimization_stop_total else 0.0
        ),
        "optimization_pointer_accuracy": (
            optimization_pointer_correct / optimization_pointer_total
            if optimization_pointer_total else None
        ),
        "train_action_accuracy": train_summary["action_accuracy"],
        "train_pointer_accuracy": train_summary.get("pointer_accuracy"),
        "train_stop_accuracy": train_summary["stop_accuracy"],
        "train_exact_episode_accuracy": train_summary["exact_episode_accuracy"],
        "post_train_action_accuracy": train_summary["action_accuracy"],
        "post_train_stop_accuracy": train_summary["stop_accuracy"],
        "validation_action_accuracy": (
            validation_summary["action_accuracy"] if validation_summary else None
        ),
        "validation_pointer_accuracy": (
            validation_summary.get("pointer_accuracy") if validation_summary else None
        ),
        "validation_stop_accuracy": (
            validation_summary["stop_accuracy"] if validation_summary else None
        ),
        "validation_exact_episode_accuracy": (
            validation_summary["exact_episode_accuracy"] if validation_summary else None
        ),
        "value_examples": value_examples,
        "pointer_examples": len(pointer_examples),
        "argument_examples": argument_examples,
        "feature_cache_tokens": cached_tokens,
        "feature_cache_max_event_tokens": max_event_tokens,
        "feature_cache_includes_validation": frozen,
        "epoch_history": epoch_history,
        "elapsed_seconds": time.monotonic() - started,
        "weights": "heads.safetensors",
        "backbone_weights": backbone_weights,
        "evaluation": "evaluation.json",
        "backbone_frozen": frozen,
        "backbone_features_cached": frozen,
        "evaluation_uses_cached_features": frozen,
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


__all__ = ["BACKBONE_MODES", "TRAINING_FORMAT", "train_rwkv_heads"]
