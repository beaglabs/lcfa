"""Post-training evaluation for LCFA recurrent RWKV controllers.

Evaluation replays each episode from a fresh RWKV recurrent state unless the
caller already owns a cache of frozen-backbone hidden states. Splits are
episode-level so transitions from one trajectory can never leak across train
and validation.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .recurrent_transitions import ACTION_VOCAB, RecurrentTransition, load_transitions
from .rwkv_controller import DEFAULT_RWKV_MODEL, RWKVControllerError
from .torch_runtime import resolve_device, resolve_dtype


EVALUATION_FORMAT = "lcfa.rwkv-controller-eval.v1"


def group_episodes(
    rows: Sequence[RecurrentTransition],
) -> tuple[tuple[RecurrentTransition, ...], ...]:
    grouped: dict[str, list[RecurrentTransition]] = defaultdict(list)
    for row in rows:
        grouped[row.episode_id].append(row)
    return tuple(
        tuple(sorted(values, key=lambda item: item.step_index))
        for _episode_id, values in sorted(grouped.items())
    )


def split_transitions(
    rows: Sequence[RecurrentTransition],
    *,
    validation_fraction: float = 0.2,
    seed: int = 20260925,
) -> tuple[tuple[RecurrentTransition, ...], tuple[RecurrentTransition, ...]]:
    """Deterministically split complete episodes into train/validation rows."""
    if not 0.0 <= float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    episodes = list(group_episodes(rows))
    if len(episodes) < 2 or validation_fraction <= 0:
        flat = tuple(row for episode in episodes for row in episode)
        return flat, ()
    rng = random.Random(seed)
    rng.shuffle(episodes)
    validation_count = max(1, int(round(len(episodes) * validation_fraction)))
    validation_count = min(validation_count, len(episodes) - 1)
    validation_ids = {episode[0].episode_id for episode in episodes[:validation_count]}
    train = tuple(row for row in rows if row.episode_id not in validation_ids)
    validation = tuple(row for row in rows if row.episode_id in validation_ids)
    return train, validation


def summarize_predictions(predictions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not predictions:
        return {
            "format": EVALUATION_FORMAT,
            "transitions": 0,
            "episodes": 0,
            "action_accuracy": None,
            "stop_accuracy": None,
            "exact_episode_accuracy": None,
            "value_examples": 0,
            "value_mae": None,
            "per_action": {},
        }
    action_correct = sum(bool(item.get("action_correct")) for item in predictions)
    stop_correct = sum(bool(item.get("stop_correct")) for item in predictions)
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    per_action: dict[str, dict[str, int]] = {
        name: {"total": 0, "correct": 0} for name in ACTION_VOCAB
    }
    value_errors: list[float] = []
    for item in predictions:
        episode_id = str(item["episode_id"])
        by_episode[episode_id].append(item)
        target = str(item["target_action"])
        if target not in per_action:
            per_action[target] = {"total": 0, "correct": 0}
        per_action[target]["total"] += 1
        per_action[target]["correct"] += int(bool(item.get("action_correct")))
        if item.get("value_target") is not None and item.get("predicted_value") is not None:
            value_errors.append(
                abs(float(item["predicted_value"]) - float(item["value_target"]))
            )
    exact = sum(
        all(
            bool(item.get("action_correct")) and bool(item.get("stop_correct"))
            for item in episode
        )
        for episode in by_episode.values()
    )
    rendered_per_action = {
        name: {
            **counts,
            "accuracy": (
                counts["correct"] / counts["total"] if counts["total"] else None
            ),
        }
        for name, counts in per_action.items()
        if counts["total"]
    }
    return {
        "format": EVALUATION_FORMAT,
        "transitions": len(predictions),
        "episodes": len(by_episode),
        "action_accuracy": action_correct / len(predictions),
        "stop_accuracy": stop_correct / len(predictions),
        "exact_episode_accuracy": exact / len(by_episode) if by_episode else None,
        "value_examples": len(value_errors),
        "value_mae": sum(value_errors) / len(value_errors) if value_errors else None,
        "per_action": rendered_per_action,
    }


def _prediction_from_hidden(
    row: RecurrentTransition,
    hidden: Any,
    *,
    heads: Any,
    action_names: Sequence[str],
) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RWKVControllerError("RWKV evaluation requires torch") from exc
    with torch.inference_mode():
        action_logits = heads["action"](hidden)
        stop_probability = float(
            torch.sigmoid(heads["stop"](hidden).float())[0, 0].item()
        )
        predicted_value = float(
            torch.sigmoid(heads["value"](hidden).float())[0, 0].item()
        )
    predicted_index = int(torch.argmax(action_logits, dim=-1)[0].item())
    predicted_action = tuple(action_names)[predicted_index]
    predicted_stop = stop_probability >= 0.5
    return {
        "episode_id": row.episode_id,
        "step_index": row.step_index,
        "target_action": row.target_action,
        "predicted_action": predicted_action,
        "action_correct": predicted_action == row.target_action,
        "target_stop": row.stop_target,
        "predicted_stop": predicted_stop,
        "stop_probability": stop_probability,
        "stop_correct": predicted_stop == row.stop_target,
        "value_target": row.value_target,
        "predicted_value": predicted_value,
    }


def evaluate_cached_heads(
    cached_episodes: Sequence[Sequence[tuple[RecurrentTransition, Any]]],
    *,
    heads: Any,
    action_names: Sequence[str] = ACTION_VOCAB,
) -> Mapping[str, Any]:
    """Evaluate fixed heads on already-computed frozen RWKV hidden states."""
    heads.eval()
    predictions = [
        _prediction_from_hidden(
            row,
            hidden,
            heads=heads,
            action_names=action_names,
        )
        for episode in cached_episodes
        for row, hidden in episode
    ]
    return {
        "summary": summarize_predictions(predictions),
        "predictions": predictions,
    }


def _event_text(row: RecurrentTransition) -> str:
    return json.dumps(row.event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def evaluate_loaded_heads(
    rows: Sequence[RecurrentTransition],
    *,
    model: Any,
    tokenizer: Any,
    heads: Any,
    device: str,
    action_names: Sequence[str] = ACTION_VOCAB,
) -> Mapping[str, Any]:
    """Replay rows through a frozen model/head pair from fresh episode states."""
    try:
        import torch
    except ImportError as exc:
        raise RWKVControllerError("RWKV evaluation requires torch") from exc
    action_names = tuple(str(name) for name in action_names)
    heads.eval()
    predictions: list[Mapping[str, Any]] = []
    for episode in group_episodes(rows):
        state: Any = None
        for row in episode:
            encoded = tokenizer(_event_text(row), return_tensors="pt", add_special_tokens=False)
            kwargs: dict[str, Any] = {
                "input_ids": encoded["input_ids"].to(device),
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
            predictions.append(
                _prediction_from_hidden(
                    row,
                    hidden,
                    heads=heads,
                    action_names=action_names,
                )
            )
    return {
        "summary": summarize_predictions(predictions),
        "predictions": predictions,
    }


def evaluate_rwkv_heads(
    transitions_path: str | Path,
    controller_dir: str | Path,
    *,
    model_id: str | None = None,
    device: str | None = "auto",
    dtype: str = "auto",
) -> Mapping[str, Any]:
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RWKVControllerError(
            "RWKV evaluation requires `pip install -e '.[rwkv]'`"
        ) from exc

    root = Path(controller_dir)
    manifest_path = root / "controller.json" if root.is_dir() else root
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("controller manifest must be a JSON object")
    root = manifest_path.parent
    resolved_model = str(model_id or manifest.get("model_id") or DEFAULT_RWKV_MODEL)
    resolved_device = resolve_device(torch, device)
    try:
        resolved_dtype_name, resolved_dtype = resolve_dtype(
            torch, resolved_device, dtype
        )
    except ValueError as exc:
        raise RWKVControllerError(str(exc)) from exc
    action_names = tuple(manifest.get("action_vocab") or ACTION_VOCAB)

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        dtype=resolved_dtype,
        trust_remote_code=True,
    ).to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RWKVControllerError("RWKV model config does not expose hidden_size")
    heads = torch.nn.ModuleDict({
        "action": torch.nn.Linear(hidden_size, len(action_names)),
        "stop": torch.nn.Linear(hidden_size, 1),
        "value": torch.nn.Linear(hidden_size, 1),
    }).to(resolved_device)
    weights_path = root / str(manifest.get("weights") or "heads.safetensors")
    state = load_file(str(weights_path), device="cpu")
    heads.load_state_dict(state, strict=True)
    result = evaluate_loaded_heads(
        load_transitions(transitions_path),
        model=model,
        tokenizer=tokenizer,
        heads=heads,
        device=resolved_device,
        action_names=action_names,
    )
    return {
        "format": EVALUATION_FORMAT,
        "model_id": resolved_model,
        "controller": str(manifest_path),
        "device": resolved_device,
        "dtype": resolved_dtype_name,
        **result,
    }


__all__ = [
    "EVALUATION_FORMAT",
    "evaluate_cached_heads",
    "evaluate_loaded_heads",
    "evaluate_rwkv_heads",
    "group_episodes",
    "split_transitions",
    "summarize_predictions",
]
