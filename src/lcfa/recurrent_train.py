"""Training for the first LCFA recurrent-controller experiment.

Phase 1 deliberately freezes RWKV-7 and trains only action/stop/value heads.
This isolates whether the pretrained recurrent state already contains useful
control information before any expensive recurrent-backbone fine-tuning.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Mapping

from .recurrent_transitions import ACTION_VOCAB, RecurrentTransition, load_transitions
from .rwkv_controller import DEFAULT_RWKV_MODEL, RWKV_CONTROLLER_FORMAT, RWKVControllerError


TRAINING_FORMAT = "lcfa.rwkv-controller-training.v1"


def _dtype(torch: Any, name: str) -> Any:
    value = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(str(name).lower())
    if value is None:
        raise RWKVControllerError(f"unsupported dtype: {name}")
    return value


def _device(torch: Any, requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _event_text(row: RecurrentTransition) -> str:
    return json.dumps(
        {
            "goal": row.goal,
            "event": row.event,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"


def _group(rows: tuple[RecurrentTransition, ...]) -> list[list[RecurrentTransition]]:
    grouped: dict[str, list[RecurrentTransition]] = defaultdict(list)
    for row in rows:
        grouped[row.episode_id].append(row)
    episodes = []
    for values in grouped.values():
        episodes.append(sorted(values, key=lambda item: item.step_index))
    return episodes


def train_rwkv_heads(
    transitions_path: str | Path,
    output_dir: str | Path,
    *,
    model_id: str = DEFAULT_RWKV_MODEL,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device: str | None = None,
    dtype: str = "bfloat16",
    seed: int = 20260925,
) -> Mapping[str, Any]:
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RWKVControllerError(
            "RWKV training requires `pip install -e '.[rwkv]'`"
        ) from exc

    rows = load_transitions(transitions_path)
    if not rows:
        raise ValueError("transition dataset is empty")
    for row in rows:
        if row.target_action not in ACTION_VOCAB:
            raise ValueError(f"unknown action target: {row.target_action}")

    random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = _device(torch, device)
    resolved_dtype = _dtype(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=resolved_dtype).to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

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
    episodes = _group(rows)

    total_updates = 0
    last_loss = 0.0
    action_correct = 0
    action_total = 0
    stop_correct = 0
    stop_total = 0
    value_examples = 0

    for _epoch in range(max(1, int(epochs))):
        random.shuffle(episodes)
        for episode in episodes:
            cache: Any = None
            for row in episode:
                encoded = tokenizer(
                    _event_text(row),
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                kwargs: dict[str, Any] = {
                    "input_ids": encoded["input_ids"].to(resolved_device),
                    "use_cache": True,
                    "output_hidden_states": True,
                    "return_dict": True,
                }
                if cache is not None:
                    kwargs["past_key_values"] = cache
                with torch.inference_mode():
                    outputs = model(**kwargs)
                hidden_states = getattr(outputs, "hidden_states", None)
                cache = getattr(outputs, "past_key_values", None)
                if not hidden_states or cache is None:
                    raise RWKVControllerError(
                        "RWKV forward must return hidden_states and recurrent cache"
                    )
                hidden = hidden_states[-1][:, -1, :].detach().float()

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
                    value_examples += 1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                optimizer.step()
                total_updates += 1
                last_loss = float(loss.detach().cpu().item())

                predicted_action = int(torch.argmax(action_logits.detach(), dim=-1)[0].item())
                action_correct += int(predicted_action == action_index[row.target_action])
                action_total += 1
                predicted_stop = bool(torch.sigmoid(stop_logit.detach().float())[0].item() >= 0.5)
                stop_correct += int(predicted_stop == row.stop_target)
                stop_total += 1

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "heads.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in heads.state_dict().items()}, str(weights_path))
    summary = {
        "format": TRAINING_FORMAT,
        "controller_format": RWKV_CONTROLLER_FORMAT,
        "model_id": model_id,
        "hidden_size": hidden_size,
        "action_vocab": list(ACTION_VOCAB),
        "epochs": max(1, int(epochs)),
        "learning_rate": float(learning_rate),
        "transitions": len(rows),
        "episodes": len(episodes),
        "updates": total_updates,
        "final_loss": last_loss,
        "train_action_accuracy": action_correct / action_total if action_total else 0.0,
        "train_stop_accuracy": stop_correct / stop_total if stop_total else 0.0,
        "value_examples": value_examples,
        "weights": "heads.safetensors",
        "backbone_frozen": True,
        "seed": seed,
    }
    (output / "controller.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["TRAINING_FORMAT", "train_rwkv_heads"]
