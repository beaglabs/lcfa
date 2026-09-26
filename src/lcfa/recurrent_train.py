"""Training for the first LCFA recurrent-controller experiment.

Phase 1 deliberately freezes RWKV-7 and trains only action/stop/value heads.
This isolates whether pretrained recurrent state already contains useful
control information before recurrent-backbone fine-tuning.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Mapping

from .recurrent_eval import evaluate_loaded_heads, group_episodes, split_transitions
from .recurrent_transitions import ACTION_VOCAB, RecurrentTransition, load_transitions
from .rwkv_controller import DEFAULT_RWKV_MODEL, RWKV_CONTROLLER_FORMAT, RWKVControllerError
from .torch_runtime import resolve_device, resolve_dtype


TRAINING_FORMAT = "lcfa.rwkv-controller-training.v1"


def _event_text(row: RecurrentTransition) -> str:
    return json.dumps(row.event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


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

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
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
        "action": torch.nn.Linear(hidden_size, len(ACTION_VOCAB)),
        "stop": torch.nn.Linear(hidden_size, 1),
        "value": torch.nn.Linear(hidden_size, 1),
    }).to(resolved_device)
    heads.train()
    optimizer = torch.optim.AdamW(heads.parameters(), lr=float(learning_rate))
    action_index = {name: index for index, name in enumerate(ACTION_VOCAB)}
    episodes = [list(episode) for episode in group_episodes(train_rows)]

    total_updates = 0
    last_loss = 0.0
    optimization_action_correct = optimization_action_total = 0
    optimization_stop_correct = optimization_stop_total = 0
    value_examples = sum(row.value_target is not None for row in train_rows)

    for _epoch in range(max(1, int(epochs))):
        random.shuffle(episodes)
        for episode in episodes:
            state: Any = None
            for row in episode:
                encoded = tokenizer(_event_text(row), return_tensors="pt", add_special_tokens=False)
                kwargs: dict[str, Any] = {
                    "input_ids": encoded["input_ids"].to(resolved_device),
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
                last_loss = float(loss.detach().cpu().item())

                predicted_action = int(torch.argmax(action_logits.detach(), dim=-1)[0].item())
                optimization_action_correct += int(
                    predicted_action == action_index[row.target_action]
                )
                optimization_action_total += 1
                predicted_stop = bool(
                    torch.sigmoid(stop_logit.detach().float())[0].item() >= 0.5
                )
                optimization_stop_correct += int(predicted_stop == row.stop_target)
                optimization_stop_total += 1

    heads.eval()
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
        "epochs": max(1, int(epochs)),
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
        "weights": "heads.safetensors",
        "evaluation": "evaluation.json",
        "backbone_frozen": True,
        "state_api": "rwkv7.state",
        "loader": "transformers-remote-code",
        "seed": seed,
    }
    (output / "controller.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["TRAINING_FORMAT", "train_rwkv_heads"]
