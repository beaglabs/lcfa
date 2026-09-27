"""End-to-end training for the LCFA typed-latent + RWKV hybrid controller."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

from .hybrid_latent import (
    HYBRID_CONTROLLER_FORMAT,
    HybridLatentConfig,
    hybrid_event,
    make_hybrid_core,
    repair_prompt,
    supervised_plan_targets,
    supervised_repair_plan,
)
from .recurrent_eval import group_episodes, split_transitions, summarize_predictions
from .recurrent_transitions import ACTION_VOCAB, RecurrentTransition, load_transitions
from .rwkv_controller import DEFAULT_POINTER_SLOTS, DEFAULT_RWKV_MODEL, RWKVControllerError
from .torch_runtime import resolve_device, resolve_dtype

HYBRID_TRAINING_FORMAT = "lcfa.hybrid-latent-rwkv-training.v1"
BACKBONE_MODES = ("frozen", "full")
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _emit(progress: ProgressCallback | None, event: str, **values: Any) -> None:
    if progress is not None:
        progress({"event": event, **values})


def _event_text(row: RecurrentTransition) -> str:
    return json.dumps(
        hybrid_event(row.event),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"


def _make_heads(torch: Any, hidden_size: int, pointer_slots: int, device: str) -> Any:
    modules: dict[str, Any] = {
        "action": torch.nn.Linear(hidden_size, len(ACTION_VOCAB)),
        "stop": torch.nn.Linear(hidden_size, 1),
        "value": torch.nn.Linear(hidden_size, 1),
    }
    if pointer_slots > 0:
        modules["pointer"] = torch.nn.Linear(hidden_size, pointer_slots)
    return torch.nn.ModuleDict(modules).to(device)


def _load_initial_controller(
    init_controller: str | Path | None,
    *,
    model: Any,
    hybrid: Any,
    heads: Any,
    model_id: str,
    device: str,
) -> Mapping[str, Any]:
    if init_controller is None:
        return {
            "controller": None,
            "head_tensors_loaded": 0,
            "hybrid_tensors_loaded": 0,
            "backbone_loaded": False,
        }
    try:
        from safetensors.torch import load_file, load_model
    except ImportError as exc:
        raise RWKVControllerError("safetensors torch support is required") from exc

    root = Path(init_controller).expanduser().resolve()
    manifest_path = root / "controller.json" if root.is_dir() else root
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("initial controller manifest must be a JSON object")
    previous_model = str(manifest.get("model_id") or "")
    if previous_model and previous_model != str(model_id):
        raise ValueError(
            f"initial controller model mismatch: {previous_model} != {model_id}"
        )
    root = manifest_path.parent

    backbone_loaded = False
    backbone_weights = manifest.get("backbone_weights")
    if backbone_weights:
        missing, unexpected = load_model(
            model,
            str(root / str(backbone_weights)),
            strict=False,
            device=device,
        )
        if missing or unexpected:
            raise RWKVControllerError(
                "initial backbone tensor mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        backbone_loaded = True

    head_loaded = 0
    weights_path = root / str(manifest.get("weights") or "heads.safetensors")
    if weights_path.is_file():
        loaded = load_file(str(weights_path), device=device)
        current = heads.state_dict()
        compatible = {
            key: tensor
            for key, tensor in loaded.items()
            if key in current and tuple(tensor.shape) == tuple(current[key].shape)
        }
        current.update(compatible)
        heads.load_state_dict(current, strict=True)
        head_loaded = len(compatible)

    hybrid_loaded = 0
    hybrid_name = manifest.get("hybrid_weights")
    if hybrid_name:
        hybrid_path = root / str(hybrid_name)
        if hybrid_path.is_file():
            loaded = load_file(str(hybrid_path), device=device)
            current = hybrid.state_dict()
            compatible = {
                key: tensor
                for key, tensor in loaded.items()
                if key in current and tuple(tensor.shape) == tuple(current[key].shape)
            }
            current.update(compatible)
            hybrid.load_state_dict(current, strict=True)
            hybrid_loaded = len(compatible)

    return {
        "controller": str(manifest_path),
        "head_tensors_loaded": head_loaded,
        "hybrid_tensors_loaded": hybrid_loaded,
        "backbone_loaded": backbone_loaded,
    }


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


def _repair_plan_loss(torch: Any, row: RecurrentTransition, plan_logits: Any) -> Any:
    targets = supervised_plan_targets(
        row.target_action,
        stop_target=row.stop_target,
        value_target=row.value_target,
    )
    losses: list[Any] = []
    for index, target in enumerate(targets):
        if target is None:
            continue
        expected = torch.tensor(
            [float(target)], device=plan_logits.device, dtype=torch.float32
        )
        losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                plan_logits[:, index].float(), expected
            )
        )
    if not losses:
        return plan_logits.sum() * 0.0
    return torch.stack(losses).mean()


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
    plan = supervised_repair_plan(
        goal=row.goal,
        action=row.target_action,
        target_inputs=row.target_inputs,
        candidate_paths=row.candidate_paths,
        value_target=row.value_target,
    )
    prompt = repair_prompt(
        row.target_action,
        row.goal,
        row.event,
        target_path=target_path,
        repair_plan=plan,
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
    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
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


def _full_episode_hidden_states(
    model: Any,
    tokenizer: Any,
    episode: Sequence[RecurrentTransition],
    *,
    device: str,
) -> tuple[list[tuple[RecurrentTransition, Any]], int, int]:
    if not episode:
        return [], 0, 0
    texts = [_event_text(row) for row in episode]
    full_text = "".join(texts)
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    token_count = int(input_ids.shape[-1])
    if token_count <= 0:
        raise RWKVControllerError("hybrid recurrent episode tokenized to an empty sequence")

    boundaries: list[int] = []
    prefix = ""
    for text in texts:
        prefix += text
        prefix_ids = tokenizer(
            prefix, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        boundary = int(prefix_ids.shape[-1]) - 1
        if boundary < 0 or boundary >= token_count:
            raise RWKVControllerError(
                f"invalid hybrid event boundary {boundary} for {token_count} tokens"
            )
        boundaries.append(boundary)

    outputs = model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise RWKVControllerError("RWKV forward must return hidden_states")
    sequence_hidden = hidden_states[-1].float()
    return (
        [
            (row, sequence_hidden[:, boundary, :])
            for row, boundary in zip(episode, boundaries)
        ],
        token_count,
        max(boundary + 1 for boundary in boundaries),
    )


def _prediction_from_hidden(
    torch: Any,
    row: RecurrentTransition,
    hidden: Any,
    *,
    heads: Any,
    action_names: Sequence[str],
) -> Mapping[str, Any]:
    with torch.inference_mode():
        action_logits = heads["action"](hidden)
        stop_probability = float(
            torch.sigmoid(heads["stop"](hidden).float())[0, 0].item()
        )
        predicted_value = float(
            torch.sigmoid(heads["value"](hidden).float())[0, 0].item()
        )
        predicted_pointer = None
        pointer_confidence = None
        if "pointer" in heads:
            pointer_probs = torch.softmax(heads["pointer"](hidden), dim=-1)[0]
            predicted_pointer = int(torch.argmax(pointer_probs).item())
            pointer_confidence = float(pointer_probs[predicted_pointer].item())
    predicted_index = int(torch.argmax(action_logits, dim=-1)[0].item())
    predicted_action = tuple(action_names)[predicted_index]
    predicted_stop = stop_probability >= 0.5
    return {
        "episode_id": row.episode_id,
        "step_index": row.step_index,
        "target_action": row.target_action,
        "predicted_action": predicted_action,
        "action_correct": predicted_action == row.target_action,
        "target_pointer": row.target_pointer,
        "predicted_pointer": predicted_pointer,
        "pointer_confidence": pointer_confidence,
        "pointer_correct": (
            None if row.target_pointer is None else predicted_pointer == row.target_pointer
        ),
        "target_stop": row.stop_target,
        "predicted_stop": predicted_stop,
        "stop_probability": stop_probability,
        "stop_correct": predicted_stop == row.stop_target,
        "value_target": row.value_target,
        "predicted_value": predicted_value,
    }


def _evaluate_cached(
    torch: Any,
    cached_episodes: Sequence[Sequence[tuple[RecurrentTransition, Any]]],
    *,
    hybrid: Any,
    heads: Any,
) -> Mapping[str, Any]:
    hybrid.eval()
    heads.eval()
    predictions: list[Mapping[str, Any]] = []
    with torch.inference_mode():
        for episode in cached_episodes:
            latent = None
            for row, raw_hidden in episode:
                fused, latent, _plan_logits, _depth = hybrid(raw_hidden, latent)
                predictions.append(
                    _prediction_from_hidden(
                        torch,
                        row,
                        fused,
                        heads=heads,
                        action_names=ACTION_VOCAB,
                    )
                )
    return {"summary": summarize_predictions(predictions), "predictions": predictions}


def _evaluate_loaded(
    torch: Any,
    rows: Sequence[RecurrentTransition],
    *,
    model: Any,
    tokenizer: Any,
    hybrid: Any,
    heads: Any,
    device: str,
) -> Mapping[str, Any]:
    model.eval()
    hybrid.eval()
    heads.eval()
    predictions: list[Mapping[str, Any]] = []
    for episode in group_episodes(rows):
        rwkv_state: Any = None
        latent: Any = None
        for row in episode:
            encoded = tokenizer(
                _event_text(row), return_tensors="pt", add_special_tokens=False
            )
            kwargs: dict[str, Any] = {
                "input_ids": encoded["input_ids"].to(device),
                "use_cache": True,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if rwkv_state is not None:
                kwargs["state"] = rwkv_state
            with torch.inference_mode():
                outputs = model(**kwargs)
                hidden_states = getattr(outputs, "hidden_states", None)
                rwkv_state = getattr(outputs, "state", None)
                if not hidden_states or rwkv_state is None:
                    raise RWKVControllerError(
                        "RWKV forward must return hidden_states and recurrent state"
                    )
                raw_hidden = hidden_states[-1][:, -1, :].detach().float()
                fused, latent, _plan_logits, _depth = hybrid(raw_hidden, latent)
            predictions.append(
                _prediction_from_hidden(
                    torch,
                    row,
                    fused,
                    heads=heads,
                    action_names=ACTION_VOCAB,
                )
            )
    return {"summary": summarize_predictions(predictions), "predictions": predictions}


def train_hybrid_controller(
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
    plan_loss_weight: float = 0.25,
    argument_loss_weight: float = 0.5,
    max_argument_chars: int = 8192,
    latent_dim: int = 256,
    latent_slots: int = 8,
    min_reasoning_steps: int = 2,
    max_reasoning_steps: int = 6,
    convergence_tolerance: float = 1e-3,
    init_controller: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> Mapping[str, Any]:
    try:
        import torch
        from safetensors.torch import save_file, save_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RWKVControllerError(
            "hybrid training requires `pip install -e '.[rwkv]'`"
        ) from exc

    resolved_mode = str(backbone_mode).strip().lower()
    if resolved_mode not in BACKBONE_MODES:
        raise ValueError(f"backbone_mode must be one of {BACKBONE_MODES}")

    rows = load_transitions(transitions_path)
    if not rows:
        raise ValueError("transition dataset is empty")
    train_rows, validation_rows = split_transitions(
        rows, validation_fraction=validation_fraction, seed=seed
    )
    if not train_rows:
        raise ValueError("training split is empty")

    pointer_examples = [
        row for row in rows if row.target_pointer is not None and int(row.target_pointer) >= 0
    ]
    max_pointer = max((int(row.target_pointer) for row in pointer_examples), default=-1)
    pointer_slots = max(DEFAULT_POINTER_SLOTS, max_pointer + 1) if pointer_examples else 0

    random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = resolve_device(torch, device)
    try:
        dtype_name, dtype_value = resolve_dtype(torch, resolved_device, dtype)
    except ValueError as exc:
        raise RWKVControllerError(str(exc)) from exc

    config = HybridLatentConfig(
        latent_dim=latent_dim,
        slots=latent_slots,
        min_reasoning_steps=min_reasoning_steps,
        max_reasoning_steps=max_reasoning_steps,
        convergence_tolerance=convergence_tolerance,
    ).normalized()
    epoch_count = max(1, int(epochs))
    started = time.monotonic()
    _emit(
        progress,
        "model-load-start",
        architecture="hybrid-latent-rwkv",
        model_id=model_id,
        device=resolved_device,
        dtype=dtype_name,
        backbone_mode=resolved_mode,
        transitions=len(rows),
        train_transitions=len(train_rows),
        validation_transitions=len(validation_rows),
        pointer_slots=pointer_slots,
        hybrid_config=dict(config.to_dict()),
        epochs=epoch_count,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype_value,
        trust_remote_code=True,
    ).to(resolved_device)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RWKVControllerError("RWKV model config does not expose hidden_size")
    hybrid = make_hybrid_core(
        torch,
        hidden_size=hidden_size,
        config=config,
        device=resolved_device,
    )
    heads = _make_heads(torch, hidden_size, pointer_slots, resolved_device)
    initialization = _load_initial_controller(
        init_controller,
        model=model,
        hybrid=hybrid,
        heads=heads,
        model_id=model_id,
        device=resolved_device,
    )
    action_index = {name: index for index, name in enumerate(ACTION_VOCAB)}

    frozen = resolved_mode == "frozen"
    model.eval() if frozen else model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(not frozen)
    if not frozen and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    hybrid.train()
    heads.train()

    all_groups = [list(episode) for episode in group_episodes(rows)]
    train_ids = {row.episode_id for row in train_rows}
    validation_ids = {row.episode_id for row in validation_rows}
    cached_episodes: list[list[tuple[RecurrentTransition, Any]]] = []
    processed_tokens = 0
    max_sequence_tokens = 0

    if frozen:
        _emit(progress, "feature-cache-start", episodes=len(all_groups), transitions=len(rows))
        for episode_index, episode in enumerate(all_groups, start=1):
            state: Any = None
            cached_episode: list[tuple[RecurrentTransition, Any]] = []
            for row in episode:
                encoded = tokenizer(
                    _event_text(row), return_tensors="pt", add_special_tokens=False
                )
                input_ids = encoded["input_ids"].to(resolved_device)
                processed_tokens += int(input_ids.shape[-1])
                max_sequence_tokens = max(max_sequence_tokens, int(input_ids.shape[-1]))
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
                cached_episode.append(
                    (row, hidden_states[-1][:, -1, :].detach().float())
                )
            cached_episodes.append(cached_episode)
            _emit(
                progress,
                "feature-cache-progress",
                episode=episode_index,
                episodes=len(all_groups),
                transitions=sum(len(item) for item in cached_episodes),
                total_transitions=len(rows),
                tokens=processed_tokens,
                max_event_tokens=max_sequence_tokens,
            )

    cached_train = [
        episode for episode in cached_episodes
        if episode and episode[0][0].episode_id in train_ids
    ]
    cached_validation = [
        episode for episode in cached_episodes
        if episode and episode[0][0].episode_id in validation_ids
    ]

    trainable_core = list(hybrid.parameters()) + list(heads.parameters())
    if frozen:
        optimizer = torch.optim.AdamW(trainable_core, lr=float(learning_rate))
    else:
        optimizer = torch.optim.AdamW([
            {"params": trainable_core, "lr": float(learning_rate)},
            {
                "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                "lr": float(backbone_learning_rate),
            },
        ])

    total_updates = 0
    argument_examples = 0
    last_loss = 0.0
    epoch_history: list[Mapping[str, Any]] = []

    for epoch_index in range(1, epoch_count + 1):
        epoch_started = time.monotonic()
        epoch_loss = 0.0
        epoch_updates = 0
        action_correct = action_total = 0
        stop_correct = stop_total = 0
        pointer_correct = pointer_total = 0
        plan_examples = 0
        argument_epoch = 0

        episode_groups: list[Any]
        if frozen:
            episode_groups = list(cached_train)
        else:
            episode_groups = [list(episode) for episode in group_episodes(train_rows)]
        random.shuffle(episode_groups)

        for episode in episode_groups:
            optimizer.zero_grad(set_to_none=True)
            if frozen:
                row_hiddens = episode
            else:
                row_hiddens, token_count, episode_max = _full_episode_hidden_states(
                    model,
                    tokenizer,
                    episode,
                    device=resolved_device,
                )
                processed_tokens += token_count
                max_sequence_tokens = max(max_sequence_tokens, episode_max)

            latent = None
            losses: list[Any] = []
            argument_losses: list[Any] = []
            for row, raw_hidden in row_hiddens:
                fused, latent, plan_logits, _depth = hybrid(raw_hidden, latent)
                control, accuracy = _control_loss(
                    torch,
                    row,
                    fused,
                    heads=heads,
                    action_index=action_index,
                    device=resolved_device,
                    pointer_loss_weight=pointer_loss_weight,
                )
                plan = _repair_plan_loss(torch, row, plan_logits)
                losses.append(control + float(plan_loss_weight) * plan)
                plan_examples += 1
                action_correct += int(accuracy["action_correct"])
                action_total += 1
                stop_correct += int(accuracy["stop_correct"])
                stop_total += 1
                if accuracy["pointer_correct"] is not None:
                    pointer_correct += int(accuracy["pointer_correct"])
                    pointer_total += 1

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
                        argument_losses.append(arg_loss)
                        argument_examples += 1
                        argument_epoch += 1

            if not losses:
                continue
            episode_loss = torch.stack(losses).mean()
            if argument_losses:
                episode_loss = episode_loss + float(argument_loss_weight) * torch.stack(
                    argument_losses
                ).mean()
            episode_loss.backward()
            params = trainable_core + [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_updates += 1
            epoch_updates += 1
            last_loss = float(episode_loss.detach().cpu().item())
            epoch_loss += last_loss

        summary = {
            "epoch": epoch_index,
            "loss": epoch_loss / epoch_updates if epoch_updates else 0.0,
            "action_accuracy": action_correct / action_total if action_total else 0.0,
            "stop_accuracy": stop_correct / stop_total if stop_total else 0.0,
            "pointer_accuracy": pointer_correct / pointer_total if pointer_total else None,
            "pointer_examples": pointer_total,
            "plan_examples": plan_examples,
            "argument_examples": argument_epoch,
            "updates": epoch_updates,
            "elapsed_seconds": time.monotonic() - epoch_started,
        }
        epoch_history.append(summary)
        _emit(progress, "epoch-done", epochs=epoch_count, **summary)

    model.eval()
    hybrid.eval()
    heads.eval()
    if frozen:
        train_evaluation = _evaluate_cached(
            torch, cached_train, hybrid=hybrid, heads=heads
        )
        validation_evaluation = (
            _evaluate_cached(torch, cached_validation, hybrid=hybrid, heads=heads)
            if cached_validation else {"summary": None, "predictions": []}
        )
    else:
        train_evaluation = _evaluate_loaded(
            torch,
            train_rows,
            model=model,
            tokenizer=tokenizer,
            hybrid=hybrid,
            heads=heads,
            device=resolved_device,
        )
        validation_evaluation = (
            _evaluate_loaded(
                torch,
                validation_rows,
                model=model,
                tokenizer=tokenizer,
                hybrid=hybrid,
                heads=heads,
                device=resolved_device,
            )
            if validation_rows else {"summary": None, "predictions": []}
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    head_path = output / "heads.safetensors"
    hybrid_path = output / "hybrid.safetensors"
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in heads.state_dict().items()},
        str(head_path),
    )
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in hybrid.state_dict().items()},
        str(hybrid_path),
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
        "format": HYBRID_TRAINING_FORMAT,
        "controller_format": HYBRID_CONTROLLER_FORMAT,
        "controller_type": "hybrid-latent-rwkv",
        "model_id": model_id,
        "hidden_size": hidden_size,
        "action_vocab": list(ACTION_VOCAB),
        "pointer_slots": pointer_slots,
        "stop_threshold": 0.5,
        "max_argument_tokens": 512,
        "hybrid_config": dict(config.to_dict()),
        "device": resolved_device,
        "dtype": dtype_name,
        "epochs": epoch_count,
        "learning_rate": float(learning_rate),
        "backbone_learning_rate": float(backbone_learning_rate),
        "backbone_mode": resolved_mode,
        "pointer_loss_weight": float(pointer_loss_weight),
        "plan_loss_weight": float(plan_loss_weight),
        "argument_loss_weight": float(argument_loss_weight),
        "init_controller": initialization["controller"],
        "init_head_tensors_loaded": initialization["head_tensors_loaded"],
        "init_hybrid_tensors_loaded": initialization["hybrid_tensors_loaded"],
        "init_backbone_loaded": initialization["backbone_loaded"],
        "transitions": len(rows),
        "episodes": len(group_episodes(rows)),
        "train_transitions": len(train_rows),
        "validation_transitions": len(validation_rows),
        "updates": total_updates,
        "final_loss": last_loss,
        "train_action_accuracy": train_summary["action_accuracy"],
        "train_pointer_accuracy": train_summary.get("pointer_accuracy"),
        "train_stop_accuracy": train_summary["stop_accuracy"],
        "train_exact_episode_accuracy": train_summary["exact_episode_accuracy"],
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
        "argument_examples": argument_examples,
        "pointer_examples": len(pointer_examples),
        "plan_examples": len(train_rows) * epoch_count,
        "training_tokens": processed_tokens,
        "training_max_sequence_tokens": max_sequence_tokens,
        "elapsed_seconds": time.monotonic() - started,
        "weights": head_path.name,
        "hybrid_weights": hybrid_path.name,
        "backbone_weights": backbone_weights,
        "evaluation": "evaluation.json",
        "patch_decoder": "shared-rwkv-language-head-dedicated-repair-contract",
        "verifier_feedback": "structured-runtime-feedback",
        "seed": seed,
    }
    (output / "controller.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit(
        progress,
        "training-done",
        architecture="hybrid-latent-rwkv",
        elapsed_seconds=summary["elapsed_seconds"],
        updates=total_updates,
        output=str(output),
    )
    return summary


__all__ = [
    "BACKBONE_MODES",
    "HYBRID_TRAINING_FORMAT",
    "train_hybrid_controller",
]
