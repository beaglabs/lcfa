"""Pointer-only training for the LCFA pairwise semantic reranker v3."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, MutableMapping, Sequence

from .hybrid_latent import HYBRID_CONTROLLER_FORMAT, HybridLatentConfig, make_hybrid_core
from .hybrid_semantic_train import HYBRID_TRAINING_FORMAT, ProgressCallback
from .hybrid_train import _emit, _event_text, _load_initial_controller, _make_heads
from .pairwise_pointer import (
    PAIRWISE_POINTER_ARCHITECTURE,
    PAIRWISE_SEMANTIC_POINTER_FORMAT,
    make_pairwise_semantic_pointer,
    pairwise_pointer_prior_logits,
)
from .recurrent_eval import group_episodes, split_transitions, summarize_predictions
from .recurrent_transitions import ACTION_VOCAB, RecurrentTransition, load_transitions
from .rwkv_controller import DEFAULT_POINTER_SLOTS, DEFAULT_RWKV_MODEL, RWKVControllerError
from .semantic_pointer import candidates_from_transition, encode_candidate_semantics
from .torch_runtime import resolve_device, resolve_dtype


def _candidate_embeddings(
    torch: Any,
    model: Any,
    tokenizer: Any,
    candidates: Sequence[Any],
    *,
    device: str,
    cache: MutableMapping[str, Any],
) -> Any:
    return encode_candidate_semantics(
        torch,
        model,
        tokenizer,
        candidates,
        device=device,
        cache=cache,
    )


def _pointer_objective(
    torch: Any,
    row: RecurrentTransition,
    hidden: Any,
    latent: Any,
    *,
    pointer: Any,
    model: Any,
    tokenizer: Any,
    candidate_cache: MutableMapping[str, Any],
    device: str,
    margin: float,
    margin_loss_weight: float,
) -> tuple[Any | None, Mapping[str, Any]]:
    if row.target_pointer is None:
        return None, {}
    candidates = candidates_from_transition(row)
    if not candidates:
        return None, {}
    target_index = int(row.target_pointer)
    local_target = next(
        (index for index, candidate in enumerate(candidates) if candidate.index == target_index),
        None,
    )
    if local_target is None:
        return None, {}
    embeddings = _candidate_embeddings(
        torch,
        model,
        tokenizer,
        candidates,
        device=device,
        cache=candidate_cache,
    )
    logits, prior, residual = pointer.components(hidden, latent, candidates, embeddings)
    expected = torch.tensor([local_target], device=device, dtype=torch.long)
    ce = torch.nn.functional.cross_entropy(logits.float(), expected)
    target_score = logits[0, local_target]
    if len(candidates) > 1:
        wrong_mask = torch.ones(len(candidates), device=device, dtype=torch.bool)
        wrong_mask[local_target] = False
        highest_wrong = logits[0, wrong_mask].max()
        target_margin = target_score - highest_wrong
        margin_loss = torch.relu(
            torch.tensor(float(margin), device=device, dtype=torch.float32) - target_margin
        )
    else:
        target_margin = torch.tensor(float("inf"), device=device)
        margin_loss = logits.sum() * 0.0
    loss = ce + float(margin_loss_weight) * margin_loss

    predicted_local = int(torch.argmax(logits.detach(), dim=-1)[0].item())
    prior_local = int(torch.argmax(prior.detach(), dim=-1)[0].item())
    predicted_index = candidates[predicted_local].index
    prior_index = candidates[prior_local].index
    residual_abs = float(residual.detach().abs().mean().item())
    finite_margin = (
        float(target_margin.detach().item())
        if bool(torch.isfinite(target_margin.detach()).item())
        else None
    )
    return loss, {
        "pointer_correct": int(predicted_index == target_index),
        "prior_correct": int(prior_index == target_index),
        "override": int(predicted_index != prior_index),
        "prior_miss_corrected": int(prior_index != target_index and predicted_index == target_index),
        "prior_hit_regressed": int(prior_index == target_index and predicted_index != target_index),
        "target_margin": finite_margin,
        "residual_abs": residual_abs,
    }


def _control_prediction(
    torch: Any,
    row: RecurrentTransition,
    hidden: Any,
    *,
    heads: Any,
) -> tuple[str, bool, float, float]:
    with torch.no_grad():
        action_logits = heads["action"](hidden)
        action_local = int(torch.argmax(action_logits, dim=-1)[0].item())
        stop_probability = float(torch.sigmoid(heads["stop"](hidden).float())[0, 0].item())
        value = float(torch.sigmoid(heads["value"](hidden).float())[0, 0].item())
    return ACTION_VOCAB[action_local], stop_probability >= 0.5, stop_probability, value


def _pointer_prediction(
    torch: Any,
    row: RecurrentTransition,
    hidden: Any,
    latent: Any,
    *,
    pointer: Any,
    model: Any,
    tokenizer: Any,
    candidate_cache: MutableMapping[str, Any],
    device: str,
) -> Mapping[str, Any]:
    candidates = candidates_from_transition(row)
    if not candidates:
        return {
            "predicted_pointer": None,
            "pointer_confidence": None,
            "prior_pointer": None,
            "prior_pointer_confidence": None,
            "target_margin": None,
            "residual_abs": None,
        }
    embeddings = _candidate_embeddings(
        torch,
        model,
        tokenizer,
        candidates,
        device=device,
        cache=candidate_cache,
    )
    with torch.no_grad():
        logits, prior, residual = pointer.components(hidden, latent, candidates, embeddings)
        probs = torch.softmax(logits.float(), dim=-1)[0]
        prior_probs = torch.softmax(prior.float(), dim=-1)[0]
        local = int(torch.argmax(probs).item())
        prior_local = int(torch.argmax(prior_probs).item())
        predicted_pointer = candidates[local].index
        prior_pointer = candidates[prior_local].index
        target_margin = None
        if row.target_pointer is not None:
            local_target = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.index == int(row.target_pointer)
                ),
                None,
            )
            if local_target is not None and len(candidates) > 1:
                wrong_mask = torch.ones(len(candidates), device=device, dtype=torch.bool)
                wrong_mask[local_target] = False
                margin_value = logits[0, local_target] - logits[0, wrong_mask].max()
                target_margin = float(margin_value.item())
        return {
            "predicted_pointer": predicted_pointer,
            "pointer_confidence": float(probs[local].item()),
            "prior_pointer": prior_pointer,
            "prior_pointer_confidence": float(prior_probs[prior_local].item()),
            "target_margin": target_margin,
            "residual_abs": float(residual.abs().mean().item()),
        }


def _diagnostics(predictions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    pointer_rows = [row for row in predictions if row.get("target_pointer") is not None]
    if not pointer_rows:
        return {
            "override_rate": None,
            "prior_miss_count": 0,
            "prior_miss_corrected": 0,
            "prior_miss_correction_rate": None,
            "prior_hit_count": 0,
            "prior_hit_regressed": 0,
            "prior_hit_preservation_rate": None,
            "mean_target_margin": None,
            "mean_abs_residual": None,
        }
    overrides = sum(row.get("predicted_pointer") != row.get("prior_pointer") for row in pointer_rows)
    prior_misses = [row for row in pointer_rows if not row.get("prior_pointer_correct")]
    prior_hits = [row for row in pointer_rows if row.get("prior_pointer_correct")]
    corrected = sum(bool(row.get("pointer_correct")) for row in prior_misses)
    regressed = sum(not bool(row.get("pointer_correct")) for row in prior_hits)
    margins = [float(row["target_margin"]) for row in pointer_rows if row.get("target_margin") is not None]
    residuals = [float(row["residual_abs"]) for row in pointer_rows if row.get("residual_abs") is not None]
    return {
        "override_rate": overrides / len(pointer_rows),
        "prior_miss_count": len(prior_misses),
        "prior_miss_corrected": corrected,
        "prior_miss_correction_rate": corrected / len(prior_misses) if prior_misses else None,
        "prior_hit_count": len(prior_hits),
        "prior_hit_regressed": regressed,
        "prior_hit_preservation_rate": (
            (len(prior_hits) - regressed) / len(prior_hits) if prior_hits else None
        ),
        "mean_target_margin": sum(margins) / len(margins) if margins else None,
        "mean_abs_residual": sum(residuals) / len(residuals) if residuals else None,
    }


def _evaluate_cached(
    torch: Any,
    episodes: Sequence[Sequence[tuple[RecurrentTransition, Any]]],
    *,
    hybrid: Any,
    heads: Any,
    pointer: Any,
    model: Any,
    tokenizer: Any,
    candidate_cache: MutableMapping[str, Any],
    device: str,
) -> Mapping[str, Any]:
    predictions: list[Mapping[str, Any]] = []
    hybrid.eval()
    heads.eval()
    pointer.eval()
    with torch.no_grad():
        for episode in episodes:
            latent = None
            for row, raw_hidden in episode:
                fused, latent, _plan_logits, _depth = hybrid(raw_hidden, latent)
                fused = fused.detach()
                latent = latent.detach()
                predicted_action, predicted_stop, stop_probability, predicted_value = _control_prediction(
                    torch,
                    row,
                    fused,
                    heads=heads,
                )
                pointer_values = _pointer_prediction(
                    torch,
                    row,
                    fused,
                    latent,
                    pointer=pointer,
                    model=model,
                    tokenizer=tokenizer,
                    candidate_cache=candidate_cache,
                    device=device,
                )
                target_pointer = row.target_pointer
                predicted_pointer = pointer_values["predicted_pointer"]
                prior_pointer = pointer_values["prior_pointer"]
                predictions.append({
                    "episode_id": row.episode_id,
                    "step_index": row.step_index,
                    "target_action": row.target_action,
                    "predicted_action": predicted_action,
                    "action_correct": predicted_action == row.target_action,
                    "target_pointer": target_pointer,
                    "predicted_pointer": predicted_pointer,
                    "pointer_confidence": pointer_values["pointer_confidence"],
                    "pointer_correct": (
                        None if target_pointer is None else predicted_pointer == target_pointer
                    ),
                    "prior_pointer": prior_pointer,
                    "prior_pointer_confidence": pointer_values["prior_pointer_confidence"],
                    "prior_pointer_correct": (
                        None if target_pointer is None else prior_pointer == target_pointer
                    ),
                    "target_margin": pointer_values["target_margin"],
                    "residual_abs": pointer_values["residual_abs"],
                    "target_stop": row.stop_target,
                    "predicted_stop": predicted_stop,
                    "stop_probability": stop_probability,
                    "stop_correct": predicted_stop == row.stop_target,
                    "value_target": row.value_target,
                    "predicted_value": predicted_value,
                })
    diagnostics = _diagnostics(predictions)
    prior_values = [
        bool(row["prior_pointer_correct"])
        for row in predictions
        if row.get("prior_pointer_correct") is not None
    ]
    return {
        "summary": summarize_predictions(predictions),
        "prior_pointer_accuracy": (
            sum(prior_values) / len(prior_values) if prior_values else None
        ),
        "diagnostics": diagnostics,
        "predictions": predictions,
    }


def train_pairwise_pointer_controller(
    transitions_path: str | Path,
    output_dir: str | Path,
    *,
    init_controller: str | Path,
    model_id: str = DEFAULT_RWKV_MODEL,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str | None = "auto",
    dtype: str = "auto",
    validation_fraction: float = 0.2,
    seed: int = 20260925,
    pointer_loss_weight: float = 1.0,
    pointer_margin: float = 1.0,
    margin_loss_weight: float = 0.5,
    latent_dim: int = 256,
    latent_slots: int = 9,
    min_reasoning_steps: int = 2,
    max_reasoning_steps: int = 6,
    convergence_tolerance: float = 1e-3,
    progress: ProgressCallback | None = None,
    **_ignored: Any,
) -> Mapping[str, Any]:
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RWKVControllerError("pairwise pointer training requires `pip install -e '.[rwkv]'`") from exc

    rows = load_transitions(transitions_path)
    if not rows:
        raise ValueError("transition dataset is empty")
    train_rows, validation_rows = split_transitions(
        rows,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if not train_rows:
        raise ValueError("training split is empty")
    pointer_examples = [
        row
        for row in rows
        if row.target_pointer is not None
        and any(
            candidate.index == int(row.target_pointer)
            for candidate in candidates_from_transition(row)
        )
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
    pointer = make_pairwise_semantic_pointer(
        torch,
        hidden_size=hidden_size,
        latent_dim=config.latent_dim,
        device=resolved_device,
    )

    model.eval()
    hybrid.eval()
    heads.eval()
    for module in (model, hybrid, heads):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    pointer.train()
    for parameter in pointer.parameters():
        parameter.requires_grad_(True)

    all_groups = [list(episode) for episode in group_episodes(rows)]
    train_ids = {row.episode_id for row in train_rows}
    validation_ids = {row.episode_id for row in validation_rows}
    cached_episodes: list[list[tuple[RecurrentTransition, Any]]] = []
    processed_tokens = 0
    max_sequence_tokens = 0
    _emit(
        progress,
        "model-load-start",
        architecture=PAIRWISE_POINTER_ARCHITECTURE,
        model_id=model_id,
        device=resolved_device,
        dtype=dtype_name,
        backbone_mode="frozen",
        training_scope="pointer-only",
        transitions=len(rows),
        train_transitions=len(train_rows),
        validation_transitions=len(validation_rows),
        semantic_pointer_examples=len(pointer_examples),
        epochs=epoch_count,
    )
    _emit(progress, "feature-cache-start", episodes=len(all_groups), transitions=len(rows))
    for episode_index, episode in enumerate(all_groups, start=1):
        state: Any = None
        cached_episode: list[tuple[RecurrentTransition, Any]] = []
        for row in episode:
            encoded = tokenizer(_event_text(row), return_tensors="pt", add_special_tokens=False)
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
            with torch.no_grad():
                outputs = model(**kwargs)
            hidden_states = getattr(outputs, "hidden_states", None)
            state = getattr(outputs, "state", None)
            if not hidden_states or state is None:
                raise RWKVControllerError("RWKV forward must return hidden_states and recurrent state")
            cached_episode.append((row, hidden_states[-1][:, -1, :].detach().float()))
        cached_episodes.append(cached_episode)
        _emit(
            progress,
            "feature-cache-progress",
            episode=episode_index,
            episodes=len(all_groups),
            transitions=sum(len(item) for item in cached_episodes),
            total_transitions=len(rows),
            tokens=processed_tokens,
        )

    cached_train = [
        episode
        for episode in cached_episodes
        if episode and episode[0][0].episode_id in train_ids
    ]
    cached_validation = [
        episode
        for episode in cached_episodes
        if episode and episode[0][0].episode_id in validation_ids
    ]
    candidate_cache: dict[str, Any] = {}
    optimizer = torch.optim.AdamW(pointer.parameters(), lr=float(learning_rate))
    total_updates = 0
    last_loss = 0.0
    epoch_history: list[Mapping[str, Any]] = []

    for epoch_index in range(1, epoch_count + 1):
        epoch_started = time.monotonic()
        groups = list(cached_train)
        random.shuffle(groups)
        epoch_loss = 0.0
        epoch_updates = 0
        pointer_correct = pointer_total = 0
        prior_correct = prior_total = 0
        action_correct = action_total = 0
        stop_correct = stop_total = 0
        margins: list[float] = []
        residuals: list[float] = []
        for episode in groups:
            optimizer.zero_grad(set_to_none=True)
            latent = None
            losses: list[Any] = []
            for row, raw_hidden in episode:
                with torch.no_grad():
                    fused, latent, _plan_logits, _depth = hybrid(raw_hidden, latent)
                    fused = fused.detach()
                    latent = latent.detach()
                    predicted_action, predicted_stop, _stop_probability, _value = _control_prediction(
                        torch,
                        row,
                        fused,
                        heads=heads,
                    )
                action_correct += int(predicted_action == row.target_action)
                action_total += 1
                stop_correct += int(predicted_stop == row.stop_target)
                stop_total += 1
                pointer_loss, stats = _pointer_objective(
                    torch,
                    row,
                    fused,
                    latent,
                    pointer=pointer,
                    model=model,
                    tokenizer=tokenizer,
                    candidate_cache=candidate_cache,
                    device=resolved_device,
                    margin=pointer_margin,
                    margin_loss_weight=margin_loss_weight,
                )
                if pointer_loss is None:
                    continue
                losses.append(float(pointer_loss_weight) * pointer_loss)
                pointer_correct += int(stats["pointer_correct"])
                pointer_total += 1
                prior_correct += int(stats["prior_correct"])
                prior_total += 1
                if stats.get("target_margin") is not None:
                    margins.append(float(stats["target_margin"]))
                residuals.append(float(stats["residual_abs"]))
            if not losses:
                continue
            episode_loss = torch.stack(losses).mean()
            episode_loss.backward()
            torch.nn.utils.clip_grad_norm_(pointer.parameters(), 1.0)
            optimizer.step()
            last_loss = float(episode_loss.detach().cpu().item())
            epoch_loss += last_loss
            epoch_updates += 1
            total_updates += 1
        summary = {
            "epoch": epoch_index,
            "loss": epoch_loss / epoch_updates if epoch_updates else 0.0,
            "action_accuracy": action_correct / action_total if action_total else 0.0,
            "stop_accuracy": stop_correct / stop_total if stop_total else 0.0,
            "pointer_accuracy": pointer_correct / pointer_total if pointer_total else None,
            "retrieval_prior_accuracy": prior_correct / prior_total if prior_total else None,
            "pointer_examples": pointer_total,
            "plan_examples": 0,
            "argument_examples": 0,
            "mean_target_margin": sum(margins) / len(margins) if margins else None,
            "mean_abs_residual": sum(residuals) / len(residuals) if residuals else None,
            "updates": epoch_updates,
            "elapsed_seconds": time.monotonic() - epoch_started,
        }
        epoch_history.append(summary)
        _emit(progress, "epoch-done", epochs=epoch_count, **summary)

    pointer.eval()
    train_evaluation = _evaluate_cached(
        torch,
        cached_train,
        hybrid=hybrid,
        heads=heads,
        pointer=pointer,
        model=model,
        tokenizer=tokenizer,
        candidate_cache=candidate_cache,
        device=resolved_device,
    )
    validation_evaluation = (
        _evaluate_cached(
            torch,
            cached_validation,
            hybrid=hybrid,
            heads=heads,
            pointer=pointer,
            model=model,
            tokenizer=tokenizer,
            candidate_cache=candidate_cache,
            device=resolved_device,
        )
        if cached_validation
        else {
            "summary": None,
            "prior_pointer_accuracy": None,
            "diagnostics": _diagnostics(()),
            "predictions": [],
        }
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    head_path = output / "heads.safetensors"
    hybrid_path = output / "hybrid.safetensors"
    pointer_path = output / "semantic_pointer.safetensors"
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in heads.state_dict().items()},
        str(head_path),
    )
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in hybrid.state_dict().items()},
        str(hybrid_path),
    )
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in pointer.state_dict().items()},
        str(pointer_path),
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
    train_diag = train_evaluation["diagnostics"]
    validation_diag = validation_evaluation["diagnostics"]
    prior_strength = float(pointer.prior_strength.detach().float().clamp(0.0, 2.0).item())
    summary = {
        "format": HYBRID_TRAINING_FORMAT,
        "controller_format": HYBRID_CONTROLLER_FORMAT,
        "controller_type": "hybrid-latent-rwkv",
        "model_id": model_id,
        "hidden_size": hidden_size,
        "action_vocab": list(ACTION_VOCAB),
        "pointer_slots": pointer_slots,
        "pointer_architecture": PAIRWISE_POINTER_ARCHITECTURE,
        "pointer_format": PAIRWISE_SEMANTIC_POINTER_FORMAT,
        "candidate_encoder": "frozen-rwkv-last-hidden-mean+lexical-metadata",
        "pointer_prior": "deterministic-retrieval",
        "pointer_prior_strength": prior_strength,
        "pointer_margin": float(pointer_margin),
        "pointer_margin_loss_weight": float(margin_loss_weight),
        "pointer_initialization": "fresh-v3-zero-residual-head",
        "semantic_pointer_weights": pointer_path.name,
        "semantic_pointer_init_tensors_loaded": 0,
        "semantic_pointer_examples": len(pointer_examples),
        "semantic_pointer_trainable_parameters": sum(
            parameter.numel() for parameter in pointer.parameters()
        ),
        "stop_threshold": 0.5,
        "max_argument_tokens": 512,
        "hybrid_config": dict(config.to_dict()),
        "device": resolved_device,
        "dtype": dtype_name,
        "epochs": epoch_count,
        "learning_rate": float(learning_rate),
        "backbone_learning_rate": 0.0,
        "backbone_mode": "frozen",
        "training_scope": "pointer-only",
        "pointer_loss_weight": float(pointer_loss_weight),
        "plan_loss_weight": 0.0,
        "argument_loss_weight": 0.0,
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
        "train_retrieval_prior_accuracy": train_evaluation.get("prior_pointer_accuracy"),
        "train_stop_accuracy": train_summary["stop_accuracy"],
        "train_exact_episode_accuracy": train_summary["exact_episode_accuracy"],
        "validation_action_accuracy": validation_summary["action_accuracy"] if validation_summary else None,
        "validation_pointer_accuracy": validation_summary.get("pointer_accuracy") if validation_summary else None,
        "validation_retrieval_prior_accuracy": validation_evaluation.get("prior_pointer_accuracy"),
        "validation_stop_accuracy": validation_summary["stop_accuracy"] if validation_summary else None,
        "validation_exact_episode_accuracy": validation_summary["exact_episode_accuracy"] if validation_summary else None,
        "train_override_rate": train_diag["override_rate"],
        "train_prior_miss_count": train_diag["prior_miss_count"],
        "train_prior_miss_corrected": train_diag["prior_miss_corrected"],
        "train_prior_miss_correction_rate": train_diag["prior_miss_correction_rate"],
        "train_prior_hit_count": train_diag["prior_hit_count"],
        "train_prior_hit_regressed": train_diag["prior_hit_regressed"],
        "train_prior_hit_preservation_rate": train_diag["prior_hit_preservation_rate"],
        "train_mean_target_margin": train_diag["mean_target_margin"],
        "train_mean_abs_residual": train_diag["mean_abs_residual"],
        "validation_override_rate": validation_diag["override_rate"],
        "validation_prior_miss_count": validation_diag["prior_miss_count"],
        "validation_prior_miss_corrected": validation_diag["prior_miss_corrected"],
        "validation_prior_miss_correction_rate": validation_diag["prior_miss_correction_rate"],
        "validation_prior_hit_count": validation_diag["prior_hit_count"],
        "validation_prior_hit_regressed": validation_diag["prior_hit_regressed"],
        "validation_prior_hit_preservation_rate": validation_diag["prior_hit_preservation_rate"],
        "validation_mean_target_margin": validation_diag["mean_target_margin"],
        "validation_mean_abs_residual": validation_diag["mean_abs_residual"],
        "argument_examples": 0,
        "pointer_examples": len(pointer_examples),
        "plan_examples": 0,
        "training_tokens": processed_tokens,
        "training_max_sequence_tokens": max_sequence_tokens,
        "elapsed_seconds": time.monotonic() - started,
        "weights": head_path.name,
        "hybrid_weights": hybrid_path.name,
        "backbone_weights": None,
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
        architecture=PAIRWISE_POINTER_ARCHITECTURE,
        elapsed_seconds=summary["elapsed_seconds"],
        updates=total_updates,
        output=str(output),
    )
    return summary


__all__ = ["train_pairwise_pointer_controller"]
