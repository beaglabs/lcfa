"""Training dispatcher for hybrid LCFA controllers."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from .hybrid_semantic_train import (
    BACKBONE_MODES,
    train_hybrid_controller as train_joint_hybrid_controller,
)
from .pairwise_pointer_train import train_pairwise_pointer_controller

TRAINING_SCOPES = ("joint", "pointer-only")


def _preserve_init_backbone(
    init_controller: str | Path,
    output_dir: str | Path,
    summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Carry a fine-tuned backbone through pointer-only training artifacts.

    Pointer-only training freezes the initialized backbone. If the initializer
    already contains a fine-tuned backbone, the resulting v3 controller must
    retain it rather than silently reverting to the base Hugging Face model.
    """
    init_root = Path(init_controller).expanduser().resolve()
    init_manifest_path = init_root / "controller.json" if init_root.is_dir() else init_root
    if not init_manifest_path.is_file():
        return summary
    init_manifest = json.loads(init_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(init_manifest, Mapping):
        return summary
    backbone_name = str(init_manifest.get("backbone_weights") or "").strip()
    if not backbone_name:
        return summary
    source = init_manifest_path.parent / backbone_name
    if not source.is_file():
        return summary

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "backbone.safetensors"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)

    manifest_path = output / "controller.json"
    payload = dict(summary)
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            payload = dict(raw)
    payload["backbone_weights"] = destination.name
    payload["init_backbone_loaded"] = True
    payload["pointer_only_backbone_preserved"] = True
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def train_hybrid_controller(
    transitions_path: Any,
    output_dir: Any,
    *,
    training_scope: str = "joint",
    init_controller: Any = None,
    progress: Any = None,
    **kwargs: Any,
) -> Mapping[str, Any]:
    scope = str(training_scope).strip().lower()
    if scope not in TRAINING_SCOPES:
        raise ValueError(f"training_scope must be one of {TRAINING_SCOPES}")
    if scope == "pointer-only":
        if init_controller is None:
            raise ValueError("pointer-only training requires --init-controller")
        if str(kwargs.get("backbone_mode", "frozen")).strip().lower() != "frozen":
            raise ValueError("pointer-only training requires --backbone-mode frozen")
        summary = train_pairwise_pointer_controller(
            transitions_path,
            output_dir,
            init_controller=init_controller,
            progress=progress,
            **kwargs,
        )
        return _preserve_init_backbone(init_controller, output_dir, summary)
    joint_kwargs = dict(kwargs)
    joint_kwargs.pop("pointer_margin", None)
    joint_kwargs.pop("margin_loss_weight", None)
    return train_joint_hybrid_controller(
        transitions_path,
        output_dir,
        training_scope="joint",
        init_controller=init_controller,
        progress=progress,
        **joint_kwargs,
    )


__all__ = [
    "BACKBONE_MODES",
    "TRAINING_SCOPES",
    "_preserve_init_backbone",
    "train_hybrid_controller",
]
