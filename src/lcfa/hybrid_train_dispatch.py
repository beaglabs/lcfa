"""Training dispatcher for hybrid LCFA controllers."""
from __future__ import annotations

from typing import Any, Mapping

from .hybrid_semantic_train import (
    BACKBONE_MODES,
    train_hybrid_controller as train_joint_hybrid_controller,
)
from .pairwise_pointer_train import train_pairwise_pointer_controller

TRAINING_SCOPES = ("joint", "pointer-only")


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
        return train_pairwise_pointer_controller(
            transitions_path,
            output_dir,
            init_controller=init_controller,
            progress=progress,
            **kwargs,
        )
    return train_joint_hybrid_controller(
        transitions_path,
        output_dir,
        training_scope="joint",
        init_controller=init_controller,
        progress=progress,
        **kwargs,
    )


__all__ = ["BACKBONE_MODES", "TRAINING_SCOPES", "train_hybrid_controller"]
