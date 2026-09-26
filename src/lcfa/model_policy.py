"""Repository-wide model support policy."""
from __future__ import annotations

from pathlib import Path


class UnsupportedModelError(ValueError):
    pass


def reject_qwen_model(model_ref: str | Path) -> None:
    """Qwen is intentionally unsupported in LCFA first-party model loaders."""
    text = str(model_ref).casefold()
    if "qwen" in text:
        raise UnsupportedModelError(
            "Qwen model support has been removed; use the RWKV recurrent controller path instead"
        )


__all__ = ["UnsupportedModelError", "reject_qwen_model"]
