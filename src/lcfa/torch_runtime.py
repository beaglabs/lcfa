"""Shared PyTorch device/dtype resolution for LCFA recurrent runtimes."""
from __future__ import annotations

from typing import Any


DTYPE_NAMES = ("auto", "bfloat16", "float16", "float32")


def resolve_device(torch: Any, requested: str | None = None) -> str:
    value = str(requested or "auto").strip().lower()
    if value not in {"", "auto"}:
        return value
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype_name(torch: Any, device: str, requested: str | None = "auto") -> str:
    value = str(requested or "auto").strip().lower()
    if value != "auto":
        if value not in DTYPE_NAMES[1:]:
            raise ValueError(f"unsupported dtype: {requested}")
        return value
    kind = str(device).split(":", 1)[0].lower()
    if kind == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(supports_bf16) and supports_bf16():
            return "bfloat16"
        return "float16"
    if kind == "mps":
        return "float16"
    return "float32"


def resolve_dtype(torch: Any, device: str, requested: str | None = "auto") -> tuple[str, Any]:
    name = resolve_dtype_name(torch, device, requested)
    return name, {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


__all__ = ["DTYPE_NAMES", "resolve_device", "resolve_dtype", "resolve_dtype_name"]
