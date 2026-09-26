from __future__ import annotations

from types import SimpleNamespace

import pytest

from lcfa.model_policy import UnsupportedModelError, reject_qwen_model
from lcfa.torch_runtime import resolve_device, resolve_dtype_name


class _Cuda:
    def __init__(self, available: bool, bf16: bool = False) -> None:
        self._available = available
        self._bf16 = bf16

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16


class _MPS:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _torch(*, cuda: bool, bf16: bool = False, mps: bool = False):
    return SimpleNamespace(
        cuda=_Cuda(cuda, bf16),
        backends=SimpleNamespace(mps=_MPS(mps)),
    )


def test_auto_runtime_prefers_cuda_and_bfloat16_when_supported() -> None:
    torch = _torch(cuda=True, bf16=True, mps=True)
    device = resolve_device(torch, "auto")
    assert device == "cuda"
    assert resolve_dtype_name(torch, device, "auto") == "bfloat16"


def test_auto_runtime_uses_float16_on_apple_mps() -> None:
    torch = _torch(cuda=False, mps=True)
    device = resolve_device(torch, "auto")
    assert device == "mps"
    assert resolve_dtype_name(torch, device, "auto") == "float16"


def test_auto_runtime_uses_float32_on_cpu() -> None:
    torch = _torch(cuda=False, mps=False)
    device = resolve_device(torch, "auto")
    assert device == "cpu"
    assert resolve_dtype_name(torch, device, "auto") == "float32"


def test_qwen_model_family_is_explicitly_unsupported() -> None:
    with pytest.raises(UnsupportedModelError, match="Qwen model support has been removed"):
        reject_qwen_model("mlx-community/Qwen3-4B-Instruct-2507-4bit")
    reject_qwen_model("RWKV/RWKV7-G1j-1.5B-20260831")
