from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open

from lcfa import (
    HashTextZPlug,
    Observation,
    ZContext,
    ZPlugError,
    ZPlugRegistry,
)
from lcfa.latent_train import encode_examples, examples_from_lines
from lcfa.zplug import _mlx_array_to_float32_numpy


def test_hash_text_zplug_is_deterministic_and_normalized() -> None:
    plug = HashTextZPlug(dim=32)
    observation = Observation("obs:1", "text", "Revenue fell while margin improved.")
    first = plug.encode(observation, ZContext(query="what changed?"))
    second = plug.encode(observation, ZContext(query="what changed?"))
    assert first.features == second.features
    assert first.feature_dim == 32
    norm = sum(value * value for value in first.features) ** 0.5
    assert abs(norm - 1.0) < 1e-5
    assert first.zplug_id == "lcfa.text.hash"


def test_zplug_registry_resolves_by_modality_and_capability() -> None:
    registry = ZPlugRegistry()
    plug = HashTextZPlug(dim=16)
    registry.register(plug)
    assert registry.resolve("text").manifest.id == plug.manifest.id
    packet = registry.encode(Observation("obs", "text", "hello"))
    assert packet.modality == "text"
    try:
        registry.resolve("image")
    except ZPlugError:
        pass
    else:
        raise AssertionError("image should not resolve before a vision zplug is installed")


def test_text_dataset_can_be_masked_and_cached_without_neural_dependencies(tmp_path: Path) -> None:
    examples = examples_from_lines(["alpha beta gamma", "delta epsilon zeta"], mask_ratio=0.34, seed=3)
    assert len(examples) == 2
    assert examples[0].student_text != examples[0].teacher_text
    cache = encode_examples(examples, HashTextZPlug(dim=24), tmp_path / "features.safetensors")
    with safe_open(str(cache), framework="np", device="cpu") as handle:
        assert handle.metadata()["format"] == "lcfa.latent-features.v1"
        assert handle.get_tensor("student").shape == (2, 24)
        assert handle.get_tensor("teacher").shape == (2, 24)


def test_mlx_hidden_conversion_casts_to_float32_before_numpy() -> None:
    sentinel = object()

    class FakeMX:
        float32 = sentinel

        @staticmethod
        def eval(value) -> None:
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.float32

    class FakeArray:
        def astype(self, dtype):
            assert dtype is sentinel
            return np.asarray([1.5, -2.0, 3.25], dtype=np.float32)

    result = _mlx_array_to_float32_numpy(FakeMX, FakeArray())
    assert result.dtype == np.float32
    assert result.tolist() == [1.5, -2.0, 3.25]
