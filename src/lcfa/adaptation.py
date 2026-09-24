"""Session-scoped fast parametric priors for stochastic-flow.

These priors are intentionally tiny and model-agnostic. They update candidate
ranking during inference without mutating the frozen semantic backbone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
import math

import numpy as np
from safetensors.numpy import save_file

from .artifact import ArtifactError


FEATURE_NAMES = (
    "logprob",
    "confidence",
    "evidence",
    "tool",
    "parse",
    "final",
    "verifier",
)


class FastPrior(Protocol):
    metadata: Mapping[str, Any]

    def score(self, features: Sequence[float]) -> float: ...
    def observe(self, features: Sequence[float], target: float) -> Mapping[str, float]: ...
    def reset(self) -> None: ...
    def snapshot(self, path: str | Path) -> Path: ...


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class NullFastPrior:
    metadata = {"type": "none", "mutable": False}

    def score(self, features: Sequence[float]) -> float:
        del features
        return 0.0

    def observe(self, features: Sequence[float], target: float) -> Mapping[str, float]:
        del features, target
        return {"prediction": 0.5, "target": 0.5, "surprise": 0.0}

    def reset(self) -> None:
        return None

    def snapshot(self, path: str | Path) -> Path:
        raise ArtifactError("cannot snapshot adaptation.type=none")


@dataclass
class NumpyFastPrior:
    feature_count: int
    learning_rate: float = 0.08
    decay: float = 0.999
    clip: float = 4.0

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.feature_count, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.initial_weights = self.weights.copy()
        self.initial_bias = np.float32(self.bias)
        self.updates = 0
        self.metadata = {
            "type": "numpy-fast",
            "mutable": True,
            "feature_count": self.feature_count,
            "learning_rate": self.learning_rate,
            "decay": self.decay,
        }

    def _vector(self, features: Sequence[float]) -> np.ndarray:
        vector = np.asarray(tuple(float(v) for v in features), dtype=np.float32)
        if vector.size != self.feature_count:
            raise ValueError(f"fast prior expected {self.feature_count} features, got {vector.size}")
        return vector

    def score(self, features: Sequence[float]) -> float:
        x = self._vector(features)
        return float(np.dot(self.weights, x) + self.bias)

    def observe(self, features: Sequence[float], target: float) -> Mapping[str, float]:
        x = self._vector(features)
        target = max(0.0, min(1.0, float(target)))
        raw = float(np.dot(self.weights, x) + self.bias)
        prediction = _sigmoid(raw)
        error = target - prediction
        self.weights *= np.float32(self.decay)
        self.weights += np.float32(self.learning_rate * error) * x
        self.bias = np.float32(float(self.bias) + self.learning_rate * error)
        self.weights = np.clip(self.weights, -self.clip, self.clip).astype(np.float32)
        self.bias = np.float32(max(-self.clip, min(self.clip, float(self.bias))))
        self.updates += 1
        return {"prediction": prediction, "target": target, "surprise": abs(error)}

    def reset(self) -> None:
        self.weights = self.initial_weights.copy()
        self.bias = np.float32(self.initial_bias)
        self.updates = 0

    def snapshot(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "prior.weights": self.weights.astype(np.float32),
                "prior.bias": np.asarray([self.bias], dtype=np.float32),
            },
            str(output),
            metadata={"format": "lcfa.fast-prior.v1", "backend": "numpy-fast"},
        )
        return output


class MLXFastPrior:
    """Tiny Apple-Silicon-native online prior updated during inference."""

    def __init__(self, feature_count: int, *, learning_rate: float = 0.08,
                 decay: float = 0.999, clip: float = 4.0) -> None:
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise ArtifactError("MLX fast prior requires `pip install -e '.[mlx]'`") from exc
        self.mx = mx
        self.feature_count = int(feature_count)
        self.learning_rate = float(learning_rate)
        self.decay = float(decay)
        self.clip = float(clip)
        self.weights = mx.zeros((self.feature_count,), dtype=mx.float32)
        self.bias = mx.array(0.0, dtype=mx.float32)
        self.updates = 0
        self.metadata = {
            "type": "mlx-fast",
            "mutable": True,
            "feature_count": self.feature_count,
            "learning_rate": self.learning_rate,
            "decay": self.decay,
        }

    def _vector(self, features: Sequence[float]):
        if len(tuple(features)) != self.feature_count:
            raise ValueError(f"fast prior expected {self.feature_count} features")
        return self.mx.array(tuple(float(v) for v in features), dtype=self.mx.float32)

    def score(self, features: Sequence[float]) -> float:
        x = self._vector(features)
        return float((self.mx.sum(self.weights * x) + self.bias).item())

    def observe(self, features: Sequence[float], target: float) -> Mapping[str, float]:
        mx = self.mx
        x = self._vector(features)
        target = max(0.0, min(1.0, float(target)))
        raw = mx.sum(self.weights * x) + self.bias
        prediction_array = 1.0 / (1.0 + mx.exp(-mx.clip(raw, -30.0, 30.0)))
        prediction = float(prediction_array.item())
        error = target - prediction
        self.weights = mx.clip(
            self.weights * self.decay + self.learning_rate * error * x,
            -self.clip,
            self.clip,
        )
        self.bias = mx.clip(self.bias + self.learning_rate * error, -self.clip, self.clip)
        mx.eval(self.weights, self.bias)
        self.updates += 1
        return {"prediction": prediction, "target": target, "surprise": abs(error)}

    def reset(self) -> None:
        self.weights = self.mx.zeros((self.feature_count,), dtype=self.mx.float32)
        self.bias = self.mx.array(0.0, dtype=self.mx.float32)
        self.mx.eval(self.weights, self.bias)
        self.updates = 0

    def snapshot(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray([float(self.bias.item())], dtype=np.float32)
        save_file(
            {"prior.weights": weights, "prior.bias": bias},
            str(output),
            metadata={"format": "lcfa.fast-prior.v1", "backend": "mlx-fast"},
        )
        return output


def create_fast_prior(kind: str, *, feature_count: int, config: Mapping[str, Any],
                      runtime: Mapping[str, Any]) -> FastPrior:
    selected = str(runtime.get("adaptation_type") or kind or "none")
    learning_rate = float(runtime.get("adaptation_learning_rate", config.get("learning_rate", 0.08)))
    decay = float(runtime.get("adaptation_decay", config.get("decay", 0.999)))
    clip = float(config.get("clip", 4.0))
    if selected == "none":
        return NullFastPrior()
    if selected == "numpy-fast":
        return NumpyFastPrior(feature_count, learning_rate=learning_rate, decay=decay, clip=clip)
    if selected == "mlx-fast":
        return MLXFastPrior(feature_count, learning_rate=learning_rate, decay=decay, clip=clip)
    raise ArtifactError(f"unsupported adaptation type: {selected!r}")


__all__ = [
    "FEATURE_NAMES", "FastPrior", "NullFastPrior", "NumpyFastPrior", "MLXFastPrior", "create_fast_prior",
]
