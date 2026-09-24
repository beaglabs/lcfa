from __future__ import annotations

import numpy as np

from lcfa.latent_train import _mlx_batch_indices, _variance_hinge_loss


class _FakeMX:
    int32 = object()

    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def array(self, values, *, dtype=None):
        self.calls.append((values, dtype))
        return {"values": values, "dtype": dtype}


def test_mlx_batch_indices_convert_numpy_indices_to_explicit_mlx_int32_array() -> None:
    mx = _FakeMX()
    host_indices = np.array([7, 2, 5], dtype=np.int64)

    result = _mlx_batch_indices(mx, host_indices)

    assert mx.calls == [([7, 2, 5], mx.int32)]
    assert result == {"values": [7, 2, 5], "dtype": mx.int32}


def test_anti_collapse_variance_is_applied_before_unit_normalization() -> None:
    representation = np.array(
        [
            [-2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
        ],
        dtype=np.float32,
    )
    normalized = representation / np.linalg.norm(representation, axis=-1, keepdims=True)

    pre_norm_penalty = float(_variance_hinge_loss(np, representation))
    post_norm_penalty = float(_variance_hinge_loss(np, normalized))

    assert pre_norm_penalty == 0.0
    assert post_norm_penalty > 0.25
