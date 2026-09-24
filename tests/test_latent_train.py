from __future__ import annotations

import numpy as np

from lcfa.latent_train import _mlx_batch_indices


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
