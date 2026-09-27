from __future__ import annotations

import pytest

from lcfa.hybrid_latent import HybridLatentConfig, make_hybrid_core


def test_new_hybrid_core_preserves_rwkv_control_features_at_initialization() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    core = make_hybrid_core(
        torch,
        hidden_size=32,
        config=HybridLatentConfig(
            latent_dim=16,
            slots=9,
            min_reasoning_steps=2,
            max_reasoning_steps=2,
            convergence_tolerance=0.0,
        ),
        device="cpu",
    )
    source = torch.randn(2, 32)
    fused, latent, plan_logits, depth = core(source)

    assert torch.equal(fused, source.float())
    assert latent.shape == (2, 9, 16)
    assert plan_logits.shape == (2, 4)
    assert depth == 2
