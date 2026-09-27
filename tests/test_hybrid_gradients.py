from __future__ import annotations

import pytest

from lcfa.hybrid_latent import HybridLatentConfig, make_hybrid_core


def test_anchored_intent_core_backpropagates_across_recurrent_steps() -> None:
    torch = pytest.importorskip("torch")
    core = make_hybrid_core(
        torch,
        hidden_size=24,
        config=HybridLatentConfig(
            latent_dim=16,
            slots=9,
            min_reasoning_steps=2,
            max_reasoning_steps=2,
            convergence_tolerance=0.0,
            intent_anchor_strength=0.98,
        ),
        device="cpu",
    )
    latent = None
    losses = []
    for _ in range(3):
        fused, latent, plan, _depth = core(torch.randn(1, 24), latent)
        losses.append(fused.square().mean() + plan.square().mean())
    torch.stack(losses).mean().backward()
    assert core.intent_projection.weight.grad is not None
    assert core.evidence_projection.weight.grad is not None
    assert core.update.weight_hh.grad is not None
