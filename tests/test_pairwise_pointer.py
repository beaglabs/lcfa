from __future__ import annotations

import pytest

from lcfa.pairwise_pointer import (
    PAIRWISE_POINTER_ARCHITECTURE,
    PAIRWISE_SEMANTIC_POINTER_FORMAT,
    make_pairwise_semantic_pointer,
    pairwise_pointer_prior_logits,
    pairwise_semantic_pointer_decision,
)
from lcfa.semantic_pointer import candidates_for_action


def test_pairwise_pointer_format_and_architecture() -> None:
    assert PAIRWISE_SEMANTIC_POINTER_FORMAT == "lcfa.semantic-pointer.v3"
    assert PAIRWISE_POINTER_ARCHITECTURE == "retrieval-prior+pairwise-rwkv-semantic-residual"


def test_pairwise_pointer_starts_exactly_at_retrieval_prior() -> None:
    torch = pytest.importorskip("torch")
    pointer = make_pairwise_semantic_pointer(
        torch,
        hidden_size=32,
        latent_dim=16,
        device="cpu",
        feature_dim=64,
        pointer_dim=32,
        pair_hidden_dim=32,
    )
    hidden = torch.randn(1, 32)
    latent = torch.randn(1, 9, 16)
    candidates = candidates_for_action(
        "repo.read",
        candidate_paths=("src/top.py", "src/second.py", "src/third.py"),
        candidate_entities=("function://top", "function://second", "function://third"),
        candidate_scores=(9.0, 5.0, 1.0),
    )
    embeddings = torch.randn(3, 32)
    logits, prior, residual = pointer.components(hidden, latent, candidates, embeddings)
    expected_prior = pairwise_pointer_prior_logits(torch, candidates, device="cpu").unsqueeze(0)
    assert torch.equal(prior, expected_prior)
    assert torch.equal(logits, expected_prior)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert float(pointer.prior_strength.item()) == pytest.approx(1.0)


def test_pairwise_query_prior_uses_query_rank_not_repository_scores() -> None:
    torch = pytest.importorskip("torch")
    candidates = candidates_for_action(
        "repo.search",
        candidate_queries=("best query", "second query"),
        candidate_scores=(1.0, 1000.0),
    )
    logits = pairwise_pointer_prior_logits(torch, candidates, device="cpu")
    assert int(torch.argmax(logits).item()) == 0


def test_zero_residual_head_gets_gradient_immediately() -> None:
    torch = pytest.importorskip("torch")
    pointer = make_pairwise_semantic_pointer(
        torch,
        hidden_size=24,
        latent_dim=8,
        device="cpu",
        feature_dim=32,
        pointer_dim=16,
        pair_hidden_dim=16,
    )
    hidden = torch.randn(1, 24)
    latent = torch.randn(1, 9, 8)
    candidates = candidates_for_action(
        "repo.edit",
        candidate_paths=("src/a.py", "src/b.py", "src/c.py"),
        candidate_scores=(10.0, 5.0, 1.0),
    )
    embeddings = torch.randn(3, 24)
    logits = pointer(hidden, latent, candidates, embeddings)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2]))
    loss.backward()
    assert pointer.residual_head.weight.grad is not None
    assert float(pointer.residual_head.weight.grad.abs().sum().item()) > 0.0
    assert pointer.prior_strength.grad is not None


def test_pairwise_pointer_can_override_prior_after_training_steps() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    pointer = make_pairwise_semantic_pointer(
        torch,
        hidden_size=16,
        latent_dim=8,
        device="cpu",
        feature_dim=32,
        pointer_dim=16,
        pair_hidden_dim=16,
    )
    hidden = torch.randn(1, 16)
    latent = torch.randn(1, 9, 8)
    candidates = candidates_for_action(
        "repo.read",
        candidate_paths=("src/prior.py", "src/target.py"),
        candidate_entities=("function://prior", "function://target"),
        candidate_scores=(10.0, 1.0),
    )
    embeddings = torch.randn(2, 16)
    optimizer = torch.optim.AdamW(pointer.parameters(), lr=5e-2)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        logits = pointer(hidden, latent, candidates, embeddings)
        target = torch.tensor([1])
        ce = torch.nn.functional.cross_entropy(logits, target)
        target_score = logits[0, 1]
        wrong_score = logits[0, 0]
        margin = torch.relu(torch.tensor(1.0) - (target_score - wrong_score))
        loss = ce + 0.5 * margin
        loss.backward()
        optimizer.step()
    index, confidence, _ = pairwise_semantic_pointer_decision(
        torch,
        pointer,
        hidden,
        latent,
        candidates,
        candidate_embeddings=embeddings,
    )
    assert index == 1
    assert confidence is not None and confidence > 0.5
