from __future__ import annotations

import pytest

from lcfa.semantic_pointer import (
    SEMANTIC_POINTER_FORMAT,
    candidates_for_action,
    candidates_from_cognition,
    hashed_candidate_features,
    make_semantic_pointer,
    pointer_prior_logits,
    retrieval_prior_decision,
    semantic_pointer_decision,
)


def test_pointer_candidates_preserve_action_index_order_and_priors() -> None:
    paths = ("src/a.py", "src/b.py")
    entities = ("function://a", "class://B")
    candidates = candidates_for_action(
        "repo.read",
        candidate_paths=paths,
        candidate_entities=entities,
        candidate_scores=(42.0, 11.0),
        candidate_evidence=(("exact:a",), ("neighbor:B",)),
    )
    assert [item.index for item in candidates] == [0, 1]
    assert [item.path for item in candidates] == list(paths)
    assert [item.entity_id for item in candidates] == list(entities)
    assert candidates[0].retrieval_score == pytest.approx(42.0)
    assert candidates[0].evidence == ("exact:a",)

    queries = candidates_for_action(
        "repo.search",
        candidate_queries=("Widget", "render avatar"),
    )
    assert [item.text for item in queries] == ["Widget", "render avatar"]


def test_pointer_candidates_use_visible_semantic_locations_and_scores() -> None:
    cognition = {
        "candidate_paths": ["src/widget.py", "src/view.py"],
        "candidate_locations": ["function://widget/render", "class://View"],
        "candidate_scores": [9.0, 4.0],
        "candidate_evidence": [["exact:render"], ["lexical:View"]],
    }
    candidates = candidates_from_cognition("repo.edit", cognition)
    assert candidates[0].entity_id == "function://widget/render"
    assert "function://widget/render" in candidates[0].descriptor()
    assert candidates[0].retrieval_score == pytest.approx(9.0)
    assert candidates[1].path == "src/view.py"


def test_hashed_candidate_features_are_deterministic_and_entity_sensitive() -> None:
    torch = pytest.importorskip("torch")
    left = candidates_for_action(
        "repo.read",
        candidate_paths=("src/a.py",),
        candidate_entities=("function://pkg/a",),
    )
    right = candidates_for_action(
        "repo.read",
        candidate_paths=("src/a.py",),
        candidate_entities=("class://pkg/A",),
    )
    left_a = hashed_candidate_features(torch, left, dim=128, device="cpu")
    left_b = hashed_candidate_features(torch, left, dim=128, device="cpu")
    right_f = hashed_candidate_features(torch, right, dim=128, device="cpu")
    assert torch.equal(left_a, left_b)
    assert not torch.equal(left_a, right_f)


def test_retrieval_prior_preserves_ranked_retriever_top1() -> None:
    torch = pytest.importorskip("torch")
    candidates = candidates_for_action(
        "repo.read",
        candidate_paths=("src/top.py", "src/second.py", "src/third.py"),
        candidate_scores=(100.0, 40.0, 5.0),
    )
    logits = pointer_prior_logits(torch, candidates, device="cpu")
    assert int(torch.argmax(logits).item()) == 0
    index, confidence, _ = retrieval_prior_decision(torch, candidates, device="cpu")
    assert index == 0
    assert confidence is not None and 0.0 < confidence <= 1.0


def test_semantic_pointer_starts_exactly_at_retrieval_prior() -> None:
    torch = pytest.importorskip("torch")
    scorer = make_semantic_pointer(
        torch,
        hidden_size=32,
        latent_dim=16,
        device="cpu",
        feature_dim=64,
        pointer_dim=32,
    )
    hidden = torch.randn(1, 32)
    latent = torch.randn(1, 9, 16)
    candidates = candidates_for_action(
        "repo.replace",
        candidate_paths=("src/a.py", "src/b.py", "tests/test_b.py"),
        candidate_entities=("function://a", "function://b", "function://test_b"),
        candidate_scores=(8.0, 6.0, 1.0),
    )
    embeddings = torch.randn(3, 32)
    logits = scorer(hidden, latent, candidates, embeddings)
    prior = pointer_prior_logits(torch, candidates, device="cpu").unsqueeze(0)
    assert SEMANTIC_POINTER_FORMAT == "lcfa.semantic-pointer.v2"
    assert torch.equal(logits, prior)
    assert float(torch.tanh(scorer.residual_gate).item()) == pytest.approx(0.0)


def test_semantic_pointer_residual_backprops_after_gate_opens() -> None:
    torch = pytest.importorskip("torch")
    scorer = make_semantic_pointer(
        torch,
        hidden_size=32,
        latent_dim=16,
        device="cpu",
        feature_dim=64,
        pointer_dim=32,
    )
    with torch.no_grad():
        scorer.residual_gate.fill_(0.2)
    hidden = torch.randn(1, 32, requires_grad=True)
    latent = torch.randn(1, 9, 16, requires_grad=True)
    candidates = candidates_for_action(
        "repo.replace",
        candidate_paths=("src/a.py", "src/b.py", "tests/test_b.py"),
        candidate_entities=("function://a", "function://b", "function://test_b"),
        candidate_scores=(3.0, 2.0, 1.0),
    )
    embeddings = torch.randn(3, 32)
    logits = scorer(hidden, latent, candidates, embeddings)
    assert logits.shape == (1, 3)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
    loss.backward()
    assert hidden.grad is not None
    assert latent.grad is not None
    assert scorer.residual_gate.grad is not None
    assert any(parameter.grad is not None for parameter in scorer.parameters())


def test_semantic_pointer_returns_original_candidate_index() -> None:
    torch = pytest.importorskip("torch")
    scorer = make_semantic_pointer(
        torch,
        hidden_size=16,
        latent_dim=8,
        device="cpu",
        feature_dim=64,
        pointer_dim=16,
    )
    hidden = torch.randn(1, 16)
    latent = torch.randn(1, 9, 8)
    candidates = candidates_for_action(
        "repo.search",
        candidate_queries=("alpha", "beta"),
    )
    embeddings = torch.randn(2, 16)
    index, confidence, logits = semantic_pointer_decision(
        torch,
        scorer,
        hidden,
        latent,
        candidates,
        candidate_embeddings=embeddings,
    )
    # Gate starts at zero, so query-rank prior deterministically picks index 0.
    assert index == 0
    assert confidence is not None and 0.0 <= confidence <= 1.0
    assert logits is not None and logits.shape == (1, 2)
