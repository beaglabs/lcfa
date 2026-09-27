from __future__ import annotations

import pytest

from lcfa.semantic_pointer import (
    candidates_for_action,
    candidates_from_cognition,
    hashed_candidate_features,
    make_semantic_pointer,
    semantic_pointer_decision,
)


def test_pointer_candidates_preserve_action_index_order() -> None:
    paths = ("src/a.py", "src/b.py")
    entities = ("function://a", "class://B")
    candidates = candidates_for_action(
        "repo.read",
        candidate_paths=paths,
        candidate_entities=entities,
    )
    assert [item.index for item in candidates] == [0, 1]
    assert [item.path for item in candidates] == list(paths)
    assert [item.entity_id for item in candidates] == list(entities)

    queries = candidates_for_action(
        "repo.search",
        candidate_queries=("Widget", "render avatar"),
    )
    assert [item.text for item in queries] == ["Widget", "render avatar"]


def test_pointer_candidates_use_visible_semantic_locations() -> None:
    cognition = {
        "candidate_paths": ["src/widget.py", "src/view.py"],
        "candidate_locations": ["function://widget/render", "class://View"],
    }
    candidates = candidates_from_cognition("repo.edit", cognition)
    assert candidates[0].entity_id == "function://widget/render"
    assert "function://widget/render" in candidates[0].descriptor()
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


def test_semantic_pointer_scores_variable_candidate_sets_and_backprops() -> None:
    torch = pytest.importorskip("torch")
    scorer = make_semantic_pointer(
        torch,
        hidden_size=32,
        latent_dim=16,
        device="cpu",
        feature_dim=64,
        pointer_dim=32,
    )
    hidden = torch.randn(1, 32, requires_grad=True)
    latent = torch.randn(1, 9, 16, requires_grad=True)
    candidates = candidates_for_action(
        "repo.replace",
        candidate_paths=("src/a.py", "src/b.py", "tests/test_b.py"),
        candidate_entities=("function://a", "function://b", "function://test_b"),
    )
    logits = scorer(hidden, latent, candidates)
    assert logits.shape == (1, 3)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
    loss.backward()
    assert hidden.grad is not None
    assert latent.grad is not None
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
    index, confidence, logits = semantic_pointer_decision(
        torch, scorer, hidden, latent, candidates
    )
    assert index in {0, 1}
    assert confidence is not None and 0.0 <= confidence <= 1.0
    assert logits is not None and logits.shape == (1, 2)
