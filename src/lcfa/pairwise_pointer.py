"""Pairwise baseline-preserving semantic reranking for LCFA pointer decisions.

v3 keeps deterministic retrieval as a learnable-strength prior and adds a
candidate-specific residual produced from pairwise query/candidate features.
The final residual projection is initialized to exactly zero, so a new v3
pointer reproduces the retrieval ranking while every upstream reranker layer
receives useful gradients from the first optimizer step.
"""
from __future__ import annotations

from typing import Any, Sequence

from .semantic_pointer import (
    PointerCandidate,
    hashed_candidate_features,
    pointer_prior_logits,
)

PAIRWISE_SEMANTIC_POINTER_FORMAT = "lcfa.semantic-pointer.v3"
PAIRWISE_POINTER_ARCHITECTURE = "retrieval-prior+pairwise-rwkv-semantic-residual"


def pairwise_pointer_prior_logits(
    torch: Any,
    candidates: Sequence[PointerCandidate],
    *,
    device: str,
) -> Any:
    """Return the deterministic prior used by v3.

    Repository candidates use retrieval scores when available. Search-query
    candidates use their existing query order because repository candidate
    scores are not query scores.
    """
    if not candidates:
        return torch.empty((0,), device=device, dtype=torch.float32)
    if all(candidate.kind == "query" for candidate in candidates):
        return -torch.arange(len(candidates), device=device, dtype=torch.float32)
    return pointer_prior_logits(torch, candidates, device=device)


def _rank_features(torch: Any, count: int, *, device: str) -> Any:
    if count <= 0:
        return torch.empty((0, 1), device=device, dtype=torch.float32)
    denominator = float(max(1, count - 1))
    return (
        torch.arange(count, device=device, dtype=torch.float32) / denominator
    ).unsqueeze(-1)


def make_pairwise_semantic_pointer(
    torch: Any,
    *,
    hidden_size: int,
    latent_dim: int,
    device: str,
    feature_dim: int = 256,
    pointer_dim: int = 256,
    pair_hidden_dim: int = 256,
) -> Any:
    """Create a v3 query/candidate pairwise residual ranker.

    Invariant: immediately after construction, output logits equal the
    deterministic retrieval prior exactly.
    """
    nn = torch.nn
    feature_width = max(32, int(feature_dim))
    pointer_width = max(32, int(pointer_dim))
    pair_width = max(32, int(pair_hidden_dim))
    latent_width = max(1, int(latent_dim))
    candidate_hidden_width = max(1, int(hidden_size))

    class PairwiseSemanticPointer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_dim = feature_width
            self.pointer_dim = pointer_width
            self.pair_hidden_dim = pair_width
            self.state_query = nn.Sequential(
                nn.Linear(int(hidden_size) + latent_width * 3, pointer_width),
                nn.GELU(),
                nn.LayerNorm(pointer_width),
            )
            self.candidate_projection = nn.Sequential(
                nn.Linear(candidate_hidden_width + feature_width, pointer_width),
                nn.GELU(),
                nn.LayerNorm(pointer_width),
            )
            pair_input_width = pointer_width * 4 + 2
            self.pair_mlp = nn.Sequential(
                nn.Linear(pair_input_width, pair_width),
                nn.GELU(),
                nn.LayerNorm(pair_width),
                nn.Linear(pair_width, pair_width),
                nn.GELU(),
                nn.LayerNorm(pair_width),
            )
            self.residual_head = nn.Linear(pair_width, 1)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
            self.prior_strength = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        @staticmethod
        def _slot(latent: Any, index: int) -> Any:
            if latent is None:
                raise ValueError("pairwise semantic pointer requires latent workspace state")
            slots = int(latent.shape[1])
            resolved = index if index < slots else max(0, slots - 1)
            return latent[:, resolved, :].float()

        def components(
            self,
            hidden: Any,
            latent: Any,
            candidates: Sequence[PointerCandidate],
            candidate_embeddings: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            batch = int(hidden.shape[0])
            if not candidates:
                empty = torch.empty((batch, 0), device=hidden.device, dtype=torch.float32)
                return empty, empty, empty
            state = torch.cat(
                [
                    hidden.float(),
                    self._slot(latent, 0),
                    self._slot(latent, 5),
                    self._slot(latent, 6),
                ],
                dim=-1,
            )
            query = self.state_query(state)
            if candidate_embeddings is None:
                candidate_embeddings = torch.zeros(
                    (len(candidates), candidate_hidden_width),
                    device=hidden.device,
                    dtype=torch.float32,
                )
            semantic = candidate_embeddings.to(device=hidden.device, dtype=torch.float32)
            expected_shape = (len(candidates), candidate_hidden_width)
            if tuple(semantic.shape) != expected_shape:
                raise ValueError(
                    "candidate embedding shape mismatch: "
                    f"expected {expected_shape} got {tuple(semantic.shape)}"
                )
            lexical = hashed_candidate_features(
                torch,
                candidates,
                dim=self.feature_dim,
                device=str(hidden.device),
            )
            candidate_state = self.candidate_projection(torch.cat([semantic, lexical], dim=-1))
            query_expanded = query.unsqueeze(1).expand(-1, len(candidates), -1)
            key_expanded = candidate_state.unsqueeze(0).expand(batch, -1, -1)
            prior = pairwise_pointer_prior_logits(
                torch,
                candidates,
                device=str(hidden.device),
            ).unsqueeze(0).expand(batch, -1)
            ranks = _rank_features(
                torch,
                len(candidates),
                device=str(hidden.device),
            ).unsqueeze(0).expand(batch, -1, -1)
            pair = torch.cat(
                [
                    query_expanded,
                    key_expanded,
                    query_expanded * key_expanded,
                    torch.abs(query_expanded - key_expanded),
                    prior.unsqueeze(-1),
                    ranks,
                ],
                dim=-1,
            )
            pair_hidden = self.pair_mlp(pair)
            residual = self.residual_head(pair_hidden).squeeze(-1)
            # Keep the prior positive and bounded while allowing the learner to
            # reduce its influence when semantic evidence repeatedly disagrees.
            prior_strength = self.prior_strength.clamp(0.0, 2.0)
            final = prior_strength * prior + residual
            return final, prior, residual

        def forward(
            self,
            hidden: Any,
            latent: Any,
            candidates: Sequence[PointerCandidate],
            candidate_embeddings: Any | None = None,
        ) -> Any:
            final, _prior, _residual = self.components(
                hidden,
                latent,
                candidates,
                candidate_embeddings,
            )
            return final

    return PairwiseSemanticPointer().to(device)


def pairwise_semantic_pointer_decision(
    torch: Any,
    scorer: Any,
    hidden: Any,
    latent: Any,
    candidates: Sequence[PointerCandidate],
    *,
    candidate_embeddings: Any | None = None,
) -> tuple[int | None, float | None, Any | None]:
    if scorer is None or latent is None or not candidates:
        return None, None, None
    logits = scorer(hidden, latent, candidates, candidate_embeddings)
    if logits.numel() == 0:
        return None, None, logits
    probs = torch.softmax(logits.float(), dim=-1)[0]
    local_index = int(torch.argmax(probs).item())
    return candidates[local_index].index, float(probs[local_index].item()), logits


__all__ = [
    "PAIRWISE_POINTER_ARCHITECTURE",
    "PAIRWISE_SEMANTIC_POINTER_FORMAT",
    "make_pairwise_semantic_pointer",
    "pairwise_pointer_prior_logits",
    "pairwise_semantic_pointer_decision",
]
