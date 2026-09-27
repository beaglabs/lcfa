"""Candidate-conditioned semantic pointing for hybrid LCFA controllers.

The legacy recurrent controller predicts an anonymous slot index.  This module
instead scores the actual repository/query candidates visible at the current
decision point against the fused controller state and the latent user-intent,
target, and repair-intent slots.

Candidate features are deterministic signed-hash lexical features over the
query/path/entity/kind descriptor.  The trainable pointer learns the state query
and candidate projection; semantic entity IDs can therefore be introduced by
newer trajectory formats without changing the scorer shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

SEMANTIC_POINTER_FORMAT = "lcfa.semantic-pointer.v1"
POINTER_ACTIONS = {"repo.search", "repo.read", "repo.replace", "repo.edit"}
DEFAULT_POINTER_FEATURE_DIM = 256
DEFAULT_POINTER_DIM = 256

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


@dataclass(frozen=True, slots=True)
class PointerCandidate:
    index: int
    text: str
    path: str | None = None
    entity_id: str | None = None
    kind: str = "candidate"

    def descriptor(self) -> str:
        parts = [f"kind:{self.kind}", f"text:{self.text}"]
        if self.path:
            parts.append(f"path:{self.path}")
        if self.entity_id:
            parts.append(f"entity:{self.entity_id}")
        return " | ".join(parts)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "path": self.path,
            "entity_id": self.entity_id,
            "kind": self.kind,
        }


def _strings(value: Any, *, limit: int = 32) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value[:limit] if str(item))


def candidates_for_action(
    action: str,
    *,
    candidate_queries: Sequence[str] = (),
    candidate_paths: Sequence[str] = (),
    candidate_entities: Sequence[str] = (),
) -> tuple[PointerCandidate, ...]:
    """Build candidates in exactly the index order used by action arguments."""
    name = str(action)
    if name == "repo.search":
        return tuple(
            PointerCandidate(index=i, text=str(query), kind="query")
            for i, query in enumerate(candidate_queries)
            if str(query)
        )
    if name not in {"repo.read", "repo.replace", "repo.edit"}:
        return ()
    entities = tuple(str(item) for item in candidate_entities)
    return tuple(
        PointerCandidate(
            index=i,
            text=str(path),
            path=str(path),
            entity_id=(entities[i] if i < len(entities) and entities[i] else None),
            kind=(
                entities[i].split("://", 1)[0]
                if i < len(entities) and "://" in entities[i]
                else "repository"
            ),
        )
        for i, path in enumerate(candidate_paths)
        if str(path)
    )


def candidates_from_cognition(action: str, cognition: Mapping[str, Any]) -> tuple[PointerCandidate, ...]:
    return candidates_for_action(
        action,
        candidate_queries=_strings(cognition.get("candidate_queries"), limit=16),
        candidate_paths=_strings(cognition.get("candidate_paths"), limit=16),
        candidate_entities=_strings(cognition.get("candidate_locations"), limit=16),
    )


def candidates_from_transition(row: Any) -> tuple[PointerCandidate, ...]:
    return candidates_for_action(
        str(getattr(row, "target_action", "")),
        candidate_queries=tuple(getattr(row, "candidate_queries", ()) or ()),
        candidate_paths=tuple(getattr(row, "candidate_paths", ()) or ()),
        candidate_entities=tuple(getattr(row, "candidate_entities", ()) or ()),
    )


def _feature_terms(candidate: PointerCandidate) -> tuple[str, ...]:
    descriptor = candidate.descriptor()
    raw = [item.casefold() for item in _TOKEN_RE.findall(descriptor)]
    # Keep whole structured fields in addition to lexical pieces.  This lets the
    # pointer distinguish e.g. identical labels in different paths/entities.
    structured = [
        f"kind={candidate.kind.casefold()}",
        f"text={candidate.text.casefold()}",
    ]
    if candidate.path:
        structured.append(f"path={candidate.path.casefold()}")
    if candidate.entity_id:
        structured.append(f"entity={candidate.entity_id.casefold()}")
    return tuple(dict.fromkeys([*raw, *structured]))


def hashed_candidate_features(
    torch: Any,
    candidates: Sequence[PointerCandidate],
    *,
    dim: int,
    device: str,
) -> Any:
    """Deterministic signed feature hashing; no model/tokenizer side channel."""
    width = max(32, int(dim))
    features = torch.zeros((len(candidates), width), device=device, dtype=torch.float32)
    for row_index, candidate in enumerate(candidates):
        terms = _feature_terms(candidate)
        if not terms:
            continue
        scale = 1.0 / math.sqrt(float(len(terms)))
        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "little") % width
            sign = 1.0 if (digest[8] & 1) == 0 else -1.0
            features[row_index, bucket] += sign * scale
    return features


def make_semantic_pointer(
    torch: Any,
    *,
    hidden_size: int,
    latent_dim: int,
    device: str,
    feature_dim: int = DEFAULT_POINTER_FEATURE_DIM,
    pointer_dim: int = DEFAULT_POINTER_DIM,
) -> Any:
    """Create a variable-length state-to-candidate scorer."""
    nn = torch.nn
    feature_width = max(32, int(feature_dim))
    pointer_width = max(32, int(pointer_dim))
    latent_width = max(1, int(latent_dim))

    class SemanticPointer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_dim = feature_width
            self.pointer_dim = pointer_width
            self.state_query = nn.Sequential(
                nn.Linear(int(hidden_size) + latent_width * 3, pointer_width),
                nn.GELU(),
                nn.LayerNorm(pointer_width),
            )
            self.candidate_projection = nn.Sequential(
                nn.Linear(feature_width, pointer_width, bias=False),
                nn.LayerNorm(pointer_width),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(4.0), dtype=torch.float32))

        @staticmethod
        def _slot(latent: Any, index: int) -> Any:
            if latent is None:
                raise ValueError("semantic pointer requires latent workspace state")
            slots = int(latent.shape[1])
            resolved = index if index < slots else max(0, slots - 1)
            return latent[:, resolved, :].float()

        def forward(
            self,
            hidden: Any,
            latent: Any,
            candidates: Sequence[PointerCandidate],
        ) -> Any:
            if not candidates:
                return torch.empty((int(hidden.shape[0]), 0), device=hidden.device)
            # Slot semantics from hybrid_latent.DEFAULT_SLOT_NAMES:
            # 0=user_intent, 5=target, 6=repair_intent.
            state = torch.cat(
                [
                    hidden.float(),
                    self._slot(latent, 0),
                    self._slot(latent, 5),
                    self._slot(latent, 6),
                ],
                dim=-1,
            )
            query = torch.nn.functional.normalize(self.state_query(state), dim=-1)
            features = hashed_candidate_features(
                torch,
                candidates,
                dim=self.feature_dim,
                device=str(hidden.device),
            )
            keys = torch.nn.functional.normalize(self.candidate_projection(features), dim=-1)
            scale = self.logit_scale.exp().clamp(1.0, 100.0)
            return scale * query @ keys.transpose(0, 1)

    return SemanticPointer().to(device)


def semantic_pointer_decision(
    torch: Any,
    scorer: Any,
    hidden: Any,
    latent: Any,
    candidates: Sequence[PointerCandidate],
) -> tuple[int | None, float | None, Any | None]:
    if scorer is None or latent is None or not candidates:
        return None, None, None
    logits = scorer(hidden, latent, candidates)
    if logits.numel() == 0:
        return None, None, logits
    probs = torch.softmax(logits.float(), dim=-1)[0]
    local_index = int(torch.argmax(probs).item())
    return candidates[local_index].index, float(probs[local_index].item()), logits


__all__ = [
    "DEFAULT_POINTER_DIM",
    "DEFAULT_POINTER_FEATURE_DIM",
    "POINTER_ACTIONS",
    "PointerCandidate",
    "SEMANTIC_POINTER_FORMAT",
    "candidates_for_action",
    "candidates_from_cognition",
    "candidates_from_transition",
    "hashed_candidate_features",
    "make_semantic_pointer",
    "semantic_pointer_decision",
]
