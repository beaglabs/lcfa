"""Baseline-preserving semantic pointing for hybrid LCFA controllers.

The pointer is a learned reranker, not a replacement for deterministic repository
retrieval.  Every decision starts from the retriever's ordered score prior and
adds a gated semantic residual conditioned on the fused RWKV controller state,
latent user intent, target, and repair-intent slots.

Candidate semantics come from the already-loaded frozen RWKV backbone.  A
zero-initialized residual gate guarantees that a freshly initialized v2 pointer
has exactly the deterministic retriever's ranking before it learns an override.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, MutableMapping, Sequence

LEGACY_SEMANTIC_POINTER_FORMAT = "lcfa.semantic-pointer.v1"
SEMANTIC_POINTER_FORMAT = "lcfa.semantic-pointer.v2"
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
    retrieval_score: float | None = None
    evidence: tuple[str, ...] = ()

    def descriptor(self) -> str:
        parts = [f"kind:{self.kind}", f"text:{self.text}"]
        if self.path:
            parts.append(f"path:{self.path}")
        if self.entity_id:
            parts.append(f"entity:{self.entity_id}")
        if self.evidence:
            parts.append("evidence:" + " ; ".join(self.evidence[:8]))
        return " | ".join(parts)

    def semantic_text(self) -> str:
        lines = [
            "LCFA repository candidate",
            f"kind: {self.kind}",
            f"text: {self.text}",
        ]
        if self.path:
            lines.append(f"path: {self.path}")
        if self.entity_id:
            lines.append(f"semantic entity: {self.entity_id}")
        if self.evidence:
            lines.append("retrieval evidence: " + " ; ".join(self.evidence[:8]))
        return "\n".join(lines)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "path": self.path,
            "entity_id": self.entity_id,
            "kind": self.kind,
            "retrieval_score": self.retrieval_score,
            "evidence": list(self.evidence),
        }


def _strings(value: Any, *, limit: int = 32) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value[:limit] if str(item))


def _floats(value: Any, *, limit: int = 32) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    out: list[float] = []
    for item in value[:limit]:
        try:
            rendered = float(item)
        except (TypeError, ValueError):
            rendered = 0.0
        out.append(rendered)
    return tuple(out)


def _evidence_rows(value: Any, *, limit: int = 32) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    rows: list[tuple[str, ...]] = []
    for raw in value[:limit]:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            rows.append(tuple(str(item) for item in raw[:8] if str(item)))
        elif raw:
            rows.append((str(raw),))
        else:
            rows.append(())
    return tuple(rows)


def candidates_for_action(
    action: str,
    *,
    candidate_queries: Sequence[str] = (),
    candidate_paths: Sequence[str] = (),
    candidate_entities: Sequence[str] = (),
    candidate_scores: Sequence[float] = (),
    candidate_evidence: Sequence[Sequence[str]] = (),
) -> tuple[PointerCandidate, ...]:
    """Build candidates in exactly the index order used by action arguments."""
    name = str(action)
    scores = tuple(float(item) for item in candidate_scores)
    evidence = tuple(tuple(str(part) for part in row) for row in candidate_evidence)
    if name == "repo.search":
        return tuple(
            PointerCandidate(
                index=i,
                text=str(query),
                kind="query",
                retrieval_score=(scores[i] if i < len(scores) else None),
                evidence=(evidence[i] if i < len(evidence) else (f"query-rank:{i}",)),
            )
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
            retrieval_score=(scores[i] if i < len(scores) else None),
            evidence=(evidence[i] if i < len(evidence) else ()),
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
        candidate_scores=_floats(cognition.get("candidate_scores"), limit=16),
        candidate_evidence=_evidence_rows(cognition.get("candidate_evidence"), limit=16),
    )


def candidates_from_transition(row: Any) -> tuple[PointerCandidate, ...]:
    return candidates_for_action(
        str(getattr(row, "target_action", "")),
        candidate_queries=tuple(getattr(row, "candidate_queries", ()) or ()),
        candidate_paths=tuple(getattr(row, "candidate_paths", ()) or ()),
        candidate_entities=tuple(getattr(row, "candidate_entities", ()) or ()),
        candidate_scores=tuple(getattr(row, "candidate_scores", ()) or ()),
        candidate_evidence=tuple(getattr(row, "candidate_evidence", ()) or ()),
    )


def _feature_terms(candidate: PointerCandidate) -> tuple[str, ...]:
    descriptor = candidate.descriptor()
    raw = [item.casefold() for item in _TOKEN_RE.findall(descriptor)]
    structured = [
        f"kind={candidate.kind.casefold()}",
        f"text={candidate.text.casefold()}",
    ]
    if candidate.path:
        structured.append(f"path={candidate.path.casefold()}")
    if candidate.entity_id:
        structured.append(f"entity={candidate.entity_id.casefold()}")
    structured.extend(f"evidence={item.casefold()}" for item in candidate.evidence[:8])
    return tuple(dict.fromkeys([*raw, *structured]))


def hashed_candidate_features(
    torch: Any,
    candidates: Sequence[PointerCandidate],
    *,
    dim: int,
    device: str,
) -> Any:
    """Stable metadata features supplementing pretrained RWKV semantics."""
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


def pointer_prior_logits(
    torch: Any,
    candidates: Sequence[PointerCandidate],
    *,
    device: str,
) -> Any:
    """Monotonic normalized retrieval prior; top-ranked retrieval stays top-1."""
    if not candidates:
        return torch.empty((0,), device=device, dtype=torch.float32)
    explicit = [candidate.retrieval_score for candidate in candidates]
    if any(value is not None for value in explicit):
        present = [float(value) for value in explicit if value is not None]
        floor = min(present) if present else 0.0
        values = [
            float(value) if value is not None else floor - float(index + 1)
            for index, value in enumerate(explicit)
        ]
        scores = torch.tensor(values, device=device, dtype=torch.float32)
        mean = scores.mean()
        std = scores.std(unbiased=False)
        if float(std.item()) > 1e-6:
            return (scores - mean) / std
        return -torch.arange(len(candidates), device=device, dtype=torch.float32)
    # Retrieval/query candidates are already in deterministic rank order.
    return -torch.arange(len(candidates), device=device, dtype=torch.float32)


def retrieval_prior_decision(
    torch: Any,
    candidates: Sequence[PointerCandidate],
    *,
    device: str,
) -> tuple[int | None, float | None, Any]:
    logits = pointer_prior_logits(torch, candidates, device=device)
    if logits.numel() == 0:
        return None, None, logits
    probs = torch.softmax(logits, dim=-1)
    local_index = int(torch.argmax(probs).item())
    return candidates[local_index].index, float(probs[local_index].item()), logits


def encode_candidate_semantics(
    torch: Any,
    model: Any,
    tokenizer: Any,
    candidates: Sequence[PointerCandidate],
    *,
    device: str,
    cache: MutableMapping[str, Any] | None = None,
) -> Any:
    """Encode candidates with the frozen pretrained RWKV representation.

    Encoding is deliberately detached from the trajectory recurrent state.  It
    is repository perception, not an additional temporal update.
    """
    if not candidates:
        hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
        return torch.empty((0, hidden_size), device=device, dtype=torch.float32)
    vectors: list[Any] = []
    for candidate in candidates:
        key = candidate.descriptor()
        cached = cache.get(key) if cache is not None else None
        if cached is not None:
            vectors.append(cached.to(device=device, dtype=torch.float32))
            continue
        encoded = tokenizer(
            candidate.semantic_text(),
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("RWKV candidate encoder requires hidden_states")
            # Mean pooling is less brittle than relying on a synthetic EOS token.
            vector = hidden_states[-1][0].float().mean(dim=0).detach().clone()
        if cache is not None:
            cache[key] = vector.detach().clone()
        vectors.append(vector)
    return torch.stack(vectors, dim=0)


def make_legacy_semantic_pointer(
    torch: Any,
    *,
    hidden_size: int,
    latent_dim: int,
    device: str,
    feature_dim: int = DEFAULT_POINTER_FEATURE_DIM,
    pointer_dim: int = DEFAULT_POINTER_DIM,
) -> Any:
    """v1 hash-only scorer retained solely for loading old artifacts."""
    nn = torch.nn
    feature_width = max(32, int(feature_dim))
    pointer_width = max(32, int(pointer_dim))
    latent_width = max(1, int(latent_dim))

    class LegacySemanticPointer(nn.Module):
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
            slots = int(latent.shape[1])
            resolved = index if index < slots else max(0, slots - 1)
            return latent[:, resolved, :].float()

        def forward(self, hidden: Any, latent: Any, candidates: Sequence[PointerCandidate]) -> Any:
            if not candidates:
                return torch.empty((int(hidden.shape[0]), 0), device=hidden.device)
            state = torch.cat(
                [hidden.float(), self._slot(latent, 0), self._slot(latent, 5), self._slot(latent, 6)],
                dim=-1,
            )
            query = torch.nn.functional.normalize(self.state_query(state), dim=-1)
            features = hashed_candidate_features(
                torch, candidates, dim=self.feature_dim, device=str(hidden.device)
            )
            keys = torch.nn.functional.normalize(self.candidate_projection(features), dim=-1)
            scale = self.logit_scale.exp().clamp(1.0, 100.0)
            return scale * query @ keys.transpose(0, 1)

    return LegacySemanticPointer().to(device)


def make_semantic_pointer(
    torch: Any,
    *,
    hidden_size: int,
    latent_dim: int,
    device: str,
    feature_dim: int = DEFAULT_POINTER_FEATURE_DIM,
    pointer_dim: int = DEFAULT_POINTER_DIM,
) -> Any:
    """Create the v2 retrieval-prior + pretrained-semantic residual reranker."""
    nn = torch.nn
    feature_width = max(32, int(feature_dim))
    pointer_width = max(32, int(pointer_dim))
    latent_width = max(1, int(latent_dim))
    candidate_hidden_width = max(1, int(hidden_size))

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
                nn.Linear(candidate_hidden_width + feature_width, pointer_width, bias=False),
                nn.GELU(),
                nn.LayerNorm(pointer_width),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(4.0), dtype=torch.float32))
            # Critical invariant: new rerankers exactly equal deterministic retrieval.
            self.residual_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

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
            candidate_embeddings: Any | None = None,
        ) -> Any:
            if not candidates:
                return torch.empty((int(hidden.shape[0]), 0), device=hidden.device)
            prior = pointer_prior_logits(torch, candidates, device=str(hidden.device)).unsqueeze(0)
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
            if candidate_embeddings is None:
                candidate_embeddings = torch.zeros(
                    (len(candidates), candidate_hidden_width),
                    device=hidden.device,
                    dtype=torch.float32,
                )
            semantic = candidate_embeddings.to(device=hidden.device, dtype=torch.float32)
            if semantic.shape != (len(candidates), candidate_hidden_width):
                raise ValueError(
                    "candidate embedding shape mismatch: "
                    f"expected {(len(candidates), candidate_hidden_width)} got {tuple(semantic.shape)}"
                )
            lexical = hashed_candidate_features(
                torch, candidates, dim=self.feature_dim, device=str(hidden.device)
            )
            candidate_state = torch.cat([semantic, lexical], dim=-1)
            keys = torch.nn.functional.normalize(self.candidate_projection(candidate_state), dim=-1)
            residual_scale = self.logit_scale.exp().clamp(1.0, 100.0)
            residual = residual_scale * query @ keys.transpose(0, 1)
            gate = torch.tanh(self.residual_gate)
            return prior + gate * residual

    return SemanticPointer().to(device)


def semantic_pointer_decision(
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
    try:
        logits = scorer(hidden, latent, candidates, candidate_embeddings)
    except TypeError:
        # Old v1 scorer signature.
        logits = scorer(hidden, latent, candidates)
    if logits.numel() == 0:
        return None, None, logits
    probs = torch.softmax(logits.float(), dim=-1)[0]
    local_index = int(torch.argmax(probs).item())
    return candidates[local_index].index, float(probs[local_index].item()), logits


__all__ = [
    "DEFAULT_POINTER_DIM",
    "DEFAULT_POINTER_FEATURE_DIM",
    "LEGACY_SEMANTIC_POINTER_FORMAT",
    "POINTER_ACTIONS",
    "PointerCandidate",
    "SEMANTIC_POINTER_FORMAT",
    "candidates_for_action",
    "candidates_from_cognition",
    "candidates_from_transition",
    "encode_candidate_semantics",
    "hashed_candidate_features",
    "make_legacy_semantic_pointer",
    "make_semantic_pointer",
    "pointer_prior_logits",
    "retrieval_prior_decision",
    "semantic_pointer_decision",
]
