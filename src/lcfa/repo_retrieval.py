"""Shared repository localization for LCFA training and rollout.

The same deterministic retrieval contract is used by historical-oracle
collection, live semantic investigation, and recurrent action argument
resolution.  This prevents the controller from being trained on gold-informed
paths while seeing a different localization mechanism at inference time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from .semantic_graph import ConceptNode, SQLiteSemanticGraph

RETRIEVAL_FORMAT = "lcfa.repo-retrieval.v1"
MAX_RETRIEVAL_QUERIES = 8
MAX_RETRIEVAL_CANDIDATES = 16

_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "was", "were", "with", "when", "after", "before", "from", "this", "that", "it",
    "be", "as", "by", "not", "fix", "make", "support", "enable", "using", "should",
    "does", "do", "can", "could", "issue", "bug", "failure", "fails", "failed",
}
_LOCATION_KINDS = {
    "file", "module", "class", "method", "function", "ast_call", "ast_identifier",
}


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    concept_id: str
    path: str
    kind: str
    label: str
    score: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    goal: str
    queries: tuple[str, ...]
    candidates: tuple[RetrievalCandidate, ...]
    schema_version: str = RETRIEVAL_FORMAT

    @property
    def candidate_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.candidates)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.concept_id for item in self.candidates)

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def extract_retrieval_queries(goal: str, *, limit: int = MAX_RETRIEVAL_QUERIES) -> tuple[str, ...]:
    """Extract high-information literal identifiers before generic words."""
    dotted = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
        goal,
    )
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_-]{2,}", goal)
    scored: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    position = 0
    for token in [*dotted, *identifiers]:
        key = token.casefold()
        if key in seen or key in _STOP:
            continue
        seen.add(key)
        score = 1.0
        score += 4.0 if "." in token else 0.0
        score += 2.0 if "_" in token else 0.0
        score += 1.5 if any(ch.isupper() for ch in token[1:]) else 0.0
        score += min(2.0, len(token) / 24.0)
        scored.append((score, position, token))
        position += 1
    scored.sort(key=lambda item: (-item[0], item[1], item[2].casefold()))
    return tuple(item[2] for item in scored[: max(1, int(limit))])


def _path(node: ConceptNode) -> str | None:
    raw = node.metadata.get("path")
    if raw:
        rendered = str(raw).strip()
        if rendered and not rendered.startswith("/"):
            return rendered
    if node.kind == "file" and node.label and not str(node.label).startswith("/"):
        return str(node.label)
    return None


def _exact_score(node: ConceptNode, query: str) -> float:
    q = query.casefold()
    label = node.label.casefold()
    node_id = node.id.casefold()
    if label == q or node_id.endswith("/" + q) or node_id.endswith(":" + q):
        return 12.0
    if q in label:
        return 7.0
    if q in node_id:
        return 5.0
    return 2.0


def build_retrieval_context(
    graph: SQLiteSemanticGraph,
    goal: str,
    *,
    query_limit: int = MAX_RETRIEVAL_QUERIES,
    candidate_limit: int = MAX_RETRIEVAL_CANDIDATES,
) -> RetrievalContext:
    """Fuse exact identifier lookup, lexical graph search, and AST relations.

    Scores are deterministic and only use the indexed base repository.  No
    historical fix path or gold patch information is accepted by this API.
    """
    queries = extract_retrieval_queries(goal, limit=query_limit)
    if not queries:
        queries = (goal[:120],)

    scores: dict[str, float] = {}
    nodes: dict[str, ConceptNode] = {}
    evidence: dict[str, list[str]] = {}

    def add(node: ConceptNode, score: float, reason: str) -> None:
        path = _path(node)
        if not path or node.kind not in _LOCATION_KINDS:
            return
        nodes[node.id] = node
        scores[node.id] = scores.get(node.id, 0.0) + float(score)
        reasons = evidence.setdefault(node.id, [])
        if reason not in reasons:
            reasons.append(reason)

    for rank, query in enumerate(queries):
        query_weight = max(1.0, 4.0 - rank * 0.35)
        # Exact identifier nodes are indexed by the Python AST visitor.
        exact_ids = (
            f"identifier://python/{query}",
            f"callable://python/{query}",
        )
        for exact_id in exact_ids:
            try:
                exact = graph.get_node(exact_id)
            except KeyError:
                continue
            for edge in graph.neighbors(exact.id, direction="both", limit=64):
                other_id = edge.target if edge.source == exact.id else edge.source
                try:
                    other = graph.get_node(other_id)
                except KeyError:
                    continue
                add(other, 10.0 * query_weight, f"exact:{query}:{edge.relation}")
                # AST occurrence -> owning symbol/file expansion.
                for edge2 in graph.neighbors(other.id, direction="both", limit=16):
                    related_id = edge2.target if edge2.source == other.id else edge2.source
                    try:
                        related = graph.get_node(related_id)
                    except KeyError:
                        continue
                    add(related, 5.0 * query_weight, f"ast:{query}:{edge2.relation}")

        for node in graph.search(query, limit=64):
            add(
                node,
                _exact_score(node, query) * query_weight,
                f"lexical:{query}",
            )
            # A matching symbol can point to its containing module/file or
            # identifiers/calls that provide structural evidence.
            for edge in graph.neighbors(node.id, direction="both", limit=12):
                other_id = edge.target if edge.source == node.id else edge.source
                try:
                    other = graph.get_node(other_id)
                except KeyError:
                    continue
                add(other, 1.5 * query_weight, f"neighbor:{query}:{edge.relation}")

    # Fall back to the old whole-goal lexical graph search only when specific
    # identifiers yielded no path-bearing candidates.
    if not scores:
        for node in graph.search(goal, limit=max(32, candidate_limit * 2)):
            add(node, 1.0, "goal-lexical")

    # Aggregate multiple symbol/AST hits onto one path while preserving the
    # strongest concept as the pointer target.
    by_path: dict[str, RetrievalCandidate] = {}
    for node_id, score in scores.items():
        node = nodes[node_id]
        path = _path(node)
        if not path:
            continue
        candidate = RetrievalCandidate(
            concept_id=node.id,
            path=path,
            kind=node.kind,
            label=node.label,
            score=score,
            evidence=tuple(evidence.get(node.id, ())),
        )
        previous = by_path.get(path)
        if previous is None:
            by_path[path] = candidate
            continue
        merged_evidence = tuple(dict.fromkeys([*previous.evidence, *candidate.evidence]))
        if candidate.score > previous.score:
            by_path[path] = RetrievalCandidate(
                candidate.concept_id,
                candidate.path,
                candidate.kind,
                candidate.label,
                candidate.score + previous.score * 0.25,
                merged_evidence,
            )
        else:
            by_path[path] = RetrievalCandidate(
                previous.concept_id,
                previous.path,
                previous.kind,
                previous.label,
                previous.score + candidate.score * 0.25,
                merged_evidence,
            )

    ranked = sorted(
        by_path.values(),
        key=lambda item: (-item.score, item.path, item.kind, item.label),
    )[: max(1, int(candidate_limit))]
    return RetrievalContext(str(goal), tuple(queries), tuple(ranked))


def retrieval_from_mapping(value: Mapping[str, Any] | None) -> RetrievalContext | None:
    if not isinstance(value, Mapping):
        return None
    goal = str(value.get("goal") or "")
    raw_queries = value.get("queries", ())
    raw_candidates = value.get("candidates", ())
    if not isinstance(raw_queries, Sequence) or isinstance(raw_queries, (str, bytes)):
        return None
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        return None
    candidates: list[RetrievalCandidate] = []
    for raw in raw_candidates:
        if not isinstance(raw, Mapping) or not raw.get("path"):
            continue
        raw_evidence = raw.get("evidence", ())
        evidence = (
            tuple(str(item) for item in raw_evidence)
            if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, (str, bytes))
            else ()
        )
        candidates.append(RetrievalCandidate(
            concept_id=str(raw.get("concept_id") or ""),
            path=str(raw["path"]),
            kind=str(raw.get("kind") or "file"),
            label=str(raw.get("label") or raw["path"]),
            score=float(raw.get("score", 0.0) or 0.0),
            evidence=evidence,
        ))
    return RetrievalContext(
        goal=goal,
        queries=tuple(str(item) for item in raw_queries if str(item)),
        candidates=tuple(candidates),
    )


def pointer_for_action(
    action: str,
    inputs: Mapping[str, Any],
    retrieval: RetrievalContext | None,
) -> int | None:
    if retrieval is None:
        return None
    if action == "repo.search":
        query = str(inputs.get("query") or "")
        try:
            return retrieval.queries.index(query)
        except ValueError:
            return None
    if action in {"repo.read", "repo.replace", "repo.edit"}:
        path = str(inputs.get("path") or "")
        try:
            return retrieval.candidate_paths.index(path)
        except ValueError:
            return None
    return None


__all__ = [
    "MAX_RETRIEVAL_CANDIDATES",
    "MAX_RETRIEVAL_QUERIES",
    "RETRIEVAL_FORMAT",
    "RetrievalCandidate",
    "RetrievalContext",
    "build_retrieval_context",
    "extract_retrieval_queries",
    "pointer_for_action",
    "retrieval_from_mapping",
]
