"""Shared repository localization for LCFA training and rollout.

The same deterministic retrieval contract is used by historical-oracle
collection, live semantic investigation, and recurrent action argument
resolution. This prevents the controller from being trained on gold-informed
paths while seeing a different localization mechanism at inference time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
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


def _lexical_terms(text: str) -> set[str]:
    return {
        item.casefold()
        for item in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text.replace("-", "_"))
        if item.casefold() not in _STOP
    }


def _path_boost(path: str, queries: Sequence[str], goal: str) -> tuple[float, tuple[str, ...]]:
    """Score filename/path evidence independently of symbol-frequency evidence."""
    rendered = str(PurePosixPath(path)).casefold()
    stem = PurePosixPath(path).stem.casefold()
    path_terms = _lexical_terms(rendered)
    goal_terms = _lexical_terms(goal)
    score = 0.0
    evidence: list[str] = []

    overlap = path_terms & goal_terms
    if overlap:
        score += 3.0 * len(overlap)
        evidence.append("path-goal:" + ",".join(sorted(overlap)))

    for rank, query in enumerate(queries):
        q = query.casefold()
        query_terms = _lexical_terms(query)
        rank_weight = max(1.0, 5.0 - rank * 0.5)
        if q and q in rendered:
            score += 4.0 * rank_weight
            evidence.append(f"path-exact:{query}")
            continue
        token_overlap = query_terms & path_terms
        if token_overlap:
            score += 1.5 * rank_weight * len(token_overlap)
            evidence.append(
                f"path-token:{query}:" + ",".join(sorted(token_overlap))
            )
        if q and q.replace("-", "_") == stem:
            score += 6.0 * rank_weight
            evidence.append(f"stem-exact:{query}")

    return score, tuple(evidence)


def build_retrieval_context(
    graph: SQLiteSemanticGraph,
    goal: str,
    *,
    query_limit: int = MAX_RETRIEVAL_QUERIES,
    candidate_limit: int = MAX_RETRIEVAL_CANDIDATES,
) -> RetrievalContext:
    """Fuse direct paths, exact identifiers, lexical search and AST relations.

    Scores are deterministic and only use the indexed base repository. No
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

        # Give issue-named files/modules their own retrieval lane before
        # high-frequency identifier matches can dominate the candidate pool.
        for node in graph.search(query, kinds=("file", "module"), limit=64):
            path = _path(node)
            if not path:
                continue
            q = query.casefold()
            normalized_path = path.casefold().replace("-", "_")
            normalized_query = q.replace("-", "_")
            path_score = 8.0 * query_weight
            if normalized_query in normalized_path:
                path_score = 18.0 * query_weight
            add(node, path_score, f"direct-path:{query}")

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
            for edge in graph.neighbors(node.id, direction="both", limit=12):
                other_id = edge.target if edge.source == node.id else edge.source
                try:
                    other = graph.get_node(other_id)
                except KeyError:
                    continue
                add(other, 1.5 * query_weight, f"neighbor:{query}:{edge.relation}")

    if not scores:
        for node in graph.search(goal, limit=max(32, candidate_limit * 2)):
            add(node, 1.0, "goal-lexical")

    by_path: dict[str, RetrievalCandidate] = {}
    for node_id, score in scores.items():
        node = nodes[node_id]
        path = _path(node)
        if not path:
            continue
        boost, boost_evidence = _path_boost(path, queries, goal)
        candidate = RetrievalCandidate(
            concept_id=node.id,
            path=path,
            kind=node.kind,
            label=node.label,
            score=score + boost,
            evidence=tuple(dict.fromkeys([*evidence.get(node.id, ()), *boost_evidence])),
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
