"""Cognitive working-state construction over the LCFA semantic graph."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping, Sequence

from .protocol import EvidenceRef, Finding, Recommendation, SolutionState
from .semantic_graph import ConceptEdge, ConceptNode, SQLiteSemanticGraph
from .state import content_hash

COGNITIVE_STATE_FORMAT = "lcfa.cognition.v1"


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    claim: str
    supports: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    confidence: float = 0.0
    status: str = "open"


@dataclass(frozen=True, slots=True)
class CognitiveState:
    goal: str
    active_concepts: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...]
    open_questions: tuple[str, ...] = ()
    candidate_locations: tuple[str, ...] = ()
    observations: tuple[Mapping[str, Any], ...] = ()
    next_actions: tuple[Mapping[str, Any], ...] = ()
    terminal: bool = False
    schema_version: str = COGNITIVE_STATE_FORMAT

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "were",
        "with", "when", "after", "before", "from", "this", "that", "it", "be", "as", "by", "not",
        "fails", "failure", "error", "issue", "bug", "should", "does", "do", "can", "could",
    }
    return {
        token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text.lower())
        if token not in stop
    }


def _node_terms(node: ConceptNode) -> set[str]:
    text = f"{node.id} {node.label} " + " ".join(f"{k} {v}" for k, v in node.metadata.items())
    return _tokens(text)


def _score(issue_terms: set[str], node: ConceptNode) -> float:
    terms = _node_terms(node)
    overlap = len(issue_terms & terms)
    if overlap == 0:
        return 0.0
    kind_weight = {
        "method": 1.35, "function": 1.3, "class": 1.25, "type": 1.2,
        "module": 1.1, "file": 1.0, "package": 0.9, "callable_ref": 0.75,
    }.get(node.kind, 0.8)
    precision = overlap / max(1, len(terms))
    recall = overlap / max(1, len(issue_terms))
    return kind_weight * (overlap + 0.5 * precision + 0.75 * recall)


class SemanticInvestigator:
    """Deterministic first-pass localizer/hypothesis builder for later latent control."""

    def __init__(self, graph: SQLiteSemanticGraph) -> None:
        self.graph = graph

    def _expand(self, nodes: Sequence[ConceptNode], *, max_neighbors: int = 4) -> list[ConceptNode]:
        seen = {node.id for node in nodes}
        expanded = list(nodes)
        for node in nodes:
            edges = self.graph.neighbors(node.id, limit=max_neighbors)
            for edge in edges:
                other_id = edge.target if edge.source == node.id else edge.source
                if other_id in seen:
                    continue
                try:
                    other = self.graph.get_node(other_id)
                except KeyError:
                    continue
                seen.add(other_id)
                expanded.append(other)
        return expanded

    def investigate(self, issue: str, *, limit: int = 12) -> SolutionState:
        issue_terms = _tokens(issue)
        raw = self.graph.search(issue, limit=max(limit * 4, 30))
        ranked = sorted(
            ((node, _score(issue_terms, node)) for node in raw),
            key=lambda item: (-item[1], item[0].kind, item[0].label),
        )
        primary = [node for node, score in ranked if score > 0][:limit]
        candidates = self._expand(primary[: max(3, min(6, len(primary)))]) if primary else []
        by_id: dict[str, ConceptNode] = {node.id: node for node in [*primary, *candidates]}
        ordered = sorted(
            by_id.values(), key=lambda node: (-_score(issue_terms, node), node.kind, node.label)
        )[: max(limit, 12)]

        hypotheses: list[Hypothesis] = []
        location_kinds = {"method", "function", "class", "module", "file"}
        locations = [node for node in ordered if node.kind in location_kinds]
        for index, node in enumerate(locations[:5], start=1):
            supporting = [node.id]
            supporting.extend(
                edge.target if edge.source == node.id else edge.source
                for edge in self.graph.neighbors(node.id, limit=4)
            )
            confidence = min(0.90, 0.35 + 0.08 * max(1, len(issue_terms & _node_terms(node))))
            hypotheses.append(Hypothesis(
                id=f"H{index}",
                claim=f"The issue is likely connected to {node.kind} {node.label}.",
                supports=tuple(dict.fromkeys(supporting)),
                confidence=confidence,
            ))

        evidence = tuple(
            EvidenceRef(
                id=node.content.content_hash,
                source=node.id,
                metadata={"concept_id": node.id, "kind": node.kind, "label": node.label},
            )
            for node in ordered
        )
        next_actions: list[Mapping[str, Any]] = []
        for node in locations[:3]:
            path = node.metadata.get("path")
            if path:
                next_actions.append({
                    "action": "repo.read", "inputs": {"path": str(path)},
                    "reason": f"Inspect candidate {node.kind} {node.label}",
                })
        if not next_actions:
            next_actions.append({"action": "repo.search", "inputs": {"query": issue}, "reason": "Broaden repository localization"})

        cognition = CognitiveState(
            goal=issue,
            active_concepts=tuple(node.id for node in ordered),
            hypotheses=tuple(hypotheses),
            open_questions=(("Which candidate best explains the observed failure?",) if hypotheses else ("Which repository concepts implement the requested behavior?",)),
            candidate_locations=tuple(node.id for node in locations),
            next_actions=tuple(next_actions),
            terminal=False,
        )
        digest = content_hash(cognition.to_dict())
        findings = tuple(
            Finding(
                kind="hypothesis", value=h.claim, confidence=h.confidence,
                evidence=tuple(ref for ref in evidence if ref.metadata.get("concept_id") in h.supports),
                metadata={"hypothesis_id": h.id, "status": h.status},
            )
            for h in hypotheses
        )
        recommendations = tuple(
            Recommendation(kind="semantic.next_action", parameters=dict(action), evidence=evidence[:4])
            for action in next_actions
        )
        return SolutionState(
            id=f"solution:semantic:{digest.removeprefix('b3:')[:24]}",
            plan_id="semantic-investigation",
            values={"cognition": cognition.to_dict()},
            findings=findings,
            recommendations=recommendations,
            evidence=evidence,
            metadata={
                "semantic_runtime": {"schema": COGNITIVE_STATE_FORMAT, "concept_count": len(ordered)},
                "language": "",
            },
        )


__all__ = ["COGNITIVE_STATE_FORMAT", "Hypothesis", "CognitiveState", "SemanticInvestigator"]
