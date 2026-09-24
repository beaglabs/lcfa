"""Training-free lexical retrieval operators for LCFA-Zero.

These operators provide a deterministic state-localization baseline for
LCFA-Bench. Learned Latent Space 1 implementations can target the same plan
semantics and benchmark cases later.
"""

from __future__ import annotations

import re
from typing import Mapping

from ..protocol import ExecutionContext, OperatorResult
from ..registry import OperatorRegistry, OperatorSpec
from ._util import mapping, sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: object) -> set[str]:
    return set(_TOKEN_RE.findall(str(value).lower()))


def _candidate_text(candidate: Mapping[str, object], fields: tuple[str, ...]) -> str:
    return " ".join(str(candidate.get(field, "")) for field in fields)


def _rank(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    query_terms = _tokens(inputs["query"])
    candidates = sequence(inputs["candidates"], "retrieval.rank.candidates")
    id_field = str(params.get("id_field", "id"))
    raw_fields = params.get("text_fields", ("text",))
    if isinstance(raw_fields, str):
        fields = (raw_fields,)
    else:
        fields = tuple(str(item) for item in sequence(raw_fields, "retrieval.rank.text_fields"))
    top_k = int(params.get("top_k", len(candidates)))
    if top_k < 1:
        raise ValueError("retrieval.rank top_k must be >= 1")

    ranked: list[dict[str, object]] = []
    for raw in candidates:
        candidate = mapping(raw, "retrieval candidate")
        candidate_id = str(candidate[id_field])
        candidate_terms = _tokens(_candidate_text(candidate, fields))
        overlap = len(query_terms & candidate_terms)
        recall = 0.0 if not query_terms else overlap / len(query_terms)
        precision = 0.0 if not candidate_terms else overlap / len(candidate_terms)
        # Favors complete query coverage first, then tighter candidates.
        score = recall + precision * 0.25
        ranked.append(
            {
                "id": candidate_id,
                "score": score,
                "matched_terms": sorted(query_terms & candidate_terms),
            }
        )

    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return OperatorResult(
        value=ranked[:top_k],
        metadata={"candidate_count": len(ranked), "top_k": top_k, "query_terms": sorted(query_terms)},
    )


def _recall_at_k(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    ranked = sequence(inputs["ranked"], "retrieval.recall_at_k.ranked")
    relevant = {str(item) for item in sequence(inputs["relevant"], "retrieval.recall_at_k.relevant")}
    k = int(params.get("k", len(ranked)))
    if k < 1:
        raise ValueError("retrieval.recall_at_k k must be >= 1")
    predicted = []
    for raw in ranked[:k]:
        item = mapping(raw, "retrieval ranked item")
        predicted.append(str(item["id"]))
    hits = relevant & set(predicted)
    recall = 1.0 if not relevant else len(hits) / len(relevant)
    return OperatorResult(
        value={
            "k": k,
            "recall": recall,
            "hits": sorted(hits),
            "retrieved": predicted,
            "relevant": sorted(relevant),
        }
    )


def register_retrieval_operators(registry: OperatorRegistry) -> OperatorRegistry:
    registry.register(
        OperatorSpec(
            "retrieval.rank",
            _rank,
            description="Rank candidate records by deterministic lexical query coverage.",
        )
    )
    registry.register(
        OperatorSpec(
            "retrieval.recall_at_k",
            _recall_at_k,
            description="Measure candidate recall at a fixed retrieval depth.",
        )
    )
    return registry
