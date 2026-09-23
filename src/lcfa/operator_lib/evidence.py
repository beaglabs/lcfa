"""Evidence quality and contradiction operators."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from ..protocol import EvidenceRef, ExecutionContext, OperatorResult
from ..registry import OperatorRegistry, OperatorSpec
from ._util import canonical_key, mapping, sequence


def _evidence_id(value: object) -> str:
    if isinstance(value, EvidenceRef):
        return value.id
    if isinstance(value, Mapping) and "id" in value:
        return str(value["id"])
    return str(value)


def _coverage(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    _params: Mapping[str, object],
) -> OperatorResult:
    required = {_evidence_id(value) for value in sequence(inputs["required"], "evidence.coverage.required")}
    available = {_evidence_id(value) for value in sequence(inputs["available"], "evidence.coverage.available")}
    covered = sorted(required & available)
    missing = sorted(required - available)
    ratio = 1.0 if not required else len(covered) / len(required)
    return OperatorResult(
        value={
            "required": len(required),
            "covered": len(covered),
            "coverage": ratio,
            "covered_ids": covered,
            "missing_ids": missing,
        }
    )


def _contradictions(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    claims = sequence(inputs["claims"], "evidence.contradictions.claims")
    key_field = str(params.get("key_field", "key"))
    value_field = str(params.get("value_field", "value"))
    source_field = str(params.get("source_field", "source"))

    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for raw_claim in claims:
        claim = mapping(raw_claim, "evidence contradiction claim")
        key = str(claim[key_field])
        value = claim[value_field]
        source = claim.get(source_field)
        grouped[key][canonical_key(value)].append({"value": value, "source": source})

    contradictions: list[dict[str, object]] = []
    for key, variants in grouped.items():
        if len(variants) <= 1:
            continue
        contradictions.append(
            {
                "key": key,
                "variants": [items for items in variants.values()],
                "variant_count": len(variants),
            }
        )
    contradictions.sort(key=lambda item: str(item["key"]))
    return OperatorResult(value=contradictions, metadata={"contradiction_count": len(contradictions)})


def _require_coverage(
    _context: ExecutionContext,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> OperatorResult:
    coverage = float(inputs["coverage"])
    minimum = float(params.get("minimum", 1.0))
    passed = coverage >= minimum
    if bool(params.get("raise_on_fail", False)) and not passed:
        raise ValueError(f"evidence coverage {coverage:.3f} is below required minimum {minimum:.3f}")
    return OperatorResult(value={"passed": passed, "coverage": coverage, "minimum": minimum})


def register_evidence_operators(registry: OperatorRegistry) -> OperatorRegistry:
    registry.register(OperatorSpec("evidence.coverage", _coverage, description="Measure required evidence coverage by stable evidence ID."))
    registry.register(OperatorSpec("evidence.contradictions", _contradictions, description="Detect conflicting values asserted for the same evidence key."))
    registry.register(OperatorSpec("evidence.require_coverage", _require_coverage, description="Evaluate or enforce a minimum evidence-coverage threshold."))
    return registry
