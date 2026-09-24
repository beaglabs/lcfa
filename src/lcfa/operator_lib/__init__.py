"""Generalized deterministic reasoning operator library for LCFA-Zero."""

from __future__ import annotations

from ..registry import OperatorRegistry
from .anomaly import register_anomaly_operators
from .cohort import register_cohort_operators
from .constraints import ConstraintViolation, register_constraint_operators
from .evidence import register_evidence_operators
from .hierarchy import register_hierarchy_operators
from .retrieval import register_retrieval_operators
from .temporal import register_temporal_operators


def register_reasoning_operators(registry: OperatorRegistry) -> OperatorRegistry:
    register_retrieval_operators(registry)
    register_temporal_operators(registry)
    register_hierarchy_operators(registry)
    register_cohort_operators(registry)
    register_anomaly_operators(registry)
    register_evidence_operators(registry)
    register_constraint_operators(registry)
    return registry


__all__ = ["ConstraintViolation", "register_reasoning_operators"]
