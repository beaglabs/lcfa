"""Compilation helpers that turn declarative recommendations into ActionGraphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from .protocol import ActionGraph, ActionNode, Recommendation, SolutionState

ActionBuilder = Callable[[Recommendation, SolutionState, int], tuple[ActionNode, ...]]


@dataclass(slots=True)
class RecommendationActionCompiler:
    """Domain-neutral compiler with profile-supplied recommendation builders."""

    builders: dict[str, ActionBuilder]

    def __init__(self) -> None:
        self.builders = {}

    def register(self, recommendation_kind: str, builder: ActionBuilder) -> None:
        if recommendation_kind in self.builders:
            raise ValueError(f"builder already registered: {recommendation_kind}")
        self.builders[recommendation_kind] = builder

    def compile(self, solution: SolutionState) -> ActionGraph:
        nodes: list[ActionNode] = []
        for index, recommendation in enumerate(solution.recommendations):
            try:
                builder = self.builders[recommendation.kind]
            except KeyError as exc:
                raise KeyError(
                    f"no action builder registered for recommendation: {recommendation.kind}"
                ) from exc
            nodes.extend(builder(recommendation, solution, index))

        return ActionGraph(
            id=f"actions:{uuid4()}",
            source_solution_id=solution.id,
            nodes=tuple(nodes),
            metadata={"source_plan_id": solution.plan_id},
        )
