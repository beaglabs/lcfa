"""High-level LCFA runtime facade."""

from __future__ import annotations

from .actions import RecommendationActionCompiler
from .operators import register_core_operators
from .protocol import ActionGraph, ActionRun, ExecutionContext, ReasoningPlan, SolutionState
from .registry import ActionRegistry, OperatorRegistry
from .runtime import ActionExecutor, ReasoningExecutor


class LCFA:
    """Training-optional LCFA runtime.

    The default instance is LCFA-Zero: a deterministic operator runtime with no
    learned weights. Profiles can register more operators, action handlers,
    planners, reasoners, or neural backends without changing the public API.
    """

    def __init__(
        self,
        *,
        operators: OperatorRegistry | None = None,
        actions: ActionRegistry | None = None,
        action_compiler: RecommendationActionCompiler | None = None,
    ) -> None:
        self.operators = operators or register_core_operators(OperatorRegistry())
        self.actions = actions or ActionRegistry()
        self.action_compiler = action_compiler or RecommendationActionCompiler()
        self.reasoning = ReasoningExecutor(self.operators)
        self.agentic = ActionExecutor(self.actions)

    def reason(self, plan: ReasoningPlan, context: ExecutionContext) -> SolutionState:
        return self.reasoning.execute(plan, context)

    def compile_actions(self, solution: SolutionState) -> ActionGraph:
        return self.action_compiler.compile(solution)

    def act(
        self,
        graph: ActionGraph,
        solution: SolutionState,
        context: ExecutionContext,
    ) -> ActionRun:
        return self.agentic.execute(graph, solution, context)
