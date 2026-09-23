# LCFA

LCFA is a domain-neutral runtime for structured state reasoning and governed agentic action propagation.

The initial implementation is **LCFA-Zero**: it requires no custom model training and no neural weights. It establishes stable protocol boundaries so learned planners, continuous-flow reasoners, GNNs, transformers, or `safetensors` artifacts can be introduced later without changing applications or domain profiles.

## Core model

```text
Query / State
    |
    v
ReasoningPlan
    |
    v
pure Operator Graph
    |
    v
SolutionState
    |
    v
RecommendationActionCompiler
    |
    v
ActionGraph
    |
    v
capability + approval checks
    |
    v
external actions
    |
    v
observations -> next reasoning pass
```

`SolutionState` is the boundary between cognition and agency. Reasoning operators are expected to be pure. Recommendations are declarative and cannot create side effects by themselves. External effects are performed only by explicitly registered action handlers after capability and approval policy checks.

## Why this shape

- **Training optional:** deterministic execution works without a trained model.
- **Domain neutral:** core entities, plans, findings, recommendations, evidence, and actions do not encode education, code, manufacturing, or another domain.
- **Evidence preserving:** evidence attached to state values propagates through reasoning operators into the resulting solution state.
- **Governed agency:** action nodes declare capabilities, approvals, and effects before they execute.
- **Replayable:** reasoning and action execution produce typed traces.
- **Upgradeable:** future learned reasoners can implement the same `ReasoningPlan -> SolutionState` contract.

## Minimal example

```python
from lcfa import EvidenceRef, EvidenceValue, ExecutionContext, LCFA, PlanNode, ReasoningPlan

engine = LCFA()

context = ExecutionContext(
    state={
        "current": EvidenceValue([70, 72, 71], (EvidenceRef("obs:current"),)),
        "baseline": EvidenceValue([79, 80, 78], (EvidenceRef("obs:baseline"),)),
    }
)

plan = ReasoningPlan(
    id="course-change",
    nodes=(
        PlanNode("current_mean", "stats.mean", {"values": "$state.current"}),
        PlanNode("baseline_mean", "stats.mean", {"values": "$state.baseline"}),
        PlanNode(
            "delta",
            "stats.delta",
            {"current": "$node.current_mean", "baseline": "$node.baseline_mean"},
            depends_on=("current_mean", "baseline_mean"),
        ),
    ),
    outputs=("delta",),
)

solution = engine.reason(plan, context)
print(solution.values["delta"])
print(solution.evidence)
```

## Development

```bash
python -m pip install -e '.[test]'
pytest
```

## Near-term roadmap

1. Stable serialized LCFA IR for reasoning plans, solution state, action graphs, and traces.
2. Profile loader for domain ontologies, policies, operators, and action builders.
3. State-store interfaces with semantic UUID -> version -> content-hash identity.
4. Deterministic temporal, hierarchy, cohort, anomaly, evidence, and constraint operator libraries.
5. Benchmark harness shared by deterministic and learned reasoners.
6. Optional artifact loader for `model.safetensors` and learned planner/reasoner backends.
7. Durable state-propagation loop: `SolutionState -> ActionGraph -> observations -> SolutionState'`.

## Status

Pre-alpha. The current branch establishes the runtime contracts and first deterministic executor; it is not yet a production security boundary.
