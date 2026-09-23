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

## Stable LCFA IR

Reasoning plans, solution states, action graphs, and execution traces can be serialized through the versioned `lcfa.ir.v1` JSON envelope.

```python
from lcfa import dumps_ir, loads_ir

encoded = dumps_ir(plan)
restored = loads_ir(encoded)
assert restored == plan
```

The protocol artifacts themselves carry schema versions such as `lcfa.plan.v1`, `lcfa.solution.v1`, and `lcfa.action_graph.v1`. This keeps the persisted execution format independent from whichever planner or reasoner produced it.

## Declarative profiles

Domain behavior belongs in profiles rather than the LCFA core. Profiles declare entity/relation vocabularies and the operators/actions they expect.

```python
from lcfa import LCFA

engine = LCFA.from_profile("profiles/education")
assert engine.profile.id == "education"
```

`profiles/education/profile.toml` is the first reference profile; education-specific entity types do not leak into the core runtime.

## Semantic state identity

LCFA separates logical identity from immutable content identity:

```text
semantic ID
    |
    v
snapshot version
    |
    v
BLAKE3(canonical payload)
```

`MemoryStateStore` is the reference implementation. Production backends can implement the same `StateStore` protocol with SQLite/libSQL, LMDB, RocksDB, object storage, or another persistence layer.

```python
from lcfa import MemoryStateStore

store = MemoryStateStore()
v1 = store.put("student:123", {"gpa": 3.2})
v2 = store.put("student:123", {"gpa": 3.4}, expected_version=1)

assert v1.identity.semantic_id == v2.identity.semantic_id
assert v1.identity.content_hash != v2.identity.content_hash
```

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

Completed in the initial runtime slice:

- stable serialized LCFA IR for reasoning plans, solution state, action graphs, and traces
- declarative profile loader
- semantic-ID -> version -> BLAKE3 content-hash state-store contracts
- reference education profile

Next:

1. Deterministic temporal, hierarchy, cohort, anomaly, evidence, and constraint operator libraries.
2. Benchmark harness shared by deterministic and learned reasoners.
3. Optional artifact loader for `model.safetensors` and learned planner/reasoner backends.
4. Durable state-propagation loop: `SolutionState -> ActionGraph -> observations -> SolutionState'`.
5. Persistent state-store backends and CAS adapters.

## Status

Pre-alpha. The current branch establishes the runtime contracts and deterministic executor; it is not yet a production security boundary.
