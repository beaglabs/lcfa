# LCFA-Bench suites

This directory is reserved for reusable `lcfa.bench.v1` benchmark suites.

A suite is backend-neutral: it contains a serialized `ReasoningPlan`, serialized `ExecutionContext`, typed value assertions, optional evidence expectations, tags, and metadata. The same suite should be run unchanged against LCFA-Zero and any learned/continuous-flow backend.

Recommended benchmark families:

- `retrieval` — entity/state localization and candidate recall
- `temporal` — windows, trend, change-point, timeline reconstruction
- `hierarchy` — parent/child expansion, multi-hop paths, aggregation
- `cohort` — peer comparisons, percentile, effect size
- `anomaly` — robust deviation and smoothing behavior
- `evidence` — provenance retention, precision/recall, contradiction handling
- `constraints` — policy/typed constraint satisfaction
- `end_to_end` — multi-operator plans with evidence-grounded `SolutionState` outputs

Use tags aggressively so one suite can produce both an overall report and capability-level summaries.

## CLI

```bash
lcfa-bench run benchmarks/core.json --name lcfa-zero --repeats 5 -o zero.json
lcfa-bench run benchmarks/core.json --factory my_model:create_engine --name lcfa-learned -o learned.json
lcfa-bench compare zero.json learned.json --baseline lcfa-zero -o comparison.json
```

A custom factory must return either a `ReasonerSubject` or an object exposing `reason(plan, context) -> SolutionState`.

## Comparison policy

LCFA-Bench reports independent metrics and baseline deltas rather than combining correctness, evidence quality, latency, and determinism into one composite score. This keeps tradeoffs explicit and makes regressions easier to diagnose.
