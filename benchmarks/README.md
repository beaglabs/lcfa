# LCFA-Bench suites

LCFA-Bench defines backend-neutral workloads that contain an exact `ReasoningPlan`, exact `ExecutionContext`, typed value assertions, optional evidence expectations, tags, and metadata. The same workload is run unchanged against LCFA-Zero and future learned/continuous-flow backends.

## Reference corpus

The built-in `lcfa-core` corpus currently contains **17 fixed cases across eight suites**:

- `retrieval` — lexical state localization and recall@k
- `temporal` — windows, trend, change-point detection
- `hierarchy` — descendant expansion, multi-hop paths, aggregation
- `cohort` — peer comparison, percentile, summary/effect size
- `anomaly` — robust deviation and EWMA smoothing
- `evidence` — provenance propagation, coverage, contradiction handling
- `constraints` — typed policy checks, filtering, composition
- `end_to_end` — multi-operator plans producing evidence-grounded `SolutionState` outputs

`manifest.json` freezes the suite and case IDs. Any intentional corpus change should update that manifest and the corpus tests in the same change.

The canonical source lives in `lcfa.bench_corpus`; built-ins can be exported to portable `lcfa.bench.v1` JSON whenever a standalone artifact is needed.

## CLI

List the canonical suites:

```bash
lcfa-bench list
```

Run one family or the complete corpus against LCFA-Zero:

```bash
lcfa-bench run builtin:temporal --name lcfa-zero --repeats 5 -o temporal-zero.json
lcfa-bench run builtin:all --name lcfa-zero --repeats 5 -o core-zero.json
```

Export a built-in suite as portable JSON:

```bash
lcfa-bench export builtin:all -o lcfa-core-all.json
```

Run the exact same corpus against a learned backend:

```bash
lcfa-bench run builtin:all \
  --factory my_model:create_engine \
  --name lcfa-learned \
  --repeats 5 \
  -o learned.json
```

Compare reports without changing the workload:

```bash
lcfa-bench compare core-zero.json learned.json \
  --baseline lcfa-zero \
  -o comparison.json
```

A custom factory must return either a `ReasonerSubject` or an object exposing `reason(plan, context) -> SolutionState`.

## Baseline contract

For the current reference corpus, LCFA-Zero is the semantic oracle and must maintain:

- case pass rate: `1.0`
- assertion accuracy: `1.0`
- error rate: `0.0`
- replay determinism: `1.0`
- evidence F1: `1.0` for suites/cases that specify evidence expectations

Latency is measured and reported but is intentionally **not frozen** because it depends on hardware and runtime environment. See `BASELINE.md`.

## Comparison policy

LCFA-Bench reports independent metrics and baseline deltas rather than combining correctness, evidence quality, latency, and determinism into one composite score. This keeps tradeoffs explicit and makes regressions easier to diagnose.

A learned backend may legitimately trade replay determinism for capability, but correctness and evidence retention remain independently visible. This prevents stochasticity from being silently conflated with wrong answers.
