# LCFA artifacts

LCFA weight-backed reasoners use a versioned directory artifact:

```text
my-model/
├── artifact.json
└── model.safetensors
```

`artifact.json` uses `format: lcfa.artifact.v1` and identifies the architecture, base backend, safetensors file, SHA-256 digest, declared tensor contract, version, and metadata. The loader rejects unsupported formats/architectures, path escapes, missing weights, and digest mismatches before execution.

The architecture string is a dispatch key. Built-in support currently includes:

- `lcfa.weighted-output.v1` — reference safetensors-backed adapter that executes the normal `ReasoningPlan -> SolutionState` path and applies tensor-backed affine calibration to floating-point solution values. It exists to prove the artifact/load/execute/benchmark contract; it is **not** presented as a trained continuous-flow model.

Architectures can be registered through `register_artifact_architecture(...)` without changing benchmark suites or callers.

## Reference artifact

`artifacts/lcfa-weighted-identity` is intentionally identity-initialized (`scale=1`, `bias=0`) and has no training. CI runs it against `builtin:all` so changes to artifact loading, tensor execution, or reasoner interchangeability cannot silently break the canonical LCFA-Bench corpus.

```bash
lcfa-bench run builtin:all \
  --artifact artifacts/lcfa-weighted-identity \
  --name lcfa-weighted-identity \
  --repeats 5 \
  -o weighted.json
```

A future continuous/ODE/transformer LCFA artifact should define a new architecture key while retaining the exact same public reasoner contract:

```python
reason(plan: ReasoningPlan, context: ExecutionContext) -> SolutionState
```
