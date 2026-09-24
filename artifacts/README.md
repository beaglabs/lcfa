# LCFA artifacts

LCFA weight-backed reasoners use a versioned directory artifact:

```text
my-model/
├── artifact.json
└── model.safetensors
```

`artifact.json` uses `format: lcfa.artifact.v1` and identifies the architecture, base backend, safetensors file, SHA-256 digest, declared tensor contract, version, runtime config, and metadata. The loader rejects unsupported formats/architectures, path escapes, missing weights, and digest mismatches before execution.

Built-in architectures:

- `lcfa.weighted-output.v1` — reference affine safetensors adapter used to prove artifact/load/execute/benchmark interchangeability.
- `lcfa.stochastic-flow.v1` — probabilistic test-time reasoning over a frozen pretrained causal-LM backbone. It anchors on LCFA's typed solution state, samples candidate reasoning states, optionally invokes whitelisted pure operators, verifies/reranks candidates, and writes selected language/rationale back into `SolutionState.metadata` without allowing generated text to directly create side effects.

## Stochastic-flow artifact

A production stochastic artifact keeps its LCFA scoring/config tensors in `model.safetensors` and points at a local pretrained model directory:

```json
{
  "format": "lcfa.artifact.v1",
  "id": "my-lcfa-reasoner",
  "version": "1.0.0",
  "architecture": "lcfa.stochastic-flow.v1",
  "base_backend": "lcfa-zero",
  "weights": {"file": "model.safetensors", "format": "safetensors", "sha256": "..."},
  "config": {
    "backbone": {"type": "transformers-local", "path": "backbone", "local_files_only": true, "device_map": "auto", "dtype": "auto"},
    "flow": {"branches": 4, "beam_width": 2, "min_steps": 1, "max_steps": 4, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 512},
    "verifier": {"enabled": true, "top_k": 2},
    "tools": {"allowed": ["stats.*", "temporal.*", "cohort.*", "hierarchy.*", "anomaly.*", "evidence.*", "constraints.*", "retrieval.*"]}
  }
}
```

Install the optional local model runtime:

```bash
pip install -e '.[transformers]'
```

Run against a local pretrained model:

```bash
lcfa-bench run builtin:all \
  --artifact artifacts/my-lcfa-reasoner \
  --backbone-path /models/my-instruct-model \
  --name my-lcfa-reasoner \
  --repeats 1 \
  -o stochastic.json
```

The Transformers adapter defaults to `local_files_only=True`, supporting disconnected/on-prem deployments.

## Reference artifacts

`artifacts/lcfa-weighted-identity` is intentionally identity-initialized and has no training.

`artifacts/lcfa-stochastic-flow-reference` uses a tiny reference-only backbone so CI can exercise sampling, verification, scoring, merge, serialization, and benchmark integration without downloading a large model. It is **not** a quality benchmark and must not be presented as a SOTA backbone.

All artifact reasoners retain:

```python
reason(plan: ReasoningPlan, context: ExecutionContext) -> SolutionState
```
