# Progress reporting and adaptive compute

`lcfa-bench run` reports human-readable progress to **stderr** by default. JSON output remains clean on stdout or in the `-o/--output` file.

Example:

```text
[lcfa] loading artifacts/lcfa-stochastic-flow-mac; suite=lcfa-core-all cases=17 repeats=1 warmup=0
[lcfa] subject ready: lcfa-stochastic-flow-mac backend=stochastic-flow
[lcfa] case 1/17 retrieval.rank-exact-domain-terms (repeat 1/1) start [1/17]
[lcfa]   adaptive budget: start=1 branch, max=2 branches x 3 steps
[lcfa]   step 1/3 parents=1 initial_branches=1
[lcfa]     model call 1 proposal: branches=1 max_new_tokens=256
[lcfa]     model call 1 proposal done in 18.2s; samples=1
[lcfa]   step 1 -> stop; confidence=0.91 evidence=1.00 uncertainty=none
[lcfa] case 1/17 retrieval.rank-exact-domain-terms done in 18.3s steps=1 candidates=1 verifiers=0
```

Use `--quiet` to disable progress output.

## Adaptive compute

The base stochastic-flow settings remain hard ceilings:

- `flow.branches` is the maximum proposal branches per parent.
- `flow.max_steps` is the maximum reasoning depth.
- `verifier.top_k` is the maximum verifier candidates per step.

`flow.adaptive_compute` controls how much of that budget is spent initially and when LCFA escalates:

```json
{
  "flow": {
    "branches": 2,
    "max_steps": 3,
    "adaptive_compute": {
      "enabled": true,
      "initial_branches": 1,
      "confidence_threshold": 0.82,
      "evidence_threshold": 0.5,
      "score_margin_threshold": 0.15,
      "verifier_accept_threshold": 0.85,
      "expand_when_uncertain": true,
      "verify_when_uncertain": true,
      "require_parsed": true
    }
  }
}
```

An easy case starts with one candidate. If the result is parsed, final, sufficiently confident, and sufficiently evidence-grounded, LCFA stops without generating the second branch or invoking the verifier.

An uncertain case can spend the remaining branch budget. If uncertainty remains, LCFA invokes the verifier for up to `verifier.top_k` candidates. It proceeds to another reasoning step only when the selected candidate remains uncertain, is not final, or requested a pure operator that requires follow-up.

The final `SolutionState` records the actual compute used at:

```python
solution.metadata["stochastic_flow"]["adaptive_compute"]
```

including:

- `steps_used`
- `generated_candidates`
- `verifier_calls`
- `expansions`
- per-step uncertainty reasons and stop/continue decisions

## 8 GB Apple-Silicon preset

`artifacts/lcfa-stochastic-flow-mac` enables adaptive compute by default. With the local MLX model already downloaded, run:

```bash
lcfa-bench run builtin:all \
  --artifact artifacts/lcfa-stochastic-flow-mac \
  --prior-snapshot /tmp/lcfa-prior.safetensors \
  --repeats 1 \
  -o /tmp/lcfa-mac.json
```

The progress stream makes long prompt-ingestion/model calls visible instead of appearing hung, while adaptive compute avoids paying the maximum branch/verifier budget on cases that are already resolved.
