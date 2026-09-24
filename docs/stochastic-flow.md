# `lcfa.stochastic-flow.v1`

`lcfa.stochastic-flow.v1` is LCFA's probabilistic reasoning backend for a frozen pretrained language model with optional session-scoped fast parametric adaptation.

The language model may propose, branch, critique, and choose pure tools, but LCFA's typed state and operators remain the authoritative computational substrate.

```text
ReasoningPlan + ExecutionContext
              |
              v
        LCFA-Zero anchor
              |
              v
      configurable backbone
         /    |    \
 candidate candidate candidate
      |        |        |
 pure tools  pure tools  pure tools
      \        |        /
       verifier/rerank
              |
      fast parametric prior
       observe -> update
              |
          next step
              |
          SolutionState
```

The selected model answer is exposed at `solution.metadata["language"]`; the complete candidate/adaptation trace is under `solution.metadata["stochastic_flow"]`.

## Backends

Backbone selection is independent from LCFA flow logic:

- `transformers-local` — local Hugging Face/PyTorch causal LM.
- `mlx-local` — Apple-Silicon-native `mlx-lm` model.
- `llama-cpp` — local GGUF through `llama-cpp-python`.
- `reference` — CI-only deterministic fixture, rejected unless the artifact is marked `reference_only`.

Example artifact configuration for an 8 GB Apple-Silicon Mac:

```json
{
  "backbone": {
    "type": "mlx-local",
    "path": "/models/my-mlx-3b",
    "max_input_tokens": 8192
  },
  "adaptation": {
    "type": "mlx-fast",
    "learning_rate": 0.08,
    "decay": 0.999,
    "score_weight": 0.5
  },
  "flow": {
    "branches": 2,
    "beam_width": 2,
    "min_steps": 1,
    "max_steps": 4,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 384
  }
}
```

Install the Apple-native extras with:

```bash
pip install -e '.[mac]'
```

Runtime overrides do not require editing the artifact:

```bash
lcfa-bench run builtin:all \
  --artifact artifacts/my-flow \
  --backbone-type mlx-local \
  --backbone-path /models/my-mlx-model \
  --adaptation-type mlx-fast \
  --adaptation-learning-rate 0.08 \
  --prior-snapshot /tmp/lcfa-prior.safetensors
```

A CPU/GGUF run can instead use:

```bash
lcfa-bench run builtin:all \
  --artifact artifacts/my-flow \
  --backbone-type llama-cpp \
  --backbone-path /models/model.gguf \
  --n-ctx 8192 \
  --n-threads 8 \
  --n-gpu-layers 0
```

## Fast parametric priors

`config.adaptation.type` is independent of the semantic backbone:

- `none` — fixed candidate scoring.
- `numpy-fast` — tiny portable online prior using NumPy.
- `mlx-fast` — the same session-scoped adaptation on MLX/unified memory.

The current fast prior does **not** fine-tune the frozen language model. It learns a small candidate-ranking function over seven observable features: model logprob, confidence, evidence validity, tool success, parse validity, finality, and verifier score. After each reasoning step, the winning candidate provides a target; the prior updates immediately and changes subsequent branch ranking in the same run.

This separation is deliberate: it is model-agnostic, cheap enough for small machines, and reversible. A later model-specific LoRA/adapter implementation can use the same adaptation interface to mutate selected backbone deltas without changing `ReasoningPlan -> SolutionState`.

Each independent `reason(...)` call resets the session prior by default. When `--prior-snapshot` or `adaptation.snapshot_path` is set, the final prior is written as `lcfa.fast-prior.v1` safetensors containing `prior.weights` and `prior.bias`.

## Candidate schema

```json
{
  "answer": "user-facing answer",
  "rationale": "brief evidence-grounded rationale",
  "confidence": 0.82,
  "evidence_ids": ["obs:123"],
  "tool_requests": [{"operator": "temporal.slope", "inputs": {"values": [1, 2, 4]}, "params": {}}],
  "final": false
}
```

`rationale` is an explicit answer artifact, not hidden chain-of-thought.

## Tool safety

Only operators matching `config.tools.allowed` may be invoked. Stochastic reasoning can access `OperatorRegistry` only; `ActionRegistry` is not reachable. External effects still require `SolutionState -> ActionGraph -> capability/approval checks -> action`.

## Safetensors scoring

The LCFA flow artifact's `model.safetensors` contains scalar scoring weights such as `score.logprob`, `score.confidence`, `score.evidence`, `score.tool`, `score.parse`, `score.final`, and `score.verifier`. A pretrained backbone may independently use its own safetensors/sharded-safetensors or GGUF representation.

The built-in 17-case corpus remains a semantic regression suite, not a SOTA-reasoning benchmark.
