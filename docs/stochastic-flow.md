# `lcfa.stochastic-flow.v1`

`lcfa.stochastic-flow.v1` is LCFA's probabilistic reasoning backend for a frozen pretrained language model.

It keeps one invariant: the language model may propose, branch, critique, and choose tools, but LCFA's typed state and pure operators remain the authoritative computational substrate.

```text
ReasoningPlan + ExecutionContext
              |
              v
        LCFA-Zero anchor
              |
              v
      frozen LM backbone
         /    |    \
 candidate candidate candidate
      |        |        |
 pure tools  pure tools  pure tools
      \        |        /
       verifier/rerank
              |
          next step
              |
          SolutionState
```

The stochastic backend currently preserves the anchor's `values`, `findings`, `recommendations`, and evidence semantics. The selected model answer is exposed at:

```python
solution.metadata["language"]
solution.metadata["stochastic_flow"]
```

Generated language therefore cannot silently replace trusted numeric/operator outputs.

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

`rationale` is an explicit answer artifact, not hidden chain-of-thought. The runtime does not request or persist private reasoning traces.

## Test-time compute

Artifacts control branch count, beam width, min/max reasoning steps, temperature, top-p, generation length, and verifier top-k. Harder deployments can spend more inference compute without changing the public LCFA contract.

## Tool safety

Only operators matching `config.tools.allowed` may be invoked. Stochastic reasoning can access `OperatorRegistry` only; `ActionRegistry` is not reachable. External effects still require `SolutionState -> ActionGraph -> capability/approval checks -> action`.

## Safetensors scoring

The LCFA artifact's `model.safetensors` contains scalar scoring weights:

- `score.logprob`
- `score.confidence`
- `score.evidence`
- `score.tool`
- `score.parse`
- `score.final`
- `score.verifier`

They combine model transition log probability, declared confidence, evidence validity, tool success, JSON validity, finality, and verifier judgment. They can later be analytically chosen, closed-form fit, or trained without changing the runtime.

## Backbone

The production adapter loads a local Hugging Face causal-LM directory lazily. The large pretrained checkpoint can therefore be swapped independently of the LCFA flow artifact.

The existing 17-case corpus is a semantic regression suite, not a SOTA-reasoning benchmark. A quality corpus should add ambiguous retrieval, noisy/partial evidence, multi-hop planning, long context, uncertainty, operation selection, and open-ended explanation.
