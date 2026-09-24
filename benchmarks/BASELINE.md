# LCFA-Zero reference baseline

This document defines the semantic baseline for the built-in `lcfa-core` benchmark corpus.

## Frozen semantic targets

For every release or branch that claims compatibility with the current corpus, LCFA-Zero should achieve:

| Metric | Required baseline |
| --- | ---: |
| Case pass rate | 1.0 |
| Assertion accuracy | 1.0 |
| Error rate | 0.0 |
| Replay determinism rate | 1.0 |
| Evidence F1 | 1.0 where evidence is asserted |

These are regression gates, not an overall product score.

## What is intentionally not frozen

Wall-clock latency (`mean`, `p50`, `p95`) is recorded by LCFA-Bench but is environment-sensitive. It should be compared only across subjects measured on the same hardware/runtime configuration.

Trace-step counts are reported as an execution-complexity signal but may legitimately change if equivalent implementations fuse or restructure operators.

## Learned backend policy

A future learned or `model.safetensors` backend should run the **same built-in suite without modifying its plans, contexts, expected values, or evidence requirements**. Comparisons should preserve raw metrics rather than collapse them into a composite score.

A learned backend may be nondeterministic and still pass correctness assertions. Replay determinism is therefore reported independently. Any correctness or evidence regression remains visible even if the learned backend is faster or more capable on additional workloads.

## Corpus identity

The current corpus identity is tracked by `benchmarks/manifest.json`. The manifest fixes suite IDs and case IDs; semantic changes to an existing case should be treated as a benchmark-version change rather than silently rewriting the oracle.
