# LCFA Mixed Real-World Benchmark

This harness builds one balanced, materialized LCFA suite from six established benchmarks:

- **GSM8K** (`openai/gsm8k`) — exact final numeric answer.
- **HumanEval** (`openai/openai_humaneval`) — native Python tests, pass/fail.
- **SWE-bench Lite** — inference context from `princeton-nlp/SWE-bench_Lite_bm25_13K`, scored by the official `SWE-bench/SWE-bench_Lite` Docker harness.
- **RULER** — pre-generated RULER workloads from `simonjegou/ruler`, stratified across RULER tasks.
- **AA-LCR 1.1** (`ArtificialAnalysis/AA-LCR`) — full source document sets and the benchmark's long-context prompt shape.
- **MMLU** (`cais/mmlu`, `all`) — exact A/B/C/D answer, stratified by subject.

The suite uses the same number of cases from every source, so `summary.case_pass_rate` is also the equal-weight macro score across the six families. Every case is tagged by source, so ordinary LCFA report `tag_summaries` contain the per-benchmark scores.

## Install

```bash
python -m pip install -e '.[mac,benchmarks]'
```

HumanEval uses the local Python interpreter. SWE-bench additionally requires the official `swebench` package and working Docker:

```bash
python -m pip install swebench
docker info
```

## Build a reproducible suite

```bash
lcfa-bench-mixed build \
  --preset quick \
  --seed 20260923 \
  --ruler-context 4096 \
  -o /tmp/lcfa-mixed-quick.json
```

Presets are `quick` = 3 cases/source (18 total), `standard` = 10/source (60 total), and `large` = 25/source (150 total). `--cases-per-source N` overrides the preset size while preserving equal weighting.

The builder downloads upstream data, selects a deterministic balanced sample, embeds the exact selected prompts and answer metadata in an ordinary `lcfa.bench.v1` JSON file, and deterministically shuffles all six sources. This means subsequent subjects run against the same materialized cases even if upstream row ordering changes.

## Run the current Mac artifact

```bash
lcfa-bench-mixed run /tmp/lcfa-mixed-quick.json \
  --artifact artifacts/lcfa-stochastic-flow-mac \
  --prior-snapshot /tmp/lcfa-prior.safetensors \
  --allow-code-exec \
  --swebench-eval \
  --repeats 1 \
  -o /tmp/lcfa-mixed-mac.json
```

The output is a normal `lcfa.bench.report.v1`, so the existing comparison command works unchanged:

```bash
lcfa-bench compare \
  /tmp/qwen-vanilla-mixed.json \
  /tmp/lcfa-mixed-mac.json \
  --baseline qwen-vanilla
```

## Scoring

### GSM8K

The model emits a final number. The grader extracts the last numeric value and compares it exactly with the value after `####` in the official answer.

### HumanEval

The model emits Python. The grader combines a completion with the official prompt, runs the official `test` plus `check(entry_point)`, and records pass/fail. Candidate code execution is disabled unless `--allow-code-exec` is supplied. Execution uses a fresh temporary directory and `python -I`, but this is process isolation, not a hardened security sandbox.

### SWE-bench Lite

The model receives the BM25 13K inference `text` and emits a patch. With `--swebench-eval`, LCFA calls the official SWE-bench evaluation harness for that Lite instance and reads its `resolved` result. It does not substitute string matching against the gold patch.

### RULER

The default materialization uses the 4096-token configuration and stratifies across RULER task names. A case passes when every normalized expected answer is present in the returned answer. `--ruler-context 8192` and `16384` are also available.

### AA-LCR 1.1

The builder downloads the official extracted-text archive, preserves `data_source_filenames` order, and reconstructs the benchmark document/question prompt. Current AA-LCR cases are roughly 72K–115K input tokens.

By default the runner uses a conservative normalized answer-equivalence check. This is useful for local LCFA-vs-baseline experiments but is not claimed to reproduce the official AA-LCR LLM equality judge. For leaderboard-parity work, pass `--lcr-judge-cmd`. The command receives JSON on stdin with `question`, `reference`, and `candidate`, and must output `{"passed": true}` or `{"passed": false}`.

### MMLU

The `all` test split is sampled across subjects. The grader accepts the final standalone A/B/C/D choice and compares it with the official class label.

## Long-context warning for the current Mac artifact

`artifacts/lcfa-stochastic-flow-mac` currently advertises a 4,096-token input limit, while RULER 4096 already consumes approximately the nominal context budget before LCFA's structured wrapper, SWE-bench's BM25 profile is 13K tokens, and AA-LCR is roughly 72K–115K tokens.

The harness intentionally keeps those real prompts. It does **not** silently replace SWE-bench or AA-LCR with short toy versions. Running the 4K artifact is therefore useful for exposing its current long-context limitation, but a full-fidelity comparison on those families needs a backbone/runtime configured for the corresponding context size.

## Report interpretation

Use the top-level `case_pass_rate` as the equal-source mixed score only when the suite was produced by this balanced builder. Per-source results are available at:

```text
tag_summaries.gsm8k.case_pass_rate
tag_summaries.humaneval.case_pass_rate
tag_summaries.swebench.case_pass_rate
tag_summaries.ruler.case_pass_rate
tag_summaries.lcr.case_pass_rate
tag_summaries.mmlu.case_pass_rate
```

Latency measures the LCFA `reason()` call. HumanEval/SWE-bench grading time is intentionally outside model latency because it is evaluator infrastructure rather than inference time.
