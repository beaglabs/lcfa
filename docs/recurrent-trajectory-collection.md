# Recurrent trajectory collection

`lcfa-recurrent collect` builds supervised LCFA controller data from real coding repairs without using SWE-bench as training data.

Each task runs in a detached git worktree. The collector:

1. checks out the requested `base_ref`;
2. optionally runs `setup_argv`;
3. runs `verify_argv` before the agent;
4. skips the task by default when the verifier already passes;
5. indexes the worktree into the LCFA semantic graph;
6. runs the semantic teacher with auto-approval inside that isolated worktree;
7. runs the same verifier again;
8. writes the semantic episode with an explicit `success` label from the verifier;
9. removes the worktree unless `--keep-worktrees` is set.

The command never interprets shell strings for setup or verification. Commands must be argv arrays.

## Task format

One JSON object per line:

```json
{"schema_version":"lcfa.recurrent-task.v1","id":"example-001","repo":"/path/to/local/git/repo","base_ref":"buggy-commit","goal":"Fix normalize_name so None is preserved","verify_argv":["python","-m","pytest","tests/test_models.py::test_optional_name","-q"],"setup_argv":["python","-m","pip","install","-e","."],"timeout_seconds":300,"metadata":{"source":"internal-regression"}}
```

Required fields:

- `id`: unique task identifier.
- `repo`: local git repository path.
- `goal`: repair request presented to the teacher.
- `verify_argv`: exact command whose exit code determines failure/success.

Optional fields:

- `base_ref`: commit/tag/ref to detach at; default `HEAD`.
- `setup_argv`: one setup command run before baseline verification.
- `timeout_seconds`: timeout for setup and verification commands; default 300.
- `metadata`: arbitrary task provenance.

`setup_argv` should prepare dependencies/environment only. It should not make the source repair itself, because tracked setup modifications would become part of the agent patch.

## Collect

```bash
lcfa-recurrent collect data/recurrent/tasks.jsonl \
  --artifact artifacts/lcfa-stochastic-flow-mac \
  --output-dir data/recurrent/episodes \
  --worktree-root /tmp/lcfa-recurrent-worktrees \
  --max-steps 12
```

Useful smoke test:

```bash
lcfa-recurrent collect data/recurrent/tasks.jsonl \
  --artifact artifacts/lcfa-stochastic-flow-mac \
  --output-dir /tmp/lcfa-episodes \
  --max-tasks 3 \
  --max-steps 8
```

Live docs remain disabled unless `--allow-docs` is supplied. Keep docs disabled for the first controlled experiment.

The collector writes one episode per collected task plus `collection.json`:

```text
data/recurrent/episodes/
├── task-001.json
├── task-002.json
└── collection.json
```

Each episode remains `lcfa.semantic-trajectory.v1` and adds:

- top-level `success`: the post-agent verifier result;
- `metadata.base_commit`;
- baseline verifier stdout/stderr/return code;
- post-agent verifier stdout/stderr/return code;
- teacher artifact and collection settings.

This means `lcfa-recurrent prepare` can use the episodes directly and the value target comes from a real external verification result rather than the agent declaring itself finished.

## Prepare and train

```bash
lcfa-recurrent prepare \
  data/recurrent/episodes/*.json \
  -o data/recurrent/transitions.jsonl

lcfa-recurrent train \
  data/recurrent/transitions.jsonl \
  -o artifacts/lcfa-rwkv-controller-real-001 \
  --model RWKV/RWKV7-G1j-1.5B-20260831 \
  --device mps \
  --dtype float16 \
  --epochs 5 \
  --validation-fraction 0.2
```

The split is by whole episode, so adjacent recurrent states from one trajectory cannot leak into validation.

## Data policy for the SWE-bench experiment

Do not use SWE-bench Lite/Verified issues, gold patches, test patches, or benchmark-specific solution material as trajectory training data. Use internal regressions, synthetic mutations, or unrelated open-source issue/fix pairs. Reserve the frozen SWE-bench tasks for evaluation.
