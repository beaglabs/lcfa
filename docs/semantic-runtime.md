# LCFA Semantic Runtime v0.1

The semantic runtime makes repository and terminal work a stateful LCFA problem instead of a growing text transcript.

## Architecture

```text
raw world
(files / repo / terminal / docs / tests)
        |
        v
content-addressed semantic graph
(AST / symbols / types / packages / observations)
        |
        v
CognitiveState inside SolutionState
(hypotheses / evidence / open questions / next actions)
        |
        v
semantic policy backbone
        |
        v
ActionGraph
        |
        v
capability + approval gates
        |
        v
workspace action
        |
        v
content-addressed observation -> graph + CognitiveState
```

`SolutionState` remains the cognition/agency boundary. Side effects still require `ActionGraph` execution.

## Content identity

Logical concepts have stable semantic IDs, while concrete content is addressed by the existing BLAKE3 content hash contract.

Examples:

- `repo://project@commit`
- `file://project@commit/src/foo.py`
- `module://project@commit/pkg.foo`
- `symbol://project@commit/pkg.foo:Bar.parse`
- `type://python/Optional[str]`
- `package://python/pydantic`
- `observation://semantic-episode:.../3`

The SQLite graph stores immutable blobs separately from logical nodes and typed edges.

## Repository decomposition

The v0.1 indexer targets Python and extracts:

- repositories, files, modules
- classes, functions, methods
- imports and declared packages
- inheritance
- parameter and return annotations
- annotated assignments
- call references
- test files
- current git commit when available

Later versions can replace or augment AST inference with Pyright/mypy, language servers, runtime traces, coverage and additional language zplugs without changing the graph contract.

## Governed actions

Registered workspace actions:

- `repo.read`
- `repo.search`
- `repo.edit`
- `repo.replace`
- `git.status`
- `git.diff`
- `test.run`
- `process.exec`
- `docs.fetch`

`process.exec` accepts argv arrays only; shell command strings are intentionally unsupported. Writes/process execution require explicit capabilities, and risky actions are compiled with approvals. `docs.fetch` is HTTPS + domain-allowlist only and blocks configurable benchmark-contamination substrings.

## CLI

Index a repository:

```bash
lcfa-semantic index .
```

Inspect concepts:

```bash
lcfa-semantic concept "normalize Optional" --neighbors
```

Construct an evidence-linked cognitive state:

```bash
lcfa-semantic investigate "Optional inherited fields lose None during validation"
```

Inspect the next governed actions:

```bash
lcfa-semantic actions "Optional inherited fields lose None during validation"
```

Run the closed-loop semantic agent with an existing stochastic artifact:

```bash
lcfa-semantic agent "Fix the failing Optional inherited-field behavior" \
  --artifact artifacts/lcfa-stochastic-flow-mac \
  --max-steps 12 \
  --auto-approve \
  -o /tmp/semantic-episode.json
```

`--auto-approve` is intended only for isolated benchmark worktrees/containers.

## Training boundary

v0.1 deliberately keeps the semantic policy backbone separate from learned latent cognition. Each run records a trajectory containing:

- cognitive state
- hypothesis
- selected typed action
- action observation
- content hash
- termination state
- final patch

These trajectories are the substrate for the next LCFA latent objective:

```text
z_t + action_t + observation_t+1 -> z_t+1
```

with policy, value/verifier, evidence, contradiction and continue/stop heads. The latent model should not be promoted to control `SolutionState` until it beats the semantic-loop baseline under frozen benchmark gates.
