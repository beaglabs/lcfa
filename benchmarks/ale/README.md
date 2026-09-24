# Agents' Last Exam integration

`../suites/ale-lcfa-48.txt` is LCFA's frozen 48-task ALE system/milestone subset. It is evaluation-only.

ALE is intentionally executed by the upstream `ale_run` framework rather than by `lcfa-bench`: ALE provisions real Linux/Windows sandboxes, stages task data, runs an agent, grades hidden references, and records normalized trajectories/artifacts.

## Use the frozen list

Clone/install `rdi-berkeley/agents-last-exam`, then copy or reference `benchmarks/suites/ale-lcfa-48.txt` as the experiment's selected task list. In the ALE experiment YAML, set the `tasks` field to that task-list path. Keep ALE `auto_resume` enabled for expensive runs unless intentionally doing a clean rerun.

The upstream unlicensed task list is the source universe for this subset. Do not silently replace failed/inconvenient tasks: changing membership requires a new LCFA suite version.

## Results

Preserve ALE's native score, event log, trajectory, and produced artifacts. LCFA-specific instrumentation should additionally record latent steps, operator/action selections, Qwen/zplug calls, tool calls, state revisions, verifier decisions, and wall time so milestone reports can compare capability and compute, not only final task score.

The GitHub Job Summary renderer (`lcfa-bench-card`) currently consumes `lcfa.bench.report.v1` reports. ALE's native result importer is intentionally separate because ALE has richer task/trajectory semantics than an ordinary LCFA benchmark case.
