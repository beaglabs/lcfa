"""Closed-loop semantic workspace agent and trajectory capture."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .artifact import load_artifact_reasoner
from .cognitive import CognitiveState, Hypothesis, SemanticInvestigator
from .engine import LCFA
from .protocol import ActionRun, ExecutionContext, SolutionState
from .semantic_graph import SQLiteSemanticGraph
from .state import content_hash
from .workspace_actions import compile_cognitive_actions, register_workspace_actions

SEMANTIC_AGENT_TRAJECTORY_FORMAT = "lcfa.semantic-trajectory.v1"


@dataclass(frozen=True, slots=True)
class SemanticAgentStep:
    index: int
    solution_id: str
    hypothesis: Mapping[str, Any] | None
    action: Mapping[str, Any] | None
    observation: Mapping[str, Any]
    terminal: bool


@dataclass(frozen=True, slots=True)
class SemanticAgentEpisode:
    id: str
    goal: str
    steps: tuple[SemanticAgentStep, ...]
    final_solution_id: str
    patch: str
    schema_version: str = SEMANTIC_AGENT_TRAJECTORY_FORMAT

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def _extract_json(text: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(0, limit - 20)] + "...<truncated>"


class SemanticWorkspaceAgent:
    """Uses a semantic graph as memory and a backbone only for structured next-action selection."""

    SYSTEM_PROMPT = '''You are the policy layer of an LCFA coding/terminal agent.
The semantic graph and observations are authoritative. Return ONLY one JSON object:
{"hypothesis":{"claim":string,"confidence":number},"action":{"name":string,"inputs":object},"final":boolean}
Allowed actions: repo.read, repo.search, repo.replace, repo.edit, test.run, git.status, git.diff, process.exec, docs.fetch.
Prefer semantic/repository inspection and targeted tests over broad shell commands.
Use repo.replace for small edits: {"path":...,"old":...,"new":...}.
Use process.exec only when no typed action is appropriate; argv must be an array and shell strings are forbidden.
Set final=true only when the requested outcome is verified. When final=true, action may be null.
confidence MUST be a JSON number from 0.0 to 1.0.'''

    def __init__(
        self,
        graph: SQLiteSemanticGraph,
        workspace_root: str | Path,
        backbone: Any,
        *,
        max_steps: int = 12,
        max_context_chars: int = 16000,
    ) -> None:
        self.graph = graph
        self.workspace_root = Path(workspace_root).resolve()
        self.backbone = backbone
        self.max_steps = max(1, int(max_steps))
        self.max_context_chars = max(4000, int(max_context_chars))
        actions = register_workspace_actions()
        self.executor = LCFA(actions=actions).agentic

    @classmethod
    def from_artifact(
        cls,
        graph: SQLiteSemanticGraph,
        workspace_root: str | Path,
        artifact: str | Path,
        *,
        max_steps: int = 12,
        max_context_chars: int = 16000,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> "SemanticWorkspaceAgent":
        reasoner = load_artifact_reasoner(artifact, base_engine=LCFA(), runtime_options=dict(runtime_options or {}))
        backbone = getattr(reasoner, "backbone", None)
        if backbone is None:
            raise TypeError("semantic agent requires an artifact exposing a stochastic backbone")
        return cls(graph, workspace_root, backbone, max_steps=max_steps, max_context_chars=max_context_chars)

    def _concept_context(self, solution: SolutionState) -> list[Mapping[str, Any]]:
        cognition = solution.values.get("cognition", {})
        ids = cognition.get("active_concepts", ()) if isinstance(cognition, Mapping) else ()
        out = []
        for node_id in list(ids)[:12]:
            try:
                node = self.graph.get_node(str(node_id))
                raw = self.graph.get_blob(node.content.content_hash)
            except KeyError:
                continue
            text = raw.decode("utf-8", errors="replace") if node.content.media_type.startswith("text/") else ""
            out.append({
                "id": node.id, "kind": node.kind, "label": node.label,
                "metadata": dict(node.metadata), "content_hash": node.content.content_hash,
                "content": _clip(text, 1800) if text else None,
            })
        return out

    def _prompt(self, goal: str, solution: SolutionState, recent: list[Mapping[str, Any]]) -> str:
        cognition = solution.values.get("cognition", {})
        payload = {
            "goal": goal,
            "cognition": cognition,
            "concepts": self._concept_context(solution),
            "recent_observations": recent[-4:],
        }
        return _clip(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), self.max_context_chars)

    def _choose(self, goal: str, solution: SolutionState, recent: list[Mapping[str, Any]], step: int) -> Mapping[str, Any]:
        samples = self.backbone.sample(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=self._prompt(goal, solution, recent),
            branches=1,
            temperature=0.2,
            top_p=0.95,
            max_new_tokens=768,
            seed=20260925 + step,
        )
        if not samples:
            return {"hypothesis": None, "action": {"name": "repo.search", "inputs": {"query": goal}}, "final": False}
        parsed = _extract_json(samples[0].text)
        if parsed is None:
            return {"hypothesis": None, "action": {"name": "repo.search", "inputs": {"query": goal}}, "final": False}
        return parsed

    def _apply_choice(self, solution: SolutionState, choice: Mapping[str, Any]) -> SolutionState:
        cognition_raw = solution.values.get("cognition", {})
        cognition = dict(cognition_raw) if isinstance(cognition_raw, Mapping) else {}
        hypotheses = list(cognition.get("hypotheses", ()))
        hyp_raw = choice.get("hypothesis")
        if isinstance(hyp_raw, Mapping) and hyp_raw.get("claim"):
            confidence = hyp_raw.get("confidence", 0.5)
            try:
                confidence_f = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence_f = 0.5
            hypotheses.append({
                "id": f"H{len(hypotheses) + 1}", "claim": str(hyp_raw["claim"]),
                "supports": list(cognition.get("active_concepts", ()))[:6], "conflicts": [],
                "confidence": confidence_f, "status": "open",
            })
        action_raw = choice.get("action")
        next_actions = []
        if isinstance(action_raw, Mapping) and action_raw.get("name"):
            next_actions.append({
                "action": str(action_raw["name"]),
                "inputs": dict(action_raw.get("inputs", {})) if isinstance(action_raw.get("inputs", {}), Mapping) else {},
                "reason": (str(hyp_raw.get("claim")) if isinstance(hyp_raw, Mapping) else "semantic policy action"),
            })
        cognition["hypotheses"] = hypotheses
        cognition["next_actions"] = next_actions
        cognition["terminal"] = bool(choice.get("final", False))
        digest = content_hash(cognition)
        return replace(
            solution,
            id=f"solution:semantic:{digest.removeprefix('b3:')[:24]}",
            values={**solution.values, "cognition": cognition},
        )

    def _ingest_observation(self, episode_id: str, step: int, solution: SolutionState, run: ActionRun) -> Mapping[str, Any]:
        observation = {
            "graph_id": run.graph_id,
            "results": {key: value.value for key, value in run.results.items()},
            "observations": dict(run.observations),
        }
        node_id = f"observation://{episode_id}/{step}"
        node = self.graph.put_node(
            node_id, "observation", f"agent step {step}", observation,
            metadata={"episode_id": episode_id, "step": step, "solution_id": solution.id},
        )
        cognition = solution.values.get("cognition", {})
        if isinstance(cognition, Mapping):
            for concept_id in list(cognition.get("active_concepts", ()))[:8]:
                self.graph.add_edge(node.id, "observed_for", str(concept_id))
        return {"concept_id": node.id, "content_hash": node.content.content_hash, **observation}

    def _context(self, *, auto_approve: bool, action_graph) -> ExecutionContext:
        approvals = frozenset(
            node.approval_key for node in action_graph.nodes
            if auto_approve and node.requires_approval and node.approval_key
        )
        return ExecutionContext(
            capabilities=frozenset({"workspace.read", "workspace.write", "process.exec", "network.docs"}),
            approvals=approvals,
            metadata={
                "workspace_root": str(self.workspace_root),
                "docs_allow_domains": (
                    "docs.python.org", "readthedocs.io", "readthedocs.org", "pypi.org",
                    "numpy.org", "pandas.pydata.org", "docs.djangoproject.com", "docs.pydantic.dev",
                ),
            },
        )

    def run(self, goal: str, *, auto_approve: bool = False) -> SemanticAgentEpisode:
        episode_id = f"semantic-episode:{uuid4()}"
        solution = SemanticInvestigator(self.graph).investigate(goal)
        recent: list[Mapping[str, Any]] = []
        steps: list[SemanticAgentStep] = []
        for index in range(1, self.max_steps + 1):
            choice = self._choose(goal, solution, recent, index)
            solution = self._apply_choice(solution, choice)
            terminal = bool(choice.get("final", False))
            action_raw = choice.get("action")
            if terminal and not isinstance(action_raw, Mapping):
                steps.append(SemanticAgentStep(index, solution.id, choice.get("hypothesis") if isinstance(choice.get("hypothesis"), Mapping) else None, None, {}, True))
                break
            graph = compile_cognitive_actions(solution)
            if not graph.nodes:
                steps.append(SemanticAgentStep(index, solution.id, choice.get("hypothesis") if isinstance(choice.get("hypothesis"), Mapping) else None, None, {}, terminal))
                if terminal:
                    break
                continue
            run = self.executor.execute(graph, solution, self._context(auto_approve=auto_approve, action_graph=graph))
            observation = self._ingest_observation(episode_id, index, solution, run)
            recent.append(observation)
            cognition = dict(solution.values.get("cognition", {}))
            obs = list(cognition.get("observations", ()))
            obs.append({"concept_id": observation["concept_id"], "content_hash": observation["content_hash"]})
            cognition["observations"] = obs
            cognition["active_concepts"] = list(dict.fromkeys([*cognition.get("active_concepts", ()), observation["concept_id"]]))
            solution = replace(solution, values={**solution.values, "cognition": cognition})
            steps.append(SemanticAgentStep(
                index=index, solution_id=solution.id,
                hypothesis=(choice.get("hypothesis") if isinstance(choice.get("hypothesis"), Mapping) else None),
                action=(dict(action_raw) if isinstance(action_raw, Mapping) else None),
                observation=observation, terminal=terminal,
            ))
            if terminal:
                break

        patch_result = self.executor.execute(
            ActionGraph(id=f"patch:{uuid4()}", source_solution_id=solution.id, nodes=(
                ActionNode(id="git-diff", action="git.diff", required_capabilities=("workspace.read",)),
            )),
            solution,
            ExecutionContext(capabilities=frozenset({"workspace.read"}), metadata={"workspace_root": str(self.workspace_root)}),
        )
        patch_value = patch_result.results["git-diff"].value
        patch = str(patch_value.get("stdout", "")) if isinstance(patch_value, Mapping) else ""
        return SemanticAgentEpisode(episode_id, goal, tuple(steps), solution.id, patch)


# Imports kept at the bottom to avoid obscuring the cognitive types above.
from .protocol import ActionGraph, ActionNode  # noqa: E402

__all__ = ["SEMANTIC_AGENT_TRAJECTORY_FORMAT", "SemanticAgentStep", "SemanticAgentEpisode", "SemanticWorkspaceAgent"]
