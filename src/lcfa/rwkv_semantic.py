"""Adapter from the recurrent RWKV controller to SemanticWorkspaceAgent.

SemanticWorkspaceAgent already accepts a stochastic-backbone-shaped policy.
This adapter preserves that API while making the actual action decision through
trained recurrent heads; JSON is only the compatibility envelope presented to
the existing agent loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backbones import BackboneSample
from .protocol import SolutionState
from .rwkv_controller import RWKVRecurrentPolicy
from .semantic_graph import SQLiteSemanticGraph


class RWKVSemanticBackbone:
    def __init__(self, policy: RWKVRecurrentPolicy, graph: SQLiteSemanticGraph) -> None:
        self.policy = policy
        self.graph = graph
        self.metadata = {
            "type": "rwkv7-semantic-controller",
            "controller": dict(policy.metadata),
        }
        self._goal: str | None = None
        self._last_action: Mapping[str, Any] | None = None
        self._last_observation_id: str | None = None
        self._step = 0

    def _solution(self, cognition: Mapping[str, Any]) -> SolutionState:
        return SolutionState(
            id="solution:rwkv-controller-proxy",
            plan_id="rwkv-controller",
            values={"cognition": dict(cognition)},
        )

    def _candidate_path(self, cognition: Mapping[str, Any]) -> str | None:
        raw = cognition.get("candidate_locations", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return None
        for concept_id in raw:
            try:
                node = self.graph.get_node(str(concept_id))
            except KeyError:
                continue
            path = node.metadata.get("path")
            if path:
                return str(path)
        return None

    def _ingest_delta(
        self,
        recent: Sequence[Mapping[str, Any]],
        solution: SolutionState,
    ) -> None:
        if not recent:
            return
        latest = recent[-1]
        observation_id = str(latest.get("content_hash") or latest.get("concept_id") or "")
        if observation_id and observation_id == self._last_observation_id:
            return
        self.policy.observe(self._last_action, latest, solution)
        self._last_observation_id = observation_id or json.dumps(latest, sort_keys=True, default=str)

    def sample(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        branches: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> tuple[BackboneSample, ...]:
        del system_prompt, temperature, top_p, max_new_tokens, seed
        try:
            payload = json.loads(user_prompt)
        except json.JSONDecodeError as exc:
            raise ValueError("semantic RWKV adapter expects SemanticWorkspaceAgent JSON prompt") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("semantic RWKV prompt must be an object")
        goal = str(payload.get("goal") or "")
        cognition = payload.get("cognition", {})
        if not isinstance(cognition, Mapping):
            cognition = {}
        recent = payload.get("recent_observations", ())
        if not isinstance(recent, Sequence) or isinstance(recent, (str, bytes)):
            recent = ()
        solution = self._solution(cognition)

        if self._goal != goal:
            self._goal = goal
            self._last_action = None
            self._last_observation_id = None
            self._step = 0
            self.policy.reset(goal, solution)
        else:
            self._ingest_delta(recent, solution)

        self._step += 1
        choice = dict(self.policy.choose(goal, solution, recent, self._step))
        controller = choice.get("controller")
        if isinstance(controller, Mapping) and controller.get("fallback_from") == "repo.read":
            path = self._candidate_path(cognition)
            if path:
                choice["action"] = {"name": "repo.read", "inputs": {"path": path}}
                choice["controller"] = {**dict(controller), "fallback_resolved": True}
        action = choice.get("action")
        self._last_action = dict(action) if isinstance(action, Mapping) else None
        text = json.dumps(choice, sort_keys=True, ensure_ascii=False)
        return tuple(BackboneSample(text, 0.0) for _ in range(max(1, int(branches))))


def load_rwkv_semantic_backbone(
    controller_dir: str | Path,
    graph: SQLiteSemanticGraph,
    *,
    model_id: str | None = None,
    device: str | None = None,
    dtype: str = "bfloat16",
) -> RWKVSemanticBackbone:
    root = Path(controller_dir)
    manifest_path = root / "controller.json" if root.is_dir() else root
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("controller manifest must be a JSON object")
    resolved_root = manifest_path.parent
    weights = str(raw.get("weights") or "heads.safetensors")
    resolved_model = str(model_id or raw.get("model_id") or "")
    if not resolved_model:
        raise ValueError("controller manifest requires model_id")
    policy = RWKVRecurrentPolicy(
        resolved_model,
        heads_path=resolved_root / weights,
        device=device,
        dtype=dtype,
        action_names=tuple(raw.get("action_vocab") or ()),
    )
    return RWKVSemanticBackbone(policy, graph)


__all__ = ["RWKVSemanticBackbone", "load_rwkv_semantic_backbone"]
