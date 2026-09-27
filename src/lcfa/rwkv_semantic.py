"""Adapter from the recurrent RWKV controller to SemanticWorkspaceAgent.

The semantic agent keeps a structured compatibility envelope while every
action-type decision comes from RWKV recurrent state. Retrieval-backed search,
read, and edit paths resolve against the same ranked candidates present in the
training event stream.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backbones import BackboneSample
from .protocol import SolutionState
from .repo_retrieval import build_retrieval_context, extract_retrieval_queries
from .rwkv_controller import RWKVRecurrentPolicy, load_rwkv_policy
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
        self._read_paths: set[str] = set()

    def _solution(self, cognition: Mapping[str, Any]) -> SolutionState:
        return SolutionState(
            id="solution:rwkv-controller-proxy",
            plan_id="rwkv-controller",
            values={"cognition": dict(cognition)},
        )

    @staticmethod
    def _sequence(cognition: Mapping[str, Any], key: str) -> tuple[str, ...]:
        raw = cognition.get(key, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        return tuple(str(item) for item in raw if str(item))

    @staticmethod
    def _valid_path(value: Any) -> str | None:
        rendered = str(value or "").strip()
        if not rendered or rendered in {".", "/"} or Path(rendered).is_absolute():
            return None
        return rendered

    def _paths_from_locations(self, cognition: Mapping[str, Any]) -> tuple[str, ...]:
        paths: list[str] = []
        for concept_id in self._sequence(cognition, "candidate_locations"):
            try:
                node = self.graph.get_node(concept_id)
            except (KeyError, AttributeError):
                continue
            path = self._valid_path(node.metadata.get("path"))
            if path and path not in paths:
                paths.append(path)
        return tuple(paths)

    @staticmethod
    def _has_aligned_priors(cognition: Mapping[str, Any]) -> bool:
        paths = cognition.get("candidate_paths", ())
        scores = cognition.get("candidate_scores", ())
        evidence = cognition.get("candidate_evidence", ())
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
            return False
        return (
            isinstance(scores, Sequence)
            and not isinstance(scores, (str, bytes))
            and isinstance(evidence, Sequence)
            and not isinstance(evidence, (str, bytes))
            and len(scores) == len(paths)
            and len(evidence) == len(paths)
        )

    @staticmethod
    def _apply_retrieval(updated: dict[str, Any], retrieval: Any) -> dict[str, Any]:
        updated["candidate_queries"] = list(retrieval.queries)
        updated["candidate_paths"] = list(retrieval.candidate_paths)
        updated["candidate_locations"] = list(retrieval.candidate_ids)
        updated["candidate_scores"] = [float(item.score) for item in retrieval.candidates]
        updated["candidate_evidence"] = [list(item.evidence) for item in retrieval.candidates]
        return updated

    def _ensure_retrieval(self, goal: str, cognition: Mapping[str, Any]) -> Mapping[str, Any]:
        updated = dict(cognition)
        queries = self._sequence(updated, "candidate_queries")
        paths = self._sequence(updated, "candidate_paths")
        if queries and paths and self._has_aligned_priors(updated):
            return updated

        # Enrich existing path candidates with the same deterministic retriever
        # used for training. Preserve the visible path order when possible.
        try:
            retrieval = build_retrieval_context(self.graph, goal)
        except (AttributeError, TypeError):
            retrieval = None
        if retrieval is not None:
            if paths:
                by_path = {candidate.path: candidate for candidate in retrieval.candidates}
                aligned = [by_path.get(path) for path in paths]
                if all(candidate is not None for candidate in aligned):
                    updated["candidate_queries"] = list(queries or retrieval.queries)
                    updated["candidate_paths"] = list(paths)
                    updated["candidate_locations"] = [
                        candidate.concept_id for candidate in aligned if candidate is not None
                    ]
                    updated["candidate_scores"] = [
                        float(candidate.score) for candidate in aligned if candidate is not None
                    ]
                    updated["candidate_evidence"] = [
                        list(candidate.evidence) for candidate in aligned if candidate is not None
                    ]
                    return updated
            return self._apply_retrieval(updated, retrieval)

        location_paths = self._paths_from_locations(updated)
        if location_paths:
            updated["candidate_paths"] = list(location_paths)
            if not queries:
                extracted = extract_retrieval_queries(goal)
                updated["candidate_queries"] = list(extracted or (goal,))
            updated.setdefault("candidate_scores", [-float(i) for i in range(len(location_paths))])
            updated.setdefault("candidate_evidence", [[] for _ in location_paths])
            return updated

        extracted = extract_retrieval_queries(goal)
        updated["candidate_queries"] = list(extracted or (goal,))
        updated.setdefault("candidate_paths", [])
        updated.setdefault("candidate_locations", [])
        updated.setdefault("candidate_scores", [])
        updated.setdefault("candidate_evidence", [])
        return updated

    def _candidate_path(
        self,
        cognition: Mapping[str, Any],
        pointer: int | None = None,
        *,
        prefer_unread: bool = False,
    ) -> str | None:
        paths = tuple(
            path
            for path in (
                self._valid_path(item)
                for item in self._sequence(cognition, "candidate_paths")
            )
            if path
        )
        if not paths:
            paths = self._paths_from_locations(cognition)
        if not paths:
            return None
        start = int(pointer or 0) % len(paths)
        ordered = [paths[(start + offset) % len(paths)] for offset in range(len(paths))]
        if prefer_unread:
            for path in ordered:
                if path not in self._read_paths:
                    return path
        return ordered[0]

    def _candidate_query(self, cognition: Mapping[str, Any], pointer: int | None) -> str | None:
        queries = self._sequence(cognition, "candidate_queries")
        if not queries:
            return None
        return queries[int(pointer or 0) % len(queries)]

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
        self._last_observation_id = observation_id or json.dumps(
            latest, sort_keys=True, default=str
        )

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
            raise ValueError(
                "semantic RWKV adapter received invalid JSON; SemanticWorkspaceAgent must compact fields before serialization"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError("semantic RWKV prompt must be an object")
        goal = str(payload.get("goal") or "")
        cognition_raw = payload.get("cognition", {})
        cognition = dict(cognition_raw) if isinstance(cognition_raw, Mapping) else {}
        cognition = dict(self._ensure_retrieval(goal, cognition))
        recent = payload.get("recent_observations", ())
        if not isinstance(recent, Sequence) or isinstance(recent, (str, bytes)):
            recent = ()
        solution = self._solution(cognition)

        if self._goal != goal:
            self._goal = goal
            self._last_action = None
            self._last_observation_id = None
            self._step = 0
            self._read_paths = set()
            self.policy.reset(goal, solution)
        else:
            self._ingest_delta(recent, solution)

        self._step += 1
        choice = dict(self.policy.choose(goal, solution, recent, self._step))
        controller_raw = choice.get("controller")
        controller = dict(controller_raw) if isinstance(controller_raw, Mapping) else {}
        pointer_raw = controller.get("pointer_index")
        pointer = int(pointer_raw) if isinstance(pointer_raw, int) else None
        action_raw = choice.get("action")
        action = dict(action_raw) if isinstance(action_raw, Mapping) else None

        if action is not None:
            name = str(action.get("name") or "")
            inputs = (
                dict(action.get("inputs", {}))
                if isinstance(action.get("inputs"), Mapping)
                else {}
            )
            if controller.get("fallback_from") == "repo.read":
                name = "repo.read"
            if name == "repo.search":
                query = self._candidate_query(cognition, pointer)
                if query:
                    inputs["query"] = query
                    controller["retrieval_query"] = query
            elif name == "repo.read":
                path = self._candidate_path(cognition, pointer, prefer_unread=True)
                if path:
                    inputs = {"path": path}
                    self._read_paths.add(path)
                    controller["retrieval_path"] = path
                else:
                    query = self._candidate_query(cognition, pointer) or goal
                    name = "repo.search"
                    inputs = {"query": query}
                    controller["fallback_from"] = "repo.read"
                    controller["fallback_reason"] = "no valid candidate file path"
            elif name in {"repo.replace", "repo.edit"}:
                path = self._candidate_path(cognition, pointer)
                if path:
                    inputs["path"] = path
                    controller["retrieval_path"] = path
            action = {"name": name, "inputs": inputs}
            choice["action"] = action
            choice["controller"] = controller

        self._last_action = dict(action) if isinstance(action, Mapping) else None
        text = json.dumps(choice, sort_keys=True, ensure_ascii=False)
        return tuple(
            BackboneSample(text, 0.0) for _ in range(max(1, int(branches)))
        )


def load_rwkv_semantic_backbone(
    controller_dir: str | Path,
    graph: SQLiteSemanticGraph,
    *,
    model_id: str | None = None,
    device: str | None = "auto",
    dtype: str = "auto",
) -> RWKVSemanticBackbone:
    return RWKVSemanticBackbone(
        load_rwkv_policy(
            controller_dir,
            model_id=model_id,
            device=device,
            dtype=dtype,
        ),
        graph,
    )


__all__ = ["RWKVSemanticBackbone", "load_rwkv_semantic_backbone"]
