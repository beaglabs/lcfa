from __future__ import annotations

import json
from pathlib import Path

from lcfa.protocol import SolutionState
from lcfa.rwkv_semantic import RWKVSemanticBackbone
from lcfa.semantic_agent import SemanticWorkspaceAgent


class _Node:
    def __init__(self, path: str | None) -> None:
        self.metadata = {} if path is None else {"path": path}


class _Graph:
    def __init__(self, paths=None) -> None:
        self.paths = dict(paths or {})

    def get_node(self, node_id: str):
        if node_id not in self.paths:
            raise KeyError(node_id)
        return _Node(self.paths[node_id])


class _Policy:
    metadata = {"type": "fake"}

    def __init__(self) -> None:
        self.reset_calls = 0
        self.observe_calls = 0

    def reset(self, goal, solution):
        self.reset_calls += 1

    def observe(self, action, observation, solution):
        self.observe_calls += 1

    def choose(self, goal, solution, recent, step):
        return {
            "hypothesis": None,
            "action": {"name": "repo.read", "inputs": {"path": ""}},
            "final": False,
            "controller": {"action_confidence": 0.9, "stop_probability": 0.1, "value": 0.5},
        }


class _PinnedMutationPolicy(_Policy):
    def __init__(self, action_name: str) -> None:
        super().__init__()
        self.action_name = action_name

    def choose(self, goal, solution, recent, step):
        del goal, solution, recent, step
        inputs = {"path": "src/lcfa/right.py"}
        if self.action_name in {"repo.replace", "repo.edit"}:
            inputs.update({"old": "value = 1", "new": "value = 2"})
        return {
            "hypothesis": None,
            "action": {"name": self.action_name, "inputs": inputs},
            "final": False,
            "controller": {
                "pointer_index": 0,
                "mutation_target_pinned": True,
                "mutation_recovery_target": "src/lcfa/right.py",
                "mutation_recovery_locked_target": "src/lcfa/right.py",
            },
        }


def _pinned_payload() -> dict:
    return {
        "goal": "fix it",
        "cognition": {
            "candidate_queries": ["fix it"],
            "candidate_paths": ["src/lcfa/wrong.py", "src/lcfa/right.py"],
            "candidate_locations": [],
            "candidate_scores": [2.0, 1.0],
            "candidate_evidence": [[], []],
        },
        "recent_observations": [],
    }


def _sample(adapter: RWKVSemanticBackbone, payload: dict) -> dict:
    sample = adapter.sample(
        system_prompt="ignored",
        user_prompt=json.dumps(payload),
        branches=1,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=1,
    )[0]
    return json.loads(sample.text)


def test_repo_read_without_valid_candidate_falls_back_to_search() -> None:
    adapter = RWKVSemanticBackbone(_Policy(), _Graph())
    payload = {
        "goal": "fix it",
        "cognition": {"candidate_locations": []},
        "recent_observations": [],
    }
    choice = _sample(adapter, payload)
    assert choice["action"] == {"name": "repo.search", "inputs": {"query": "fix it"}}
    assert choice["controller"]["fallback_from"] == "repo.read"


def test_candidate_path_rejects_root_and_absolute_paths() -> None:
    graph = _Graph({"a": ".", "b": "/tmp/nope", "c": "src/lcfa/foo.py"})
    adapter = RWKVSemanticBackbone(_Policy(), graph)
    cognition = {"candidate_locations": ["a", "b", "c"]}
    assert adapter._candidate_path(cognition) == "src/lcfa/foo.py"


def test_runtime_pinned_read_path_is_not_rewritten_by_pointer() -> None:
    adapter = RWKVSemanticBackbone(_PinnedMutationPolicy("repo.read"), _Graph())
    choice = _sample(adapter, _pinned_payload())
    assert choice["action"] == {
        "name": "repo.read",
        "inputs": {"path": "src/lcfa/right.py"},
    }
    assert choice["controller"]["runtime_path_preserved"] is True
    assert choice["controller"]["retrieval_path"] == "src/lcfa/right.py"


def test_runtime_pinned_edit_path_is_not_rewritten_by_pointer() -> None:
    adapter = RWKVSemanticBackbone(_PinnedMutationPolicy("repo.replace"), _Graph())
    choice = _sample(adapter, _pinned_payload())
    assert choice["action"] == {
        "name": "repo.replace",
        "inputs": {
            "path": "src/lcfa/right.py",
            "old": "value = 1",
            "new": "value = 2",
        },
    }
    assert choice["controller"]["runtime_path_preserved"] is True
    assert choice["controller"]["retrieval_path"] == "src/lcfa/right.py"


def test_semantic_prompt_remains_valid_json_when_large(tmp_path: Path) -> None:
    class _PromptGraph:
        def get_node(self, _node_id):
            raise KeyError

    agent = SemanticWorkspaceAgent(_PromptGraph(), tmp_path, object(), max_context_chars=4000)
    solution = SolutionState(
        id="solution:test",
        plan_id="test",
        values={
            "cognition": {
                "active_concepts": [],
                "candidate_locations": [],
                "open_questions": ["x" * 12000],
            }
        },
    )
    prompt = agent._prompt("fix it", solution, [{"stdout": "z" * 20000}])
    payload = json.loads(prompt)
    assert payload["goal"] == "fix it"
    assert "...<truncated>" not in prompt[-32:]
