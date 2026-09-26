from __future__ import annotations

import json
from pathlib import Path

from lcfa.backbones import BackboneSample
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


def test_repo_read_without_valid_candidate_falls_back_to_search() -> None:
    adapter = RWKVSemanticBackbone(_Policy(), _Graph())
    payload = {
        "goal": "fix it",
        "cognition": {"candidate_locations": []},
        "recent_observations": [],
    }
    sample = adapter.sample(
        system_prompt="ignored",
        user_prompt=json.dumps(payload),
        branches=1,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=1,
    )[0]
    choice = json.loads(sample.text)
    assert choice["action"] == {"name": "repo.search", "inputs": {"query": "fix it"}}
    assert choice["controller"]["fallback_from"] == "repo.read"


def test_candidate_path_rejects_root_and_absolute_paths() -> None:
    graph = _Graph({"a": ".", "b": "/tmp/nope", "c": "src/lcfa/foo.py"})
    adapter = RWKVSemanticBackbone(_Policy(), graph)
    cognition = {"candidate_locations": ["a", "b", "c"]}
    assert adapter._candidate_path(cognition) == "src/lcfa/foo.py"


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
    assert len(prompt) <= 4000
