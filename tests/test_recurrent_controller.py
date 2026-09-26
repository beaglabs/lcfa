from __future__ import annotations

import json
from pathlib import Path

from lcfa import (
    ACTION_VOCAB,
    PythonRepoIndexer,
    RWKVSemanticBackbone,
    SQLiteSemanticGraph,
    SolutionState,
    episode_to_transitions,
    load_transitions,
    prepare_transition_file,
)


def _episode(*, success=None):
    value = {
        "id": "semantic-episode:test",
        "goal": "Fix Optional name normalization",
        "steps": [
            {
                "index": 1,
                "solution_id": "solution:1",
                "hypothesis": {"claim": "Inspect normalize_name", "confidence": 0.8},
                "action": {"name": "repo.read", "inputs": {"path": "pkg/models.py"}},
                "observation": {"concept_id": "observation://1", "content_hash": "b3:one"},
                "terminal": False,
            },
            {
                "index": 2,
                "solution_id": "solution:2",
                "hypothesis": {"claim": "Behavior is correct", "confidence": 0.95},
                "action": None,
                "observation": {},
                "terminal": True,
            },
        ],
        "final_solution_id": "solution:2",
        "patch": "",
        "schema_version": "lcfa.semantic-trajectory.v1",
    }
    if success is not None:
        value["success"] = success
    return value


def test_episode_to_recurrent_transitions_uses_explicit_targets_only() -> None:
    rows = episode_to_transitions(_episode())
    assert len(rows) == 2
    assert rows[0].target_action == "repo.read"
    assert rows[0].stop_target is False
    assert rows[1].target_action == "stop"
    assert rows[1].stop_target is True
    assert rows[0].value_target is None

    graded = episode_to_transitions(_episode(success=True))
    assert all(row.value_target == 1.0 for row in graded)
    assert set(row.target_action for row in graded) <= set(ACTION_VOCAB)


def test_prepare_transition_file_round_trips_jsonl(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode.json"
    output = tmp_path / "transitions.jsonl"
    episode_path.write_text(json.dumps(_episode(success=False)), encoding="utf-8")
    count = prepare_transition_file([episode_path], output)
    assert count == 2
    rows = load_transitions(output)
    assert len(rows) == 2
    assert rows[0].value_target == 0.0
    assert rows[-1].target_action == "stop"


class _FakeRecurrentPolicy:
    metadata = {"type": "fake-rwkv"}

    def __init__(self) -> None:
        self.reset_calls = 0
        self.observe_calls = 0
        self.choose_calls = 0

    def reset(self, goal: str, solution: SolutionState) -> None:
        assert goal
        assert "cognition" in solution.values
        self.reset_calls += 1

    def observe(self, action, observation, solution: SolutionState) -> None:
        assert observation
        assert "cognition" in solution.values
        self.observe_calls += 1

    def choose(self, goal, solution: SolutionState, recent, step):
        del recent, step
        self.choose_calls += 1
        if self.choose_calls == 1:
            return {
                "hypothesis": None,
                "action": {"name": "repo.search", "inputs": {"query": goal}},
                "final": False,
                "controller": {"fallback_from": "repo.read", "value": 0.4},
            }
        return {
            "hypothesis": None,
            "action": None,
            "final": True,
            "controller": {"value": 0.9},
        }


def _repo(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "models.py").write_text(
        "class User:\n    def normalize_name(self, value):\n        return value\n",
        encoding="utf-8",
    )


def test_rwkv_semantic_backbone_preserves_recurrent_state_and_resolves_repo_read(tmp_path: Path) -> None:
    _repo(tmp_path)
    with SQLiteSemanticGraph(tmp_path / ".lcfa" / "semantic.db") as graph:
        PythonRepoIndexer(graph, tmp_path).index()
        candidate = next(node for node in graph.search("normalize_name", limit=10) if node.kind == "method")
        policy = _FakeRecurrentPolicy()
        backbone = RWKVSemanticBackbone(policy, graph)
        payload = {
            "goal": "inspect normalize_name",
            "cognition": {
                "candidate_locations": [candidate.id],
                "active_concepts": [candidate.id],
            },
            "recent_observations": [],
        }
        first = backbone.sample(
            system_prompt="ignored",
            user_prompt=json.dumps(payload),
            branches=1,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=32,
        )
        choice = json.loads(first[0].text)
        assert choice["action"] == {"name": "repo.read", "inputs": {"path": "pkg/models.py"}}
        assert policy.reset_calls == 1
        assert policy.observe_calls == 0

        payload["recent_observations"] = [{"content_hash": "b3:obs", "result": "read"}]
        second = backbone.sample(
            system_prompt="ignored",
            user_prompt=json.dumps(payload),
            branches=1,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=32,
        )
        assert json.loads(second[0].text)["final"] is True
        assert policy.reset_calls == 1
        assert policy.observe_calls == 1
