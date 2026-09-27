from __future__ import annotations

import json

from lcfa.hybrid_semantic_controller import pointer_runtime_mode
from lcfa.phase_runtime import (
    PhaseMutationRuntimePolicy,
    attach_controller_trace,
    collect_file_evidence,
    controller_target_path,
    validate_replace_inputs,
)


def test_collect_file_evidence_extracts_nested_repo_read_text() -> None:
    observation = {
        "results": {
            "node": {
                "path": "src/lcfa/example.py",
                "text": "value = 1\n",
            }
        },
        "observations": {
            "file": {
                "path": "src/lcfa/example.py",
                "text": "value = 1\n",
            }
        },
    }
    assert collect_file_evidence(observation) == {
        "src/lcfa/example.py": "value = 1\n"
    }


def test_controller_target_path_prefers_repair_ir() -> None:
    controller = {
        "repair_ir": {"target_path": "src/lcfa/target.py"},
        "retrieval_path": "src/lcfa/other.py",
    }
    assert controller_target_path(controller) == "src/lcfa/target.py"


def test_validate_replace_inputs_requires_unique_grounded_span() -> None:
    source = "alpha = 1\nbeta = 2\n"
    valid = validate_replace_inputs(
        {"old": "beta = 2", "new": "beta = 3"},
        target_path="x.py",
        source_text=source,
    )
    assert valid == {
        "path": "x.py",
        "old": "beta = 2",
        "new": "beta = 3",
    }
    assert validate_replace_inputs(
        {"old": "missing", "new": "changed"},
        target_path="x.py",
        source_text=source,
    ) is None
    assert validate_replace_inputs(
        {"old": "alpha", "new": "alpha"},
        target_path="x.py",
        source_text=source,
    ) is None


def test_pointer_runtime_mode_falls_back_when_learned_pointer_regresses() -> None:
    assert pointer_runtime_mode(0.125, 0.25) == "retrieval-prior"
    assert pointer_runtime_mode(0.5, 0.25) == "semantic"
    assert pointer_runtime_mode(None, 0.25) == "semantic"


def test_attach_controller_trace_persists_step_diagnostics(tmp_path) -> None:
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps({
            "schema_version": "lcfa.semantic-trajectory.v1",
            "steps": [
                {"index": 1, "action": {"name": "repo.search"}},
                {"index": 2, "action": {"name": "repo.read"}},
            ],
        }),
        encoding="utf-8",
    )
    count = attach_controller_trace(
        path,
        [
            {"repair_phase": "discover"},
            {"repair_phase": "ground", "phase_overrode_action": True},
        ],
    )
    assert count == 2
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["steps"][0]["controller"]["repair_phase"] == "discover"
    assert raw["steps"][1]["controller"]["repair_phase"] == "ground"
    assert raw["steps"][1]["controller"]["phase_overrode_action"] is True


class _FakeInnerPolicy:
    metadata = {"type": "fake"}

    def __init__(self) -> None:
        self.observed = []
        self.calls = 0

    def reset(self, goal, solution) -> None:
        del goal, solution

    def observe(self, action, observation, solution) -> None:
        del solution
        self.observed.append((action, observation))

    def choose(self, goal, solution, recent, step):
        del goal, solution, recent, step
        self.calls += 1
        target = (
            "src/lcfa/example.py"
            if self.calls == 1
            else "src/lcfa/drifted.py"
        )
        return {
            "hypothesis": None,
            "action": {"name": "repo.search", "inputs": {"query": "fallback"}},
            "final": False,
            "controller": {
                "repair_phase": "ground",
                "phase_selected_action": "repo.edit",
                "fallback_from": "repo.edit",
                "repair_ir": {"target_path": target},
            },
        }


def test_phase_runtime_replaces_silent_search_fallback_with_target_read() -> None:
    policy = PhaseMutationRuntimePolicy(_FakeInnerPolicy())
    result = policy.choose("fix it", object(), (), 1)
    assert result["action"] == {
        "name": "repo.read",
        "inputs": {"path": "src/lcfa/example.py"},
    }
    controller = result["controller"]
    assert controller["mutation_recovery_attempted"] is True
    assert controller["mutation_recovery_status"] == "needs-target-evidence"
    assert controller["mutation_target_pinned"] is True
    trace = policy.consume_controller_trace()
    assert len(trace) == 1
    assert trace[0]["repair_phase"] == "ground"


def test_phase_runtime_keeps_target_pinned_when_pointer_drifts() -> None:
    policy = PhaseMutationRuntimePolicy(_FakeInnerPolicy())
    first = policy.choose("fix it", object(), (), 1)
    assert first["action"]["inputs"]["path"] == "src/lcfa/example.py"

    policy.observe(
        first["action"],
        {
            "observations": {
                "file": {
                    "path": "src/lcfa/example.py",
                    "text": "value = 1\n",
                }
            }
        },
        object(),
    )

    second = policy.choose("fix it", object(), (), 2)
    controller = second["controller"]
    assert controller["mutation_recovery_proposed_target"] == "src/lcfa/drifted.py"
    assert controller["mutation_recovery_target"] == "src/lcfa/example.py"
    assert controller["mutation_target_changed_by_pointer"] is True
    assert second["action"] == {
        "name": "repo.read",
        "inputs": {"path": "src/lcfa/example.py"},
    }
