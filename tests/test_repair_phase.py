from __future__ import annotations

from lcfa.repair_phase import (
    DISCOVER,
    DONE,
    GROUND,
    REPAIR,
    VERIFY,
    RepairPhaseState,
    advance_repair_phase,
    phase_action_policy,
    stop_allowed,
    verifier_outcome,
)
from lcfa.recurrent_transitions import ACTION_VOCAB


def _action(name: str):
    return {"name": name, "inputs": {}}


def test_discovery_blocks_premature_verification_and_stop() -> None:
    state = RepairPhaseState()
    allowed, _biases = phase_action_policy(state, ACTION_VOCAB, has_target=True)

    assert state.phase == DISCOVER
    assert "repo.search" in allowed
    assert "repo.read" in allowed
    assert "verify.run" not in allowed
    assert "test.run" not in allowed
    assert "stop" not in allowed
    assert not stop_allowed(state)


def test_evidence_advances_to_ground_and_eventually_hard_locks_mutation() -> None:
    state = RepairPhaseState()
    state = advance_repair_phase(state, _action("repo.search"), {"ok": True})
    state = advance_repair_phase(state, _action("repo.read"), {"ok": True})

    assert state.phase == GROUND

    state = advance_repair_phase(state, _action("repo.read"), {"ok": True})
    state = advance_repair_phase(state, _action("repo.search"), {"ok": True})
    allowed, biases = phase_action_policy(state, ACTION_VOCAB, has_target=True)

    assert set(allowed) == {"repo.edit", "repo.replace"}
    assert biases["repo.edit"] > 0
    assert biases["repo.replace"] > 0


def test_mutation_requires_verification_before_stop() -> None:
    state = RepairPhaseState(phase=GROUND, evidence_actions=4, reads=2)
    state = advance_repair_phase(state, _action("repo.edit"), {"edit": {"ok": True}})

    assert state.phase == VERIFY
    assert state.mutations == 1
    assert not stop_allowed(state)

    allowed, _biases = phase_action_policy(state, ACTION_VOCAB, has_target=True)
    assert "verify.run" in allowed
    assert "test.run" in allowed
    assert "repo.search" not in allowed
    assert "repo.read" not in allowed
    assert "stop" not in allowed


def test_successful_verifier_unlocks_done_and_stop() -> None:
    state = RepairPhaseState(phase=VERIFY, evidence_actions=4, reads=2, mutations=1)
    observation = {"process": {"returncode": 0, "stdout": "3 passed"}}

    assert verifier_outcome(_action("verify.run"), observation) is True
    state = advance_repair_phase(state, _action("verify.run"), observation)

    assert state.phase == DONE
    assert state.verifier_passed is True
    assert stop_allowed(state)

    allowed, _biases = phase_action_policy(state, ACTION_VOCAB, has_target=True)
    assert allowed == ("stop",)


def test_failed_verifier_reopens_repair_phase() -> None:
    state = RepairPhaseState(phase=VERIFY, evidence_actions=4, reads=2, mutations=1)
    observation = {"process": {"returncode": 1, "stderr": "failure"}}

    assert verifier_outcome(_action("test.run"), observation) is False
    state = advance_repair_phase(state, _action("test.run"), observation)

    assert state.phase == REPAIR
    assert state.verifier_passed is False
    assert not stop_allowed(state)

    allowed, _biases = phase_action_policy(state, ACTION_VOCAB, has_target=True)
    assert "repo.edit" in allowed
    assert "repo.replace" in allowed
    assert "verify.run" not in allowed
    assert "stop" not in allowed


def test_unknown_verifier_result_stays_in_verify_phase() -> None:
    state = RepairPhaseState(phase=VERIFY, mutations=1)
    state = advance_repair_phase(
        state,
        _action("verify.run"),
        {"process": {"stdout": "running"}},
    )

    assert state.phase == VERIFY
    assert state.verifications == 1
    assert state.verifier_passed is False
