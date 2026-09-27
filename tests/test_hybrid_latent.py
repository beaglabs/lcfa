from __future__ import annotations

import pytest

from lcfa.hybrid_latent import (
    HybridLatentConfig,
    hybrid_event,
    latent_slot_names,
    make_hybrid_core,
    repair_plan_from_state,
    repair_prompt,
    structured_runtime_feedback,
    supervised_plan_targets,
)


def test_hybrid_config_normalizes_reasoning_shape() -> None:
    config = HybridLatentConfig(
        latent_dim=4,
        slots=2,
        min_reasoning_steps=3,
        max_reasoning_steps=1,
        convergence_tolerance=-1,
        intent_anchor_strength=2.0,
    ).normalized()
    assert config.latent_dim == 16
    assert config.slots == 9
    assert config.min_reasoning_steps == 3
    assert config.max_reasoning_steps == 3
    assert config.convergence_tolerance == 0.0
    assert config.intent_anchor_strength == 1.0


def test_typed_latent_slots_have_stable_intent_semantics() -> None:
    assert latent_slot_names(9) == (
        "user_intent",
        "problem",
        "hypothesis_primary",
        "hypothesis_alternative",
        "evidence",
        "target",
        "repair_intent",
        "constraints",
        "verifier_expectation",
    )
    assert latent_slot_names(11)[-2:] == ("scratch_0", "scratch_1")


def test_verifier_output_becomes_structured_feedback() -> None:
    action = {"name": "verify.run", "inputs": {}}
    observation = {
        "results": {
            "verify": {
                "exit_code": 1,
                "stdout": "1 failed, 4 passed",
                "stderr": "AssertionError: expected fixed",
            }
        }
    }
    feedback = structured_runtime_feedback(action, observation)
    assert feedback["kind"] == "verifier_feedback"
    assert feedback["status"] == "failed"
    assert feedback["exit_code"] == 1
    assert "1 failed" in feedback["stdout"]
    assert "AssertionError" in feedback["stderr"]


def test_training_and_live_transition_use_hybrid_feedback_schema() -> None:
    event = hybrid_event({
        "kind": "transition",
        "action": {"name": "test.run", "inputs": {}},
        "observation": {
            "process": {
                "exit_code": 1,
                "stdout": "failed test_widget",
                "stderr": "ValueError: bad widget",
            }
        },
    })
    assert event["kind"] == "transition"
    assert event["feedback"]["kind"] == "verifier_feedback"
    assert event["feedback"]["status"] == "failed"
    assert event["feedback"]["exit_code"] == 1


def test_repair_plan_contains_contract_grounding_and_validated_ir() -> None:
    plan = repair_plan_from_state(
        goal="Fix controller loader",
        action="repo.replace",
        pointer_index=1,
        plan_probabilities=(0.9, 0.1, 0.8, 0.2),
        cognition={
            "candidate_paths": ["src/a.py", "src/controller.py"],
            "candidate_locations": [
                "symbol://repo@head/controller:load",
                "ast_call://repo@head/controller:from_pretrained",
            ],
            "active_concepts": ["symbol://repo@head/controller:load", "type://python/Loader"],
            "open_questions": ["Does the loader require remote code?"],
        },
        slots=9,
        reasoning_depth=5,
    )
    assert plan["task_intent"]["request"] == "Fix controller loader"
    assert plan["repair_ir"]["target_path"] == "src/controller.py"
    assert plan["repair_ir"]["intended_action"] == "repo.replace"
    assert plan["repair_ir"]["operations"][0]["kind"] == "replace_span"
    assert plan["repair_ir_validation"]["valid"] is True
    assert plan["repair_readiness"] == pytest.approx(0.9)
    assert plan["expected_verifier_success"] == pytest.approx(0.8)
    assert plan["reasoning_depth"] == 5
    assert plan["latent_slots"][0] == "user_intent"
    assert "repair_intent" in plan["latent_slots"]


def test_plan_supervision_separates_evidence_repair_and_stop() -> None:
    assert supervised_plan_targets(
        "repo.read", stop_target=False, value_target=0.25
    ) == (0.0, 1.0, 0.25, 0.0)
    assert supervised_plan_targets(
        "repo.replace", stop_target=False, value_target=1.0
    ) == (1.0, 0.0, 1.0, 0.0)
    assert supervised_plan_targets(
        "stop", stop_target=True, value_target=1.0
    ) == (0.0, 0.0, 1.0, 1.0)


def test_patch_renderer_prompt_is_conditioned_on_intent_and_repair_ir() -> None:
    prompt = repair_prompt(
        "repo.replace",
        "Fix VALUE",
        {"kind": "goal", "goal": "Fix VALUE"},
        target_path="value.py",
        repair_plan={
            "task_intent": {"request": "Fix VALUE", "requested_outcomes": ["Fix VALUE"]},
            "grounded_intent": {"candidate_paths": ["value.py"]},
            "repair_ir": {
                "target_path": "value.py",
                "semantic_draft": "replace broken with fixed",
                "invariants": ["preserve unrelated behavior"],
            },
            "repair_ir_validation": {"valid": True},
        },
    )
    assert "constrained semantic patch compiler" in prompt
    assert "user intent contract" in prompt.lower()
    assert "value.py" in prompt
    assert "replace broken with fixed" in prompt
    assert "path, old and new" in prompt


def test_hybrid_core_keeps_user_intent_anchor_across_observations_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    core = make_hybrid_core(
        torch,
        hidden_size=32,
        config=HybridLatentConfig(
            latent_dim=16,
            slots=9,
            min_reasoning_steps=2,
            max_reasoning_steps=3,
            convergence_tolerance=0.0,
            intent_anchor_strength=1.0,
        ),
        device="cpu",
    )
    first = torch.randn(1, 32)
    fused1, latent1, plan1, depth1 = core(first)
    intent1 = latent1[:, 0, :].detach().clone()
    fused2, latent2, plan2, depth2 = core(torch.randn(1, 32), latent1)
    assert fused1.shape == (1, 32)
    assert fused2.shape == (1, 32)
    assert latent1.shape == (1, 9, 16)
    assert latent2.shape == (1, 9, 16)
    assert plan1.shape == (1, 4)
    assert plan2.shape == (1, 4)
    assert depth1 == 3
    assert depth2 == 3
    assert torch.allclose(intent1, latent2[:, 0, :], atol=1e-5, rtol=1e-5)
