from __future__ import annotations

from dataclasses import replace

from lcfa.intent_ir import (
    RepairOperation,
    compile_repair_ir,
    compile_task_intent,
    ground_task_intent,
    rectify_repair_ir,
    validate_rendered_inputs,
    validate_repair_ir,
)


def test_user_request_becomes_explicit_intent_contract() -> None:
    contract = compile_task_intent(
        "Make the avatar fill the whole container. Preserve the surrounding layout. "
        "Do not invent authentication state."
    )
    assert contract.request.startswith("Make the avatar")
    assert contract.requested_outcomes == ("Make the avatar fill the whole container",)
    assert contract.preserve_constraints == ("Preserve the surrounding layout",)
    assert contract.prohibited_outcomes == ("Do not invent authentication state",)
    assert contract.acceptance_conditions


def test_intent_grounding_uses_visible_semantic_ids() -> None:
    contract = compile_task_intent("Fix the controller loader")
    grounded = ground_task_intent(
        contract,
        {
            "candidate_paths": ["src/lcfa/rwkv_controller.py", "src/lcfa/recurrent_train.py"],
            "candidate_locations": [
                "symbol://repo@head/lcfa.rwkv_controller:RWKVRecurrentPolicy.__init__",
                "ast_call://repo@head/lcfa.rwkv_controller:AutoModelForCausalLM.from_pretrained",
            ],
            "active_concepts": ["type://python/str"],
        },
        pointer_index=0,
    )
    assert grounded.candidate_paths[0] == "src/lcfa/rwkv_controller.py"
    assert grounded.targets[0].entity_id.startswith("symbol://")
    assert grounded.targets[1].entity_id.startswith("ast_call://")
    assert grounded.targets[0].path == "src/lcfa/rwkv_controller.py"


def test_grounded_intent_compiles_to_typed_repair_ir() -> None:
    contract = compile_task_intent("Enable remote-code loading in the RWKV controller")
    grounded = ground_task_intent(
        contract,
        {
            "candidate_paths": ["src/lcfa/rwkv_controller.py"],
            "candidate_locations": [
                "symbol://repo@head/lcfa.rwkv_controller:RWKVRecurrentPolicy.__init__",
                "ast_call://repo@head/lcfa.rwkv_controller:AutoModelForCausalLM.from_pretrained",
            ],
        },
    )
    ir = compile_repair_ir(
        grounded,
        action="repo.replace",
        expected_verifier_success=0.9,
        reasoning_depth=5,
    )
    assert ir.target_path == "src/lcfa/rwkv_controller.py"
    assert ir.operations[0].kind == "replace_span"
    assert ir.operations[0].target_entity.startswith("symbol://")
    assert "Enable remote-code loading" in ir.semantic_draft
    assert ir.expected_verifier_success == 0.9
    assert validate_repair_ir(ir, grounded).valid


def test_repair_ir_rejects_and_rectifies_invisible_target() -> None:
    contract = compile_task_intent("Fix the loader")
    grounded = ground_task_intent(
        contract,
        {
            "candidate_paths": ["src/right.py"],
            "candidate_locations": ["symbol://repo@head/right:load"],
        },
    )
    ir = compile_repair_ir(grounded, action="repo.replace")
    invalid = replace(
        ir,
        target_path="src/hidden.py",
        operations=(
            RepairOperation(
                kind="replace_span",
                target_entity="symbol://repo@head/hidden:load",
                target_path="src/hidden.py",
                parameters={"action": "repo.replace"},
            ),
        ),
        target_entities=("symbol://repo@head/hidden:load",),
    )
    result = validate_repair_ir(invalid, grounded)
    assert not result.valid
    rectified = rectify_repair_ir(invalid, grounded)
    assert rectified.target_path == "src/right.py"
    assert rectified.target_entities == ("symbol://repo@head/right:load",)
    assert validate_repair_ir(rectified, grounded).valid


def test_rendered_patch_cannot_drift_from_repair_ir_target() -> None:
    contract = compile_task_intent("Fix the loader")
    grounded = ground_task_intent(
        contract,
        {"candidate_paths": ["src/right.py"], "candidate_locations": ["symbol://right:load"]},
    )
    ir = compile_repair_ir(grounded, action="repo.replace")
    good = validate_rendered_inputs(
        ir,
        "repo.replace",
        {"path": "src/right.py", "old": "broken", "new": "fixed"},
        visible_paths=grounded.candidate_paths,
    )
    bad = validate_rendered_inputs(
        ir,
        "repo.replace",
        {"path": "src/elsewhere.py", "old": "broken", "new": "fixed"},
        visible_paths=grounded.candidate_paths,
    )
    assert good.valid
    assert not bad.valid
    assert any("diverges" in error for error in bad.errors)
