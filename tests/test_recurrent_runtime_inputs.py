from __future__ import annotations

from lcfa.rwkv_controller import _valid_generated_inputs


def test_repo_replace_generated_inputs_require_action_contract() -> None:
    assert _valid_generated_inputs(
        "repo.replace",
        {"path": "src/lcfa/example.py", "old": "before", "new": "after"},
    )
    assert not _valid_generated_inputs(
        "repo.replace",
        {"path": "src/lcfa/example.py", "name": "call_1"},
    )
    assert not _valid_generated_inputs(
        "repo.replace",
        {"path": "src/lcfa/example.py", "old": "", "new": "after"},
    )
    assert _valid_generated_inputs(
        "repo.replace",
        {"path": "src/lcfa/example.py", "old": "before", "new": ""},
    )


def test_repo_edit_generated_inputs_require_content() -> None:
    assert _valid_generated_inputs(
        "repo.edit",
        {"path": "src/lcfa/example.py", "content": "print('ok')\n"},
    )
    assert not _valid_generated_inputs(
        "repo.edit",
        {"path": "src/lcfa/example.py", "arguments": {"content": "nested"}},
    )
