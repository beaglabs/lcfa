from __future__ import annotations

from types import SimpleNamespace

import pytest

from lcfa.recurrent_collect import (
    CollectionError,
    RecurrentCollectionTask,
    _select_oracle_repair_targets,
)


def _task(*, verify_argv: tuple[str, ...]) -> RecurrentCollectionTask:
    return RecurrentCollectionTask(
        id="oracle-localization",
        repo=".",
        goal="Fix the RWKV-7 controller loader",
        verify_argv=verify_argv,
    )


def test_hidden_historical_file_is_not_required_or_replayed(tmp_path) -> None:
    visible = tmp_path / "src/lcfa/rwkv_controller.py"
    hidden = tmp_path / "src/lcfa/recurrent_train.py"
    visible.parent.mkdir(parents=True)
    visible.write_text("# controller\n", encoding="utf-8")
    hidden.write_text("# training\n", encoding="utf-8")

    targets = (
        SimpleNamespace(path="src/lcfa/recurrent_train.py"),
        SimpleNamespace(path="src/lcfa/rwkv_controller.py"),
    )
    selected = _select_oracle_repair_targets(
        _task(verify_argv=(
            "python",
            "-c",
            "Path('src/lcfa/rwkv_controller.py').read_text()",
        )),
        worktree=tmp_path,
        repair_targets=targets,
        inspected_paths=("src/lcfa/rwkv_controller.py",),
    )

    assert [target.path for target in selected] == ["src/lcfa/rwkv_controller.py"]


def test_non_visible_historical_file_may_be_used_when_base_retrieval_found_it(tmp_path) -> None:
    path = tmp_path / "src/lcfa/recurrent_train.py"
    path.parent.mkdir(parents=True)
    path.write_text("# training\n", encoding="utf-8")
    target = SimpleNamespace(path="src/lcfa/recurrent_train.py")

    selected = _select_oracle_repair_targets(
        _task(verify_argv=("python", "-c", "assert True")),
        worktree=tmp_path,
        repair_targets=(target,),
        inspected_paths=("src/lcfa/recurrent_train.py",),
    )

    assert selected == (target,)


def test_task_visible_repair_path_must_still_be_retrieved(tmp_path) -> None:
    path = tmp_path / "src/lcfa/rwkv_controller.py"
    path.parent.mkdir(parents=True)
    path.write_text("# controller\n", encoding="utf-8")

    with pytest.raises(CollectionError, match="task-visible repair file"):
        _select_oracle_repair_targets(
            _task(verify_argv=(
                "python",
                "-c",
                "Path('src/lcfa/rwkv_controller.py').read_text()",
            )),
            worktree=tmp_path,
            repair_targets=(SimpleNamespace(path="src/lcfa/rwkv_controller.py"),),
            inspected_paths=("src/lcfa/backbones.py",),
        )
