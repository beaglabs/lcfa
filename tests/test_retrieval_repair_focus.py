from __future__ import annotations

from pathlib import Path
import subprocess

from lcfa.repair_supervision import historical_repair_targets, is_documentation_path
from lcfa.repo_index import PythonRepoIndexer
from lcfa.repo_retrieval import build_retrieval_context
from lcfa.semantic_graph import SQLiteSemanticGraph


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "lcfa@example.invalid")
    _git(repo, "config", "user.name", "LCFA Test")


def test_historical_repair_targets_ignore_documentation_by_default(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    (repo / "src" / "controller.py").write_text("VALUE = 'broken'\n", encoding="utf-8")
    (repo / "docs" / "controller.md").write_text("old docs\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")

    (repo / "src" / "controller.py").write_text("VALUE = 'fixed'\n", encoding="utf-8")
    (repo / "docs" / "controller.md").write_text("new docs\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fix")
    fix = _git(repo, "rev-parse", "HEAD")

    targets = historical_repair_targets(repo, base, fix)
    assert [target.path for target in targets] == ["src/controller.py"]
    assert is_documentation_path("docs/controller.md") is True
    assert is_documentation_path("README.md") is True
    assert is_documentation_path("requirements.txt") is False

    all_targets = historical_repair_targets(
        repo, base, fix, include_documentation=True
    )
    assert {target.path for target in all_targets} == {
        "src/controller.py",
        "docs/controller.md",
    }


def test_shared_retriever_uses_issue_filename_evidence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    (repo / "rwkv_controller.py").write_text(
        "DEFAULT_RWKV_MODEL = 'old'\n\n"
        "def load_controller():\n"
        "    return DEFAULT_RWKV_MODEL\n",
        encoding="utf-8",
    )
    for index in range(7):
        (repo / f"helper_{index}.py").write_text(
            "def helper():\n"
            "    trust_remote_code = True\n"
            "    loader = 'RWKV loader checkpoint tokenizer model'\n"
            "    return trust_remote_code, loader\n",
            encoding="utf-8",
        )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    goal = (
        "Fix the RWKV-7 controller loader so the default checkpoint is correct "
        "and tokenizer/model loading enables trust_remote_code"
    )
    with SQLiteSemanticGraph(tmp_path / "semantic.db") as graph:
        PythonRepoIndexer(graph, repo).index()
        retrieval = build_retrieval_context(graph, goal)

    assert "rwkv_controller.py" in retrieval.candidate_paths[:5]
    controller = next(
        candidate
        for candidate in retrieval.candidates
        if candidate.path == "rwkv_controller.py"
    )
    assert any(
        evidence.startswith("path-")
        for evidence in controller.evidence
    )
