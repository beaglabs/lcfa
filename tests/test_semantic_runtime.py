from __future__ import annotations

from pathlib import Path

from lcfa import (
    ActionExecutor,
    ExecutionContext,
    PythonRepoIndexer,
    SQLiteSemanticGraph,
    SemanticInvestigator,
    compile_cognitive_actions,
    register_workspace_actions,
)


def _repo(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname="demo"\nversion="0.1"\ndependencies=["pydantic>=2.7"]\n',
        encoding="utf-8",
    )
    (root / "pkg" / "models.py").write_text(
        """from typing import Optional\n\nclass User:\n    def normalize_name(self, value: Optional[str]) -> str:\n        return value.strip() if value else \"\"\n\ndef load_user(name: Optional[str]) -> User:\n    user = User()\n    user.normalize_name(name)\n    return user\n""",
        encoding="utf-8",
    )
    (root / "tests" / "test_models.py").write_text(
        """from pkg.models import User\n\ndef test_optional_name():\n    assert User().normalize_name(None) == \"\"\n""",
        encoding="utf-8",
    )


def test_python_repo_index_builds_content_addressed_graph(tmp_path: Path) -> None:
    _repo(tmp_path)
    with SQLiteSemanticGraph(tmp_path / ".lcfa" / "semantic.db") as graph:
        result = PythonRepoIndexer(graph, tmp_path).index()
        assert result.files == 2
        assert result.symbols >= 4
        assert result.packages >= 2
        assert result.types >= 2
        assert result.snapshot.node_count > 0
        hits = graph.search("normalize optional", limit=20)
        assert any(node.label == "normalize_name" for node in hits)
        node = next(node for node in hits if node.label == "normalize_name")
        assert node.content.content_hash.startswith("b3:")
        assert any(edge.relation in {"accepts", "returns", "calls"} for edge in graph.neighbors(node.id))


def test_investigation_emits_evidence_linked_hypotheses_and_actions(tmp_path: Path) -> None:
    _repo(tmp_path)
    with SQLiteSemanticGraph(tmp_path / ".lcfa" / "semantic.db") as graph:
        PythonRepoIndexer(graph, tmp_path).index()
        solution = SemanticInvestigator(graph).investigate(
            "Optional user names fail during normalize_name handling"
        )
        cognition = solution.values["cognition"]
        assert cognition["active_concepts"]
        assert cognition["hypotheses"]
        assert cognition["candidate_locations"]
        assert solution.evidence
        assert all(ref.id.startswith("b3:") for ref in solution.evidence)
        action_graph = compile_cognitive_actions(solution)
        assert action_graph.nodes
        assert action_graph.nodes[0].action in {"repo.read", "repo.search"}
        assert "workspace.read" in action_graph.nodes[0].required_capabilities


def test_compiled_read_action_executes_through_governance(tmp_path: Path) -> None:
    _repo(tmp_path)
    with SQLiteSemanticGraph(tmp_path / ".lcfa" / "semantic.db") as graph:
        PythonRepoIndexer(graph, tmp_path).index()
        solution = SemanticInvestigator(graph).investigate("normalize_name Optional")
        action_graph = compile_cognitive_actions(solution)
    registry = register_workspace_actions()
    run = ActionExecutor(registry).execute(
        action_graph,
        solution,
        ExecutionContext(
            capabilities=frozenset({"workspace.read"}),
            metadata={"workspace_root": str(tmp_path)},
        ),
    )
    assert run.results
    first = run.results[action_graph.nodes[0].id].value
    assert "path" in first or "hits" in first
