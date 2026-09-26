from __future__ import annotations

from pathlib import Path
import subprocess

from lcfa.repo_index import PythonRepoIndexer
from lcfa.repo_retrieval import build_retrieval_context, extract_retrieval_queries
from lcfa.semantic_graph import SQLiteSemanticGraph


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_shared_retrieval_indexes_exact_identifiers_and_ast_occurrences(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "lcfa@example.invalid")
    _git(repo, "config", "user.name", "LCFA Test")
    source = repo / "loader.py"
    source.write_text(
        "from transformers import AutoTokenizer\n\n"
        "def load_tokenizer(model_id):\n"
        "    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n",
        encoding="utf-8",
    )
    (repo / "other.py").write_text(
        "def unrelated():\n    return 'nothing to do with loading'\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    with SQLiteSemanticGraph(tmp_path / "semantic.db") as graph:
        PythonRepoIndexer(graph, repo).index()
        call_identifier = graph.get_node(
            "identifier://python/AutoTokenizer.from_pretrained"
        )
        keyword_identifier = graph.get_node(
            "identifier://python/trust_remote_code"
        )
        assert call_identifier.kind == "identifier"
        assert keyword_identifier.kind == "identifier"
        ast_neighbors = graph.neighbors(call_identifier.id, direction="in", limit=20)
        assert any(
            graph.get_node(edge.source).kind in {"ast_call", "ast_identifier"}
            for edge in ast_neighbors
        )

        goal = (
            "Fix AutoTokenizer.from_pretrained so RWKV loading enables "
            "trust_remote_code=True"
        )
        queries = extract_retrieval_queries(goal)
        assert "AutoTokenizer.from_pretrained" in queries[:3]
        assert "trust_remote_code" in queries[:5]

        retrieval = build_retrieval_context(graph, goal)
        assert retrieval.candidate_paths
        assert retrieval.candidate_paths[0] == "loader.py"
        assert "other.py" not in retrieval.candidate_paths[:1]
        assert any(
            evidence.startswith("exact:AutoTokenizer.from_pretrained")
            or evidence.startswith("ast:AutoTokenizer.from_pretrained")
            for candidate in retrieval.candidates
            for evidence in candidate.evidence
        )
