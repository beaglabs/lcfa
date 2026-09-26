"""Python repository decomposition into the LCFA semantic graph."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tomllib
from typing import Any, Iterable

from .semantic_graph import ConceptSnapshot, SQLiteSemanticGraph

_SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or root.name


def _expr_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _expr_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _annotation(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return _expr_name(node) or None


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except Exception:
        return None


def _iter_python(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


@dataclass(frozen=True, slots=True)
class IndexResult:
    root: str
    commit: str | None
    files: int
    symbols: int
    packages: int
    types: int
    snapshot: ConceptSnapshot


class _Visitor(ast.NodeVisitor):
    def __init__(self, indexer: "PythonRepoIndexer", path: Path, module_id: str, module_name: str, source: str) -> None:
        self.indexer = indexer
        self.path = path
        self.module_id = module_id
        self.module_name = module_name
        self.source = source
        self.stack: list[str] = []
        self.class_stack: list[str] = []

    def _owner(self) -> str:
        return self.stack[-1] if self.stack else self.module_id

    def _symbol_id(self, name: str) -> str:
        scope = ".".join([*self.class_stack, name]) if self.class_stack else name
        return f"symbol://{self.indexer.repo_key}/{self.module_name}:{scope}"

    def _put_type(self, text: str | None, owner: str, relation: str) -> None:
        if not text:
            return
        type_id = f"type://python/{text}"
        self.indexer.graph.put_node(type_id, "type", text, {"type": text}, metadata={"language": "python"})
        self.indexer.graph.add_edge(owner, relation, type_id)
        self.indexer.types.add(type_id)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            package = alias.name.split(".")[0]
            pkg_id = f"package://python/{package}"
            self.indexer.graph.put_node(pkg_id, "package", package, {"name": package}, metadata={"ecosystem": "python"})
            self.indexer.graph.add_edge(self._owner(), "imports", pkg_id, metadata={"name": alias.name})
            self.indexer.packages.add(pkg_id)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.module:
            package = node.module.split(".")[0]
            pkg_id = f"package://python/{package}"
            self.indexer.graph.put_node(pkg_id, "package", package, {"name": package}, metadata={"ecosystem": "python"})
            self.indexer.graph.add_edge(self._owner(), "imports", pkg_id, metadata={"name": node.module})
            self.indexer.packages.add(pkg_id)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        symbol_id = self._symbol_id(node.name)
        segment = ast.get_source_segment(self.source, node) or node.name
        params = []
        all_args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for arg in all_args:
            ann = _annotation(arg.annotation)
            params.append({"name": arg.arg, "annotation": ann})
        returns = _annotation(node.returns)
        kind = "method" if self.class_stack else "function"
        self.indexer.graph.put_node(
            symbol_id, kind, node.name, segment,
            media_type="text/x-python",
            metadata={
                "module": self.module_name, "path": str(self.path.relative_to(self.indexer.root)),
                "lineno": node.lineno, "end_lineno": getattr(node, "end_lineno", None),
                "parameters": params, "returns": returns,
                "async": isinstance(node, ast.AsyncFunctionDef),
            },
        )
        self.indexer.graph.add_edge(self._owner(), "defines", symbol_id)
        self.indexer.symbols.add(symbol_id)
        for item in params:
            self._put_type(item["annotation"], symbol_id, "accepts")
        self._put_type(returns, symbol_id, "returns")
        self.stack.append(symbol_id)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        symbol_id = self._symbol_id(node.name)
        segment = ast.get_source_segment(self.source, node) or node.name
        bases = [_expr_name(base) for base in node.bases]
        self.indexer.graph.put_node(
            symbol_id, "class", node.name, segment, media_type="text/x-python",
            metadata={
                "module": self.module_name, "path": str(self.path.relative_to(self.indexer.root)),
                "lineno": node.lineno, "end_lineno": getattr(node, "end_lineno", None), "bases": bases,
            },
        )
        self.indexer.graph.add_edge(self._owner(), "defines", symbol_id)
        self.indexer.symbols.add(symbol_id)
        for base in bases:
            base_id = f"type://python/{base}"
            self.indexer.graph.put_node(base_id, "type", base, {"type": base}, metadata={"language": "python"})
            self.indexer.graph.add_edge(symbol_id, "inherits", base_id)
            self.indexer.types.add(base_id)
        self.stack.append(symbol_id)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()
        self.stack.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        ann = _annotation(node.annotation)
        self._put_type(ann, self._owner(), "annotated_with")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = _expr_name(node.func)
        if name:
            target_id = f"callable://python/{name}"
            self.indexer.graph.put_node(target_id, "callable_ref", name, {"name": name}, metadata={"language": "python"})
            self.indexer.graph.add_edge(self._owner(), "calls", target_id, metadata={"lineno": getattr(node, "lineno", None)})
        self.generic_visit(node)


class PythonRepoIndexer:
    def __init__(self, graph: SQLiteSemanticGraph, root: str | Path) -> None:
        self.graph = graph
        self.root = Path(root).resolve()
        self.commit = _git_commit(self.root)
        self.repo_key = f"{self.root.name}@{self.commit or 'working'}"
        self.symbols: set[str] = set()
        self.packages: set[str] = set()
        self.types: set[str] = set()
        self.files = 0

    def _index_pyproject(self) -> None:
        path = self.root / "pyproject.toml"
        if not path.exists():
            return
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        deps = data.get("project", {}).get("dependencies", [])
        for raw in deps:
            name = str(raw).split(";", 1)[0].strip().split("[", 1)[0]
            for separator in (">=", "<=", "==", "~=", "!=", ">", "<"):
                name = name.split(separator, 1)[0]
            name = name.strip()
            if not name:
                continue
            pkg_id = f"package://python/{name.lower()}"
            self.graph.put_node(pkg_id, "package", name, {"requirement": str(raw)}, metadata={"ecosystem": "python", "declared": True})
            self.graph.add_edge(f"repo://{self.repo_key}", "depends_on", pkg_id)
            self.packages.add(pkg_id)

    def index(self) -> IndexResult:
        repo_id = f"repo://{self.repo_key}"
        self.graph.put_node(
            repo_id, "repository", self.root.name,
            {"root": str(self.root), "commit": self.commit},
            metadata={"root": str(self.root), "commit": self.commit, "language": "python"},
        )
        self._index_pyproject()
        for path in _iter_python(self.root):
            rel = str(path.relative_to(self.root))
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=rel, type_comments=True)
            except (UnicodeDecodeError, SyntaxError):
                continue
            self.files += 1
            file_id = f"file://{self.repo_key}/{rel}"
            module_name = _module_name(self.root, path)
            module_id = f"module://{self.repo_key}/{module_name}"
            self.graph.put_node(file_id, "file", rel, source, media_type="text/x-python", metadata={"path": rel, "language": "python"})
            self.graph.put_node(module_id, "module", module_name, {"module": module_name, "path": rel}, metadata={"path": rel, "language": "python"})
            self.graph.add_edge(repo_id, "contains", file_id)
            self.graph.add_edge(file_id, "defines", module_id)
            _Visitor(self, path, module_id, module_name, source).visit(tree)
            if rel.startswith("tests/") or path.name.startswith("test_") or "/test" in rel:
                self.graph.add_edge(repo_id, "has_test", file_id)
        snapshot = self.graph.snapshot(str(self.root), metadata={"commit": self.commit, "repo_key": self.repo_key})
        return IndexResult(
            root=str(self.root), commit=self.commit, files=self.files, symbols=len(self.symbols),
            packages=len(self.packages), types=len(self.types), snapshot=snapshot,
        )


__all__ = ["IndexResult", "PythonRepoIndexer"]
