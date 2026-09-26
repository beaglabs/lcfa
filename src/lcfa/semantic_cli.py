"""CLI for LCFA semantic repository indexing and investigation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from .cognitive import SemanticInvestigator
from .repo_index import PythonRepoIndexer
from .semantic_graph import SQLiteSemanticGraph
from .workspace_actions import compile_cognitive_actions


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(item) for item in value]
    return value


def _print(value: Any) -> None:
    print(json.dumps(_safe(value), indent=2, sort_keys=True, ensure_ascii=False))


def _default_db(root: str | Path) -> Path:
    return Path(root).resolve() / ".lcfa" / "semantic.db"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcfa-semantic", description="Content-addressed semantic runtime for repositories and terminal agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="index a Python repository into a persistent semantic graph")
    index.add_argument("root", nargs="?", default=".")
    index.add_argument("--db")

    concept = sub.add_parser("concept", help="search or inspect semantic concepts")
    concept.add_argument("query")
    concept.add_argument("--root", default=".")
    concept.add_argument("--db")
    concept.add_argument("--limit", type=int, default=20)
    concept.add_argument("--neighbors", action="store_true")

    investigate = sub.add_parser("investigate", help="form an evidence-linked cognitive SolutionState for a goal/issue")
    investigate.add_argument("goal")
    investigate.add_argument("--root", default=".")
    investigate.add_argument("--db")
    investigate.add_argument("--limit", type=int, default=12)
    investigate.add_argument("--output", "-o")

    actions = sub.add_parser("actions", help="compile the next semantic actions for a goal")
    actions.add_argument("goal")
    actions.add_argument("--root", default=".")
    actions.add_argument("--db")
    actions.add_argument("--limit", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(getattr(args, "root", ".")).resolve()
    db_path = Path(args.db).resolve() if getattr(args, "db", None) else _default_db(root)

    with SQLiteSemanticGraph(db_path) as graph:
        if args.command == "index":
            result = PythonRepoIndexer(graph, root).index()
            _print(result)
            return 0

        if args.command == "concept":
            try:
                node = graph.get_node(args.query)
                payload: dict[str, Any] = {"node": node}
                if args.neighbors:
                    payload["neighbors"] = graph.neighbors(node.id)
                _print(payload)
                return 0
            except KeyError:
                nodes = graph.search(args.query, limit=args.limit)
                payload = {"matches": nodes}
                if args.neighbors:
                    payload["neighbors"] = {
                        node.id: graph.neighbors(node.id, limit=20) for node in nodes[:5]
                    }
                _print(payload)
                return 0

        solution = SemanticInvestigator(graph).investigate(args.goal, limit=args.limit)
        if args.command == "actions":
            _print(compile_cognitive_actions(solution))
            return 0

        payload = _safe(solution)
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
