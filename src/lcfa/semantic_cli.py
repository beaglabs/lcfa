"""CLI for LCFA semantic repository indexing, investigation, and agent execution."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from .cognitive import SemanticInvestigator
from .repo_index import PythonRepoIndexer
from .semantic_agent import SemanticWorkspaceAgent
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

    agent = sub.add_parser("agent", help="run the closed-loop semantic coding/terminal agent")
    agent.add_argument("goal")
    agent.add_argument("--root", default=".")
    agent.add_argument("--db")
    controller = agent.add_mutually_exclusive_group(required=True)
    controller.add_argument("--artifact", help="stochastic language-policy artifact")
    controller.add_argument("--rwkv-controller", help="directory containing trained controller.json + heads.safetensors")
    agent.add_argument("--rwkv-model", help="override controller manifest RWKV model id")
    agent.add_argument("--rwkv-device", help="RWKV device override, e.g. mps/cuda/cpu")
    agent.add_argument("--rwkv-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    agent.add_argument("--max-steps", type=int, default=12)
    agent.add_argument("--allow-docs", action="store_true", help="enable allowlisted live documentation retrieval")
    agent.add_argument("--auto-approve", action="store_true", help="approve workspace edits/process actions; use only inside an isolated benchmark/worktree")
    agent.add_argument("--output", "-o")
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
                    payload["neighbors"] = {node.id: graph.neighbors(node.id, limit=20) for node in nodes[:5]}
                _print(payload)
                return 0

        if args.command == "agent":
            PythonRepoIndexer(graph, root).index()
            if args.rwkv_controller:
                from .rwkv_semantic import load_rwkv_semantic_backbone

                backbone = load_rwkv_semantic_backbone(
                    args.rwkv_controller,
                    graph,
                    model_id=args.rwkv_model,
                    device=args.rwkv_device,
                    dtype=args.rwkv_dtype,
                )
                agent_runtime = SemanticWorkspaceAgent(
                    graph,
                    root,
                    backbone,
                    max_steps=args.max_steps,
                    allow_docs=args.allow_docs,
                )
            else:
                agent_runtime = SemanticWorkspaceAgent.from_artifact(
                    graph,
                    root,
                    args.artifact,
                    max_steps=args.max_steps,
                    allow_docs=args.allow_docs,
                )
            episode = agent_runtime.run(args.goal, auto_approve=args.auto_approve)
            text = json.dumps(_safe(episode), indent=2, sort_keys=True, ensure_ascii=False)
            if args.output:
                Path(args.output).write_text(text + "\n", encoding="utf-8")
            else:
                print(text)
            return 0

        solution = SemanticInvestigator(graph).investigate(args.goal, limit=args.limit)
        if args.command == "actions":
            _print(compile_cognitive_actions(solution))
            return 0

        text = json.dumps(_safe(solution), indent=2, sort_keys=True, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
