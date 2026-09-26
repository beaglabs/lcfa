"""CLI for LCFA recurrent-controller collection, datasets, training, and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .recurrent_collect import collect_trajectories
from .recurrent_eval import evaluate_rwkv_heads
from .recurrent_train import train_rwkv_heads
from .recurrent_transitions import prepare_transition_file
from .rwkv_controller import DEFAULT_RWKV_MODEL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcfa-recurrent",
        description="Collect/prepare LCFA trajectories and train/evaluate recurrent RWKV policy heads.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser(
        "collect",
        help="run a semantic teacher over JSONL coding tasks in isolated git worktrees",
    )
    collect.add_argument("tasks", help="JSONL tasks with repo, goal, base_ref, and verify_argv")
    collect.add_argument("--artifact", required=True, help="semantic teacher stochastic artifact")
    collect.add_argument("--output-dir", "-o", required=True)
    collect.add_argument("--worktree-root", default="/tmp/lcfa-recurrent-worktrees")
    collect.add_argument("--max-steps", type=int, default=12)
    collect.add_argument("--max-tasks", type=int)
    collect.add_argument("--allow-docs", action="store_true")
    collect.add_argument("--keep-worktrees", action="store_true")
    collect.add_argument(
        "--allow-baseline-pass",
        action="store_true",
        help="collect tasks even when verify_argv already passes before the agent",
    )

    prepare = sub.add_parser("prepare", help="convert semantic episode JSON files into transition JSONL")
    prepare.add_argument("episodes", nargs="+")
    prepare.add_argument("--output", "-o", required=True)

    train = sub.add_parser("train", help="train action/stop/value heads on frozen RWKV recurrent state")
    train.add_argument("transitions")
    train.add_argument("--output-dir", "-o", required=True)
    train.add_argument("--model", default=DEFAULT_RWKV_MODEL)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--device")
    train.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    train.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="episode-level held-out fraction; one-episode datasets remain train-only",
    )
    train.add_argument("--seed", type=int, default=20260925)

    evaluate = sub.add_parser(
        "evaluate",
        help="replay a transition dataset from fresh recurrent states using a trained controller",
    )
    evaluate.add_argument("transitions")
    evaluate.add_argument("--controller", required=True)
    evaluate.add_argument("--model", help="override controller manifest model_id")
    evaluate.add_argument("--device")
    evaluate.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    evaluate.add_argument("--output", "-o")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "collect":
        summary = collect_trajectories(
            args.tasks,
            artifact=args.artifact,
            output_dir=args.output_dir,
            worktree_root=args.worktree_root,
            max_steps=args.max_steps,
            allow_docs=args.allow_docs,
            keep_worktrees=args.keep_worktrees,
            require_baseline_failure=not args.allow_baseline_pass,
            max_tasks=args.max_tasks,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    if args.command == "prepare":
        count = prepare_transition_file(args.episodes, args.output)
        print(json.dumps({"transitions": count, "output": args.output}, indent=2))
        return 0

    if args.command == "evaluate":
        result = evaluate_rwkv_heads(
            args.transitions,
            args.controller,
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    summary = train_rwkv_heads(
        args.transitions,
        args.output_dir,
        model_id=args.model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        dtype=args.dtype,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
