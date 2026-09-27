"""CLI for the LCFA typed-latent + RWKV hybrid alpha."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from .hybrid_collect import collect_hybrid_trajectories
from .hybrid_improve import improve_hybrid_controller
from .hybrid_semantic_train import BACKBONE_MODES, train_hybrid_controller
from .recurrent_corrections import prepare_corrective_transition_file
from .recurrent_transitions import merge_transition_files, prepare_transition_file
from .rwkv_controller import DEFAULT_RWKV_MODEL


def _runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )


def _training_args(parser: argparse.ArgumentParser, *, model_default: bool = True) -> None:
    if model_default:
        parser.add_argument("--model", default=DEFAULT_RWKV_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--backbone-mode",
        choices=BACKBONE_MODES,
        default="full",
        help="full also trains the RWKV patch renderer; frozen trains latent/core heads only",
    )
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-6)
    parser.add_argument("--pointer-loss-weight", type=float, default=0.5)
    parser.add_argument("--plan-loss-weight", type=float, default=0.25)
    parser.add_argument("--argument-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-argument-chars", type=int, default=8192)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--latent-slots", type=int, default=9)
    parser.add_argument("--min-reasoning-steps", type=int, default=2)
    parser.add_argument("--max-reasoning-steps", type=int, default=6)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-3)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcfa-hybrid",
        description=(
            "Train and run the LCFA typed-latent workspace fused with RWKV recurrent memory."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train the hybrid latent + RWKV controller")
    train.add_argument("transitions")
    train.add_argument("--output-dir", "-o", required=True)
    train.add_argument(
        "--init-controller",
        help="initialize from either an RWKV-only or prior hybrid controller",
    )
    _training_args(train)

    rollout = sub.add_parser("rollout", help="run autonomous hybrid trajectories")
    rollout.add_argument("tasks")
    rollout.add_argument("--controller", required=True)
    rollout.add_argument("--output-dir", "-o", required=True)
    rollout.add_argument("--worktree-root", default="/tmp/lcfa-hybrid-worktrees")
    rollout.add_argument("--max-steps", type=int, default=12)
    rollout.add_argument("--max-tasks", type=int)
    rollout.add_argument("--allow-docs", action="store_true")
    rollout.add_argument("--keep-worktrees", action="store_true")
    rollout.add_argument("--allow-baseline-pass", action="store_true")
    _runtime_args(rollout)

    improve = sub.add_parser(
        "improve",
        help="roll out, relabel failures, merge verified trajectories, and retrain",
    )
    improve.add_argument("tasks")
    improve.add_argument("--base-transitions", required=True)
    improve.add_argument("--controller", required=True)
    improve.add_argument("--output-root", required=True)
    improve.add_argument("--rounds", type=int, default=1)
    improve.add_argument("--model")
    improve.add_argument("--max-steps", type=int, default=12)
    improve.add_argument("--allow-docs", action="store_true")
    _training_args(improve, model_default=False)

    prepare = sub.add_parser("prepare", help="prepare recurrent transitions from episodes")
    prepare.add_argument("episodes", nargs="+")
    prepare.add_argument("--output", "-o", required=True)

    correct = sub.add_parser("correct", help="create verifier-backed corrective transitions")
    correct.add_argument("episodes", nargs="+")
    correct.add_argument("--tasks", required=True)
    correct.add_argument("--output", "-o", required=True)

    merge = sub.add_parser("merge", help="merge transition corpora")
    merge.add_argument("transitions", nargs="+")
    merge.add_argument("--output", "-o", required=True)
    return parser


def _progress(event: Mapping[str, Any]) -> None:
    kind = str(event.get("event") or "")
    if kind in {"task-start", "task-done", "round-start", "round-data", "round-done"}:
        print("[lcfa-hybrid] " + json.dumps(dict(event), sort_keys=True, default=str), file=sys.stderr, flush=True)
    elif kind == "epoch-done":
        pointer = event.get("pointer_accuracy")
        pointer_text = "n/a" if pointer is None else f"{float(pointer):.3f}"
        print(
            "[lcfa-hybrid] "
            f"epoch={event.get('epoch')}/{event.get('epochs')} "
            f"loss={float(event.get('loss', 0.0)):.6f} "
            f"action={float(event.get('action_accuracy', 0.0)):.3f} "
            f"pointer={pointer_text} stop={float(event.get('stop_accuracy', 0.0)):.3f} "
            f"plan={event.get('plan_examples')} args={event.get('argument_examples')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "feature-cache-progress":
        print(
            "[lcfa-hybrid] cache "
            f"episode={event.get('episode')}/{event.get('episodes')} "
            f"transitions={event.get('transitions')}/{event.get('total_transitions')} "
            f"tokens={event.get('tokens')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "training-done":
        print(
            f"[lcfa-hybrid] trained updates={event.get('updates')} output={event.get('output')}",
            file=sys.stderr,
            flush=True,
        )


def _train_kwargs(args: argparse.Namespace) -> Mapping[str, Any]:
    return {
        "model_id": args.model or DEFAULT_RWKV_MODEL,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "dtype": args.dtype,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "backbone_mode": args.backbone_mode,
        "backbone_learning_rate": args.backbone_learning_rate,
        "pointer_loss_weight": args.pointer_loss_weight,
        "plan_loss_weight": args.plan_loss_weight,
        "argument_loss_weight": args.argument_loss_weight,
        "max_argument_chars": args.max_argument_chars,
        "latent_dim": args.latent_dim,
        "latent_slots": args.latent_slots,
        "min_reasoning_steps": args.min_reasoning_steps,
        "max_reasoning_steps": args.max_reasoning_steps,
        "convergence_tolerance": args.convergence_tolerance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        count = prepare_transition_file(args.episodes, args.output)
        if count == 0:
            Path(args.output).unlink(missing_ok=True)
            parser.error("no recurrent transitions were produced")
        print(json.dumps({"transitions": count, "output": args.output}, indent=2))
        return 0

    if args.command == "correct":
        count = prepare_corrective_transition_file(args.episodes, args.tasks, args.output)
        if count == 0:
            Path(args.output).unlink(missing_ok=True)
        print(json.dumps({"corrective_transitions": count, "output": args.output}, indent=2))
        return 0

    if args.command == "merge":
        count = merge_transition_files(args.transitions, args.output)
        print(json.dumps({"transitions": count, "output": args.output}, indent=2))
        return 0

    if args.command == "rollout":
        summary = collect_hybrid_trajectories(
            args.tasks,
            controller=args.controller,
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            output_dir=args.output_dir,
            worktree_root=args.worktree_root,
            max_steps=args.max_steps,
            allow_docs=args.allow_docs,
            keep_worktrees=args.keep_worktrees,
            require_baseline_failure=not args.allow_baseline_pass,
            max_tasks=args.max_tasks,
            progress=_progress,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "improve":
        kwargs = dict(_train_kwargs(args))
        summary = improve_hybrid_controller(
            args.tasks,
            base_transitions=args.base_transitions,
            controller=args.controller,
            output_root=args.output_root,
            rounds=args.rounds,
            max_steps=args.max_steps,
            allow_docs=args.allow_docs,
            progress=_progress,
            **kwargs,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    kwargs = dict(_train_kwargs(args))
    summary = train_hybrid_controller(
        args.transitions,
        args.output_dir,
        init_controller=args.init_controller,
        progress=_progress,
        **kwargs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
