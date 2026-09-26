"""CLI for LCFA recurrent collection, correction, training, and improvement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from .recurrent_collect import COLLECTION_MODES, collect_trajectories
from .recurrent_corrections import prepare_corrective_transition_file
from .recurrent_eval import evaluate_rwkv_heads
from .recurrent_improve import improve_controller
from .recurrent_train import BACKBONE_MODES, train_rwkv_heads
from .recurrent_transitions import merge_transition_files, prepare_transition_file
from .rwkv_controller import DEFAULT_RWKV_MODEL


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="override controller manifest RWKV model id")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cuda:N, or cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )


def _add_training_args(parser: argparse.ArgumentParser, *, include_model_default: bool = True) -> None:
    if include_model_default:
        parser.add_argument("--model", default=DEFAULT_RWKV_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cuda:N, or cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="episode-level held-out fraction; one-episode datasets remain train-only",
    )
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--backbone-mode",
        choices=BACKBONE_MODES,
        default="frozen",
        help="frozen caches RWKV features; full fine-tunes RWKV plus control heads",
    )
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-6)
    parser.add_argument("--pointer-loss-weight", type=float, default=0.5)
    parser.add_argument("--argument-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-argument-chars", type=int, default=8192)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcfa-recurrent",
        description=(
            "Collect, correct, train, evaluate, and iteratively improve recurrent RWKV software agents."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser(
        "collect",
        help="collect leakage-free historical oracle trajectories or RWKV self-rollouts",
    )
    collect.add_argument(
        "tasks",
        help="JSONL tasks with repo, goal, base_ref, verify_argv, and optional fix_ref",
    )
    collect.add_argument("--mode", choices=COLLECTION_MODES, default="oracle")
    collect.add_argument(
        "--controller",
        help="trained RWKV controller directory; required only for --mode rollout",
    )
    _add_runtime_args(collect)
    collect.add_argument("--output-dir", "-o", required=True)
    collect.add_argument("--worktree-root", default="/tmp/lcfa-recurrent-worktrees")
    collect.add_argument("--max-steps", type=int, default=12)
    collect.add_argument("--max-tasks", type=int)
    collect.add_argument("--allow-docs", action="store_true")
    collect.add_argument("--keep-worktrees", action="store_true")
    collect.add_argument(
        "--allow-baseline-pass",
        action="store_true",
        help="collect tasks even when verify_argv already passes before the repair",
    )

    prepare = sub.add_parser(
        "prepare",
        help="convert semantic episode JSON files/directories into transition JSONL",
    )
    prepare.add_argument("episodes", nargs="+")
    prepare.add_argument("--output", "-o", required=True)

    correct = sub.add_parser(
        "correct",
        help="relabel failed rollout states with historical verifier-backed corrective targets",
    )
    correct.add_argument("episodes", nargs="+")
    correct.add_argument("--tasks", required=True)
    correct.add_argument("--output", "-o", required=True)

    merge = sub.add_parser(
        "merge",
        help="merge recurrent transition corpora without changing their event schema",
    )
    merge.add_argument("transitions", nargs="+")
    merge.add_argument("--output", "-o", required=True)

    train = sub.add_parser(
        "train",
        help="train action/pointer/stop/value heads and optionally fine-tune RWKV",
    )
    train.add_argument("transitions")
    train.add_argument("--output-dir", "-o", required=True)
    _add_training_args(train)

    evaluate = sub.add_parser(
        "evaluate",
        help="replay a transition dataset from fresh recurrent states using a trained controller",
    )
    evaluate.add_argument("transitions")
    evaluate.add_argument("--controller", required=True)
    _add_runtime_args(evaluate)
    evaluate.add_argument("--output", "-o")

    improve = sub.add_parser(
        "improve",
        help="roll out, correct failures, retain verified successes, and retrain for N rounds",
    )
    improve.add_argument("tasks")
    improve.add_argument("--base-transitions", required=True)
    improve.add_argument("--controller", required=True)
    improve.add_argument("--output-root", required=True)
    improve.add_argument("--rounds", type=int, default=1)
    improve.add_argument("--model")
    improve.add_argument("--max-steps", type=int, default=12)
    improve.add_argument("--allow-docs", action="store_true")
    _add_training_args(improve, include_model_default=False)
    return parser


def _collection_progress(event: Mapping[str, Any]) -> None:
    index = event.get("index", "?")
    total = event.get("total", "?")
    task_id = event.get("task_id", "?")
    mode = event.get("mode", "?")
    if event.get("event") == "task-start":
        print(
            f"[lcfa] collect[{mode}] {index}/{total} {task_id} start",
            file=sys.stderr,
            flush=True,
        )
        return
    status = event.get("status", "unknown")
    success = event.get("success")
    steps = event.get("steps", 0)
    elapsed = float(event.get("elapsed_seconds", 0.0) or 0.0)
    print(
        f"[lcfa] collect[{mode}] {index}/{total} {task_id} done status={status} "
        f"success={success} steps={steps} elapsed={elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _training_progress(event: Mapping[str, Any]) -> None:
    kind = str(event.get("event") or "")
    if kind == "model-load-start":
        print(
            "[lcfa] train load-model "
            f"device={event.get('device')} dtype={event.get('dtype')} "
            f"mode={event.get('backbone_mode', 'frozen')} "
            f"pointers={event.get('pointer_slots', 0)} "
            f"train={event.get('train_transitions')} validation={event.get('validation_transitions')} "
            f"epochs={event.get('epochs')}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "model-loaded":
        print(
            f"[lcfa] train model-loaded elapsed={float(event.get('elapsed_seconds', 0.0)):.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "feature-cache-start":
        print(
            "[lcfa] train cache-features "
            f"episodes={event.get('episodes')} transitions={event.get('transitions')}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "feature-cache-progress":
        print(
            "[lcfa] train cache-features "
            f"episode={event.get('episode')}/{event.get('episodes')} "
            f"transitions={event.get('transitions')}/{event.get('total_transitions')} "
            f"tokens={event.get('tokens')} max_event_tokens={event.get('max_event_tokens')} "
            f"elapsed={float(event.get('elapsed_seconds', 0.0)):.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "feature-cache-done":
        print(
            "[lcfa] train features-ready "
            f"tokens={event.get('tokens')} max_event_tokens={event.get('max_event_tokens')} "
            f"elapsed={float(event.get('elapsed_seconds', 0.0)):.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "epoch-done":
        pointer = event.get("pointer_accuracy")
        pointer_text = "n/a" if pointer is None else f"{float(pointer):.3f}"
        print(
            "[lcfa] train "
            f"epoch={event.get('epoch')}/{event.get('epochs')} "
            f"loss={float(event.get('loss', 0.0)):.6f} "
            f"action_acc={float(event.get('action_accuracy', 0.0)):.3f} "
            f"pointer_acc={pointer_text} "
            f"stop_acc={float(event.get('stop_accuracy', 0.0)):.3f} "
            f"elapsed={float(event.get('elapsed_seconds', 0.0)):.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "evaluation-start":
        print(
            "[lcfa] train final-evaluation "
            f"train={event.get('train_transitions')} validation={event.get('validation_transitions')}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "evaluation-done":
        validation = event.get("validation_action_accuracy")
        validation_text = "n/a" if validation is None else f"{float(validation):.3f}"
        print(
            "[lcfa] train evaluation-done "
            f"train_action_acc={float(event.get('train_action_accuracy', 0.0)):.3f} "
            f"validation_action_acc={validation_text} "
            f"elapsed={float(event.get('elapsed_seconds', 0.0)):.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "training-done":
        print(
            "[lcfa] train done "
            f"updates={event.get('updates')} elapsed={float(event.get('elapsed_seconds', 0.0)):.1f}s "
            f"output={event.get('output')}",
            file=sys.stderr,
            flush=True,
        )


def _improvement_progress(event: Mapping[str, Any]) -> None:
    kind = str(event.get("event") or "")
    if kind == "round-start":
        print(
            f"[lcfa] improve round={event.get('round')}/{event.get('rounds')} start",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "round-data":
        print(
            "[lcfa] improve data "
            f"round={event.get('round')} successful={event.get('successful_transitions')} "
            f"corrective={event.get('corrective_transitions')} "
            f"total={event.get('accumulated_transitions')} "
            f"resolved={event.get('rollout_successful')}/{event.get('rollout_collected')} "
            f"errors={event.get('rollout_errors')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "round-done":
        rollout = event.get("rollout", {})
        if isinstance(rollout, Mapping):
            print(
                f"[lcfa] improve round={event.get('round')} done "
                f"success_rate={rollout.get('success_rate')} controller={event.get('output_controller')}",
                file=sys.stderr,
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "collect":
        if args.mode == "rollout" and not args.controller:
            parser.error("lcfa-recurrent collect --mode rollout requires --controller")
        summary = collect_trajectories(
            args.tasks,
            mode=args.mode,
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
            progress=_collection_progress,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    if args.command == "prepare":
        count = prepare_transition_file(args.episodes, args.output)
        if count == 0:
            Path(args.output).unlink(missing_ok=True)
            parser.error(
                "no recurrent transitions were produced; inspect collection.json for task errors before training"
            )
        print(json.dumps({"transitions": count, "output": args.output}, indent=2))
        return 0

    if args.command == "correct":
        count = prepare_corrective_transition_file(
            args.episodes, args.tasks, args.output
        )
        if count == 0:
            Path(args.output).unlink(missing_ok=True)
        print(json.dumps({"corrective_transitions": count, "output": args.output}, indent=2))
        return 0

    if args.command == "merge":
        count = merge_transition_files(args.transitions, args.output)
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

    if args.command == "improve":
        summary = improve_controller(
            args.tasks,
            base_transitions=args.base_transitions,
            controller=args.controller,
            output_root=args.output_root,
            rounds=args.rounds,
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            max_steps=args.max_steps,
            allow_docs=args.allow_docs,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            backbone_mode=args.backbone_mode,
            backbone_learning_rate=args.backbone_learning_rate,
            pointer_loss_weight=args.pointer_loss_weight,
            argument_loss_weight=args.argument_loss_weight,
            max_argument_chars=args.max_argument_chars,
            progress=_improvement_progress,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
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
        backbone_mode=args.backbone_mode,
        backbone_learning_rate=args.backbone_learning_rate,
        pointer_loss_weight=args.pointer_loss_weight,
        argument_loss_weight=args.argument_loss_weight,
        max_argument_chars=args.max_argument_chars,
        progress=_training_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
