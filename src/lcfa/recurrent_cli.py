"""CLI for LCFA recurrent-controller datasets and RWKV head training."""
from __future__ import annotations

import argparse
import json

from .recurrent_train import train_rwkv_heads
from .recurrent_transitions import prepare_transition_file
from .rwkv_controller import DEFAULT_RWKV_MODEL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcfa-recurrent",
        description="Prepare LCFA semantic trajectories and train recurrent RWKV policy heads.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
    train.add_argument("--seed", type=int, default=20260925)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        count = prepare_transition_file(args.episodes, args.output)
        print(json.dumps({"transitions": count, "output": args.output}, indent=2))
        return 0

    summary = train_rwkv_heads(
        args.transitions,
        args.output_dir,
        model_id=args.model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
