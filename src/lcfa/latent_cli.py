"""CLI for text-first LCFA latent-flow data preparation and training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

from .latent_encode import encode_examples, make_mlx_text_zplug
from .latent_train import (
    dump_examples_jsonl,
    examples_from_lines,
    load_examples_jsonl,
    train_mlx_latent_predictor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcfa-latent", description="Prepare and train LCFA latent-flow text models.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="mask a plain-text corpus into student/teacher latent pairs")
    prepare.add_argument("input", help="UTF-8 text file, one training document/example per line")
    prepare.add_argument("--output", "-o", required=True)
    prepare.add_argument("--mask-ratio", type=float, default=0.25)
    prepare.add_argument("--seed", type=int, default=0)

    encode = sub.add_parser("encode", help="freeze backbone enrichment features into safetensors")
    encode.add_argument("dataset", help="JSONL produced by `lcfa-latent prepare`")
    encode.add_argument("--model", required=True, help="local MLX-LM model directory")
    encode.add_argument("--output", "-o", required=True)
    encode.add_argument("--max-tokens", type=int, default=2048)
    encode.add_argument("--batch-size", type=int, default=2,
                        help="training examples per MLX forward batch (student+teacher doubles the sequence batch)")
    encode.add_argument("--pad-to", type=int, default=32,
                        help="right-pad token lengths to this bucket size to reduce MLX shape recompilation")
    encode.add_argument("--checkpoint-every", type=int, default=128,
                        help="atomically checkpoint this many completed examples for automatic resume")
    encode.add_argument("--no-resume", action="store_true",
                        help="discard any partial checkpoints/output and start encoding from zero")
    encode.add_argument("--quiet", action="store_true")

    train = sub.add_parser("train", help="train LCFA latent predictor from cached frozen features")
    train.add_argument("features", help="safetensors produced by `lcfa-latent encode`")
    train.add_argument("--output", "-o", required=True, help="artifact output directory")
    train.add_argument("--zplug-model-path", help="model path stored in artifact manifest for runtime enrichment")
    train.add_argument("--zplug-max-tokens", type=int, default=2048)
    train.add_argument("--latent-dim", type=int, default=256)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--ema-decay", type=float, default=0.99)
    train.add_argument("--variance-weight", type=float, default=0.05)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        lines = Path(args.input).read_text(encoding="utf-8").splitlines()
        examples = examples_from_lines(lines, mask_ratio=args.mask_ratio, seed=args.seed)
        path = dump_examples_jsonl(examples, args.output)
        print(f"prepared {len(examples)} examples -> {path}")
        return 0

    if args.command == "encode":
        examples = load_examples_jsonl(args.dataset)
        started = perf_counter()
        plug = make_mlx_text_zplug(args.model, max_tokens=args.max_tokens, pad_to=args.pad_to)

        def report(index: int, total: int, example_id: str) -> None:
            if not args.quiet:
                elapsed = perf_counter() - started
                rate = index / elapsed if elapsed > 0 else 0.0
                remaining = (total - index) / rate if rate > 0 else 0.0
                print(
                    f"[lcfa-latent] encode {index}/{total} {example_id} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={remaining:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )

        path = encode_examples(
            examples,
            plug,
            args.output,
            progress=report,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            resume=not args.no_resume,
        )
        print(str(path))
        return 0

    started = perf_counter()

    def training_progress(epoch: int, epochs: int, step: int, loss: float) -> None:
        if not args.quiet:
            elapsed = perf_counter() - started
            print(
                f"[lcfa-latent] epoch {epoch}/{epochs} step={step} loss={loss:.6f} elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    out = train_mlx_latent_predictor(
        args.features,
        args.output,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        variance_weight=args.variance_weight,
        seed=args.seed,
        zplug_type="mlx-text",
        zplug_model_path=args.zplug_model_path,
        zplug_max_tokens=args.zplug_max_tokens,
        progress=training_progress,
    )
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
