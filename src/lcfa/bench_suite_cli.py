"""CLI for materializing LCFA's frozen evaluation suites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bench import dumps_benchmark_suite
from .bench_suites import FROZEN_SUITES, build_frozen_suite, suite_catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcfa-bench-suite", description="Build frozen LCFA evaluation suites.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list frozen evaluation suites")

    build = sub.add_parser("build", help="materialize a frozen LCFA-native suite")
    build.add_argument("suite", choices=FROZEN_SUITES)
    build.add_argument("--cache-dir")
    build.add_argument("--output", "-o", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        print(json.dumps(suite_catalog(), indent=2))
        return 0

    suite = build_frozen_suite(args.suite, cache_dir=args.cache_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps_benchmark_suite(suite) + "\n", encoding="utf-8")
    print(f"wrote {len(suite.cases)} cases -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
