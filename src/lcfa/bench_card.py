"""Render LCFA benchmark reports as GitHub Actions Job Summary Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _num(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{float(value):.1f}{suffix}"


def render_card(report: Mapping[str, Any], *, title: str | None = None) -> str:
    summary = dict(report.get("summary") or {})
    subject = str(report.get("subject_name") or "LCFA")
    suite = str(report.get("suite_id") or "benchmark")
    heading = title or f"LCFA Benchmark · {suite}"

    lines = [
        f"## {heading}",
        "",
        f"**Subject:** `{subject}`  ",
        f"**Suite:** `{suite}` · v{report.get('suite_version', '1')}",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Passed | {summary.get('passed_cases', 0)} / {summary.get('case_count', 0)} |",
        f"| Pass rate | **{_pct(summary.get('case_pass_rate'))}** |",
        f"| Error rate | {_pct(summary.get('error_rate'))} |",
        f"| Assertion accuracy | {_pct(summary.get('assertion_accuracy'))} |",
        f"| Mean latency | {_num(summary.get('latency_mean_ms'), ' ms')} |",
        f"| P95 latency | {_num(summary.get('latency_p95_ms'), ' ms')} |",
        f"| Mean trace steps | {_num(summary.get('mean_trace_steps'))} |",
        "",
    ]

    tags = report.get("tag_summaries") or {}
    if tags:
        lines.extend(["<details>", "<summary>Breakdown</summary>", "", "| Tag | Pass rate | Cases |", "|---|---:|---:|"])
        for tag, tag_summary in sorted(tags.items()):
            lines.append(
                f"| `{tag}` | {_pct(tag_summary.get('case_pass_rate'))} | {tag_summary.get('case_count', 0)} |"
            )
        lines.extend(["", "</details>", ""])

    failed = [case for case in report.get("cases", ()) if not case.get("passed", False)]
    if failed:
        lines.extend(["<details>", f"<summary>Failures ({len(failed)})</summary>", ""])
        for case in failed[:50]:
            detail = case.get("error") or "grader mismatch"
            lines.append(f"- `{case.get('id', 'unknown')}` — {detail}")
        if len(failed) > 50:
            lines.append(f"- …and {len(failed) - 50} more")
        lines.extend(["", "</details>", ""])

    return "\n".join(lines).rstrip() + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcfa-bench-card", description="Render an LCFA benchmark report as Markdown.")
    parser.add_argument("report", help="lcfa.bench.report.v1 JSON file")
    parser.add_argument("--title")
    parser.add_argument("--output", "-o", help="write Markdown to a file instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    text = render_card(report, title=args.title)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
