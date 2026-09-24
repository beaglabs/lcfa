"""Compatibility fixups for the mixed benchmark harness.

Kept separate so the fixes stay small and auditable while the mixed harness is
under active development. Importing this module patches the module-level helper
bindings used dynamically by build_mixed_suite() and MixedBenchmarkRunner.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Mapping, Sequence

from . import bench_mixed as _impl
from .bench import BenchmarkCase
from .protocol import SolutionState


_original_case = _impl._case


def _case(
    source: str,
    case_id: str,
    prompt: str,
    evaluation: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] = (),
) -> BenchmarkCase:
    """Build a case without duplicating the full prompt into plan metadata."""
    case = _original_case(
        source,
        case_id,
        prompt,
        evaluation,
        metadata=metadata,
        tags=tags,
    )
    return replace(
        case,
        plan=replace(case.plan, metadata={"benchmark_source": source}),
    )


def _language(solution: SolutionState) -> str:
    """Preserve leading code indentation while trimming boundary newlines."""
    value = solution.metadata.get("language", "") if solution.metadata else ""
    return str(value or "").strip("\r\n")


def _strip_fences(text: str) -> str:
    """Remove optional markdown fences without stripping Python indentation."""
    bounded = text.strip("\r\n")
    match = re.fullmatch(
        r"```(?:python|diff|patch)?[ \t]*\r?\n?(.*?)\r?\n?```",
        bounded,
        flags=re.I | re.S,
    )
    return match.group(1).strip("\r\n") if match else bounded


def _ruler_cases(
    count: int,
    rng: Any,
    cache_dir: str | None,
    context_length: int,
):
    """RULER builder that preserves the benchmark's answer_prefix field."""
    rows, split = _impl._load_hf_rows(
        _impl._SOURCE_DATASETS["ruler"],
        config=str(context_length),
        cache_dir=cache_dir,
    )
    indices = _impl._stratified_indices(rows, count, "task", rng)
    cases: list[BenchmarkCase] = []
    for index in indices:
        row = rows[index]
        answers = row.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        prefix = str(row.get("answer_prefix") or "")
        prompt = (
            "Use the supplied context to answer the question. Return only the requested "
            "answer value or values, without explanation.\n\n"
            f"{row['context']}\n\n{row['question']}\n{prefix}"
        )
        task = str(row.get("task", "unknown"))
        cases.append(
            _case(
                "ruler",
                f"{task}-{index}",
                prompt,
                {"kind": "ruler", "references": [str(item) for item in answers]},
                metadata={
                    "dataset_index": index,
                    "task": task,
                    "context_length": context_length,
                    "max_new_tokens": row.get("max_new_tokens"),
                },
                tags=("long-context", f"ruler:{task}"),
            )
        )
    return cases, split


# build_mixed_suite and MixedBenchmarkRunner resolve these names from the module
# globals at call time, so replacing the bindings fixes all public entry points
# without changing serialized suite/report formats.
_impl._case = _case
_impl._language = _language
_impl._strip_fences = _strip_fences
_impl._ruler_cases = _ruler_cases

# Re-export the supported mixed benchmark API.
DEFAULT_SEED = _impl.DEFAULT_SEED
MIXED_PRESETS = _impl.MIXED_PRESETS
MIXED_SOURCES = _impl.MIXED_SOURCES
MixedBenchmarkError = _impl.MixedBenchmarkError
MixedBenchmarkRunner = _impl.MixedBenchmarkRunner
build_mixed_suite = _impl.build_mixed_suite

__all__ = [
    "DEFAULT_SEED",
    "MIXED_PRESETS",
    "MIXED_SOURCES",
    "MixedBenchmarkError",
    "MixedBenchmarkRunner",
    "build_mixed_suite",
]
