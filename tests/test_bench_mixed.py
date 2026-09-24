from __future__ import annotations

import lcfa.bench_mixed_fixups  # applies audited mixed-harness helper fixups
from lcfa.bench_mixed import (
    MixedBenchmarkRunner,
    _case,
    _grade_choice,
    _grade_humaneval,
    _grade_numeric,
    _grade_ruler,
    _lcr_local_equivalent,
)
from lcfa.bench import BenchmarkSuite
from lcfa.protocol import SolutionState


def test_mixed_scalar_graders() -> None:
    assert _grade_numeric("42", "42").passed
    assert _grade_numeric("The answer is 1,234.5", "1234.5").passed
    assert _grade_choice("B", "B").passed
    assert _grade_choice("Answer: D", "D").passed
    assert _grade_ruler("alpha; beta", ["alpha", "beta"]).passed
    assert _lcr_local_equivalent("The result is 65.8%.", ["65.8%"])


def test_humaneval_executes_indented_completion() -> None:
    grade = _grade_humaneval(
        "    return 1",
        "def return1():\n",
        "def check(candidate):\n    assert candidate() == 1",
        "return1",
        timeout=5,
    )
    assert grade.passed


def test_case_does_not_duplicate_prompt_into_plan_metadata() -> None:
    prompt = "x" * 50_000
    case = _case(
        "lcr",
        "long",
        prompt,
        {"kind": "lcr", "question": "q", "references": ["a"]},
    )
    assert case.context.metadata["query"] == prompt
    assert "task" not in case.plan.metadata
    assert prompt not in repr(case.plan.metadata)


class _Subject:
    name = "fake-mixed"
    metadata = {"backend": "test"}

    def reason(self, plan, context):
        source = context.metadata["benchmark_source"]
        answer = {"gsm8k": "7", "mmlu": "C"}[source]
        return SolutionState(
            id=f"solution:{source}",
            plan_id=plan.id,
            values={},
            metadata={"language": answer},
        )


def test_mixed_runner_uses_standard_report_and_source_tags() -> None:
    suite = BenchmarkSuite(
        id="test-mixed",
        cases=(
            _case(
                "gsm8k",
                "one",
                "return 7",
                {"kind": "numeric", "reference": "7"},
            ),
            _case(
                "mmlu",
                "two",
                "return C",
                {"kind": "choice", "reference": "C"},
            ),
        ),
        metadata={"family": "mixed-real", "balanced": True},
    )
    report = MixedBenchmarkRunner().run(_Subject(), suite)
    assert report.format == "lcfa.bench.report.v1"
    assert report.summary.case_pass_rate == 1.0
    assert report.tag_summaries["gsm8k"].case_pass_rate == 1.0
    assert report.tag_summaries["mmlu"].case_pass_rate == 1.0
