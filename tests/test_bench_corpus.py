from __future__ import annotations

import json
from pathlib import Path

from lcfa import (
    BenchmarkRunner,
    LCFA,
    ReasonerSubject,
    dumps_benchmark_suite,
    load_benchmark_report,
    load_benchmark_suite,
    loads_benchmark_suite,
)
from lcfa.bench_cli import main as bench_main
from lcfa.bench_corpus import all_suites


ROOT = Path(__file__).resolve().parents[1]


def test_builtin_corpus_inventory_is_stable_and_unique() -> None:
    suites = all_suites()
    assert len(suites) == 8
    assert sum(len(suite.cases) for suite in suites) == 17
    assert len({suite.id for suite in suites}) == len(suites)

    case_ids = [case.id for suite in suites for case in suite.cases]
    assert len(set(case_ids)) == len(case_ids)
    assert all(case.tags for suite in suites for case in suite.cases)


def test_manifest_freezes_builtin_corpus_inventory() -> None:
    manifest = json.loads((ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8"))
    suites = all_suites()

    assert manifest["format"] == "lcfa.bench.corpus.v1"
    assert manifest["case_count"] == sum(len(suite.cases) for suite in suites)
    assert [item["id"] for item in manifest["suites"]] == [suite.id for suite in suites]
    assert [item["cases"] for item in manifest["suites"]] == [
        [case.id for case in suite.cases]
        for suite in suites
    ]


def test_builtin_corpus_round_trips_through_benchmark_format() -> None:
    for suite in all_suites():
        restored = loads_benchmark_suite(dumps_benchmark_suite(suite))
        assert restored == suite


def test_lcfa_zero_passes_reference_corpus_with_perfect_correctness() -> None:
    engine = LCFA()
    subject = ReasonerSubject.from_engine("lcfa-zero", engine)
    runner = BenchmarkRunner(repeats=2)

    for suite in all_suites():
        report = runner.run(subject, suite)
        assert report.summary.case_pass_rate == 1.0, (suite.id, report)
        assert report.summary.assertion_accuracy == 1.0, (suite.id, report)
        assert report.summary.error_rate == 0.0, (suite.id, report)
        assert report.summary.replay_determinism_rate == 1.0, (suite.id, report)
        if report.summary.evidence_f1 is not None:
            assert report.summary.evidence_f1 == 1.0, (suite.id, report)


def test_retrieval_baseline_is_registered_by_default_runtime() -> None:
    names = set(LCFA().operators.names())
    assert "retrieval.rank" in names
    assert "retrieval.recall_at_k" in names


def test_cli_can_export_and_run_builtin_corpus(tmp_path: Path) -> None:
    exported = tmp_path / "temporal.json"
    assert bench_main(["export", "builtin:temporal", "-o", str(exported)]) == 0
    suite = load_benchmark_suite(exported)
    assert suite.id == "lcfa-core-temporal"
    assert len(suite.cases) == 3

    report_path = tmp_path / "all-zero.json"
    assert (
        bench_main(
            [
                "run",
                "builtin:all",
                "--name",
                "lcfa-zero",
                "--repeats",
                "1",
                "-o",
                str(report_path),
            ]
        )
        == 0
    )
    report = load_benchmark_report(report_path)
    assert report.suite_id == "lcfa-core-all"
    assert report.summary.case_count == 17
    assert report.summary.case_pass_rate == 1.0
    assert report.summary.assertion_accuracy == 1.0
    assert report.summary.error_rate == 0.0
