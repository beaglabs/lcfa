from __future__ import annotations

from lcfa import BenchmarkRunner, LCFA, ReasonerSubject, dumps_benchmark_suite, loads_benchmark_suite
from lcfa.bench_corpus import all_suites


def test_builtin_corpus_inventory_is_stable_and_unique() -> None:
    suites = all_suites()
    assert len(suites) == 8
    assert sum(len(suite.cases) for suite in suites) == 17
    assert len({suite.id for suite in suites}) == len(suites)

    case_ids = [case.id for suite in suites for case in suite.cases]
    assert len(set(case_ids)) == len(case_ids)
    assert all(case.tags for suite in suites for case in suite.cases)


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
