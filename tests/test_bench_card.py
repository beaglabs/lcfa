from lcfa.bench_card import render_card


def test_render_card_contains_summary_and_failures() -> None:
    report = {
        "subject_name": "candidate",
        "suite_id": "gsm8k-system-240",
        "suite_version": "1",
        "summary": {
            "case_count": 240,
            "passed_cases": 180,
            "case_pass_rate": 0.75,
            "assertion_accuracy": 0.75,
            "error_rate": 0.01,
            "latency_mean_ms": 42.0,
            "latency_p95_ms": 80.0,
            "mean_trace_steps": 3.5,
        },
        "tag_summaries": {
            "math": {"case_count": 240, "case_pass_rate": 0.75},
        },
        "cases": [
            {"id": "ok", "passed": True},
            {"id": "bad", "passed": False, "error": "wrong answer"},
        ],
    }
    card = render_card(report)
    assert "LCFA Benchmark · gsm8k-system-240" in card
    assert "180 / 240" in card
    assert "75.0%" in card
    assert "`bad` — wrong answer" in card
