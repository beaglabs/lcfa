from __future__ import annotations

from pathlib import Path

from lcfa.bench_cli import main


def test_cli_reports_progress_to_stderr_without_corrupting_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "report.json"
    rc = main([
        "run",
        "builtin:retrieval",
        "--repeats",
        "1",
        "--output",
        str(output),
    ])
    captured = capsys.readouterr()

    assert rc == 0
    assert output.exists()
    assert captured.out == ""
    assert "[lcfa] loading" in captured.err
    assert "case 1/2" in captured.err
    assert "case 2/2" in captured.err
    assert "[lcfa] complete:" in captured.err


def test_cli_quiet_disables_progress(tmp_path: Path, capsys) -> None:
    output = tmp_path / "report.json"
    rc = main([
        "run",
        "builtin:retrieval",
        "--repeats",
        "1",
        "--quiet",
        "--output",
        str(output),
    ])
    captured = capsys.readouterr()

    assert rc == 0
    assert output.exists()
    assert captured.out == ""
    assert captured.err == ""
