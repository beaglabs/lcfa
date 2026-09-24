from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from lcfa import (
    ARTIFACT_FORMAT,
    BenchmarkRunner,
    BenchmarkSuite,
    EvidenceValue,
    ExecutionContext,
    LCFA,
    PlanNode,
    ReasonerSubject,
    ReasoningPlan,
    WEIGHTED_OUTPUT_ARCHITECTURE,
    all_suites,
    load_artifact_manifest,
    load_artifact_reasoner,
)
from lcfa.bench_cli import main as bench_main


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ARTIFACT = ROOT / "artifacts" / "lcfa-weighted-identity"


def _write_artifact(path: Path, *, scale: float, bias: float) -> Path:
    path.mkdir(parents=True)
    weights_path = path / "model.safetensors"
    save_file(
        {
            "numeric.scale": np.asarray([scale], dtype=np.float32),
            "numeric.bias": np.asarray([bias], dtype=np.float32),
        },
        str(weights_path),
        metadata={
            "format": ARTIFACT_FORMAT,
            "architecture": WEIGHTED_OUTPUT_ARCHITECTURE,
            "id": "test-weighted",
        },
    )
    digest = sha256(weights_path.read_bytes()).hexdigest()
    (path / "artifact.json").write_text(
        json.dumps(
            {
                "format": ARTIFACT_FORMAT,
                "id": "test-weighted",
                "version": "0.0.0-test",
                "architecture": WEIGHTED_OUTPUT_ARCHITECTURE,
                "base_backend": "lcfa-zero",
                "weights": {
                    "file": "model.safetensors",
                    "format": "safetensors",
                    "sha256": digest,
                },
                "tensors": {
                    "numeric.scale": {"shape": [1], "dtype": "F32"},
                    "numeric.bias": {"shape": [1], "dtype": "F32"},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_reference_artifact_manifest_and_weights_load() -> None:
    manifest = load_artifact_manifest(REFERENCE_ARTIFACT)
    assert manifest.format == ARTIFACT_FORMAT
    assert manifest.id == "lcfa-weighted-identity"
    assert manifest.architecture == WEIGHTED_OUTPUT_ARCHITECTURE

    reasoner = load_artifact_reasoner(REFERENCE_ARTIFACT)
    assert reasoner.numeric_scale == 1.0
    assert reasoner.numeric_bias == 0.0


def test_weight_backed_adapter_actually_applies_safetensors(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact", scale=2.0, bias=1.0)
    reasoner = load_artifact_reasoner(artifact)
    plan = ReasoningPlan(
        id="weighted-mean",
        nodes=(PlanNode("mean", "stats.mean", {"values": "$state.values"}),),
        outputs=("mean",),
    )
    solution = reasoner.reason(
        plan,
        ExecutionContext(state={"values": EvidenceValue([1.0, 3.0])}),
    )
    assert solution.values["mean"] == 5.0
    assert solution.metadata["artifact"]["artifact_id"] == "test-weighted"


def test_reference_weighted_artifact_passes_builtin_all() -> None:
    suites = all_suites()
    combined = BenchmarkSuite(
        id="lcfa-core-all",
        cases=tuple(case for suite in suites for case in suite.cases),
    )
    reasoner = load_artifact_reasoner(REFERENCE_ARTIFACT, base_engine=LCFA())
    report = BenchmarkRunner(repeats=2).run(
        ReasonerSubject(
            name=reasoner.artifact.id,
            reasoner=reasoner.reason,
            metadata=reasoner.metadata,
        ),
        combined,
    )
    assert report.summary.case_count == 17
    assert report.summary.case_pass_rate == 1.0
    assert report.summary.assertion_accuracy == 1.0
    assert report.summary.error_rate == 0.0
    assert report.summary.replay_determinism_rate == 1.0
    assert report.summary.evidence_f1 == 1.0


def test_bench_cli_runs_artifact_against_builtin_all(tmp_path: Path) -> None:
    output = tmp_path / "weighted.json"
    rc = bench_main(
        [
            "run",
            "builtin:all",
            "--artifact",
            str(REFERENCE_ARTIFACT),
            "--name",
            "lcfa-weighted-identity",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["subject_name"] == "lcfa-weighted-identity"
    assert report["summary"]["case_pass_rate"] == 1.0
