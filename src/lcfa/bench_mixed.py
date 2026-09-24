"""Mixed real-world benchmark support for LCFA-Bench.

This module materializes a balanced suite from GSM8K, HumanEval, SWE-bench Lite,
RULER, AA-LCR, and MMLU, then evaluates the language answer produced by a
stochastic LCFA subject with benchmark-appropriate graders.

Dataset/network dependencies are imported lazily so the core package and CI do
not require Hugging Face tooling.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import platform
import random
import re
import shlex
import subprocess
import sys
import tempfile
from time import perf_counter_ns
from typing import Any, Mapping, Sequence
from uuid import uuid4
import zipfile

from .bench import (
    AssertionResult,
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkSuite,
    CaseResult,
    _solution_signature,
    _summarize,
)
from .protocol import ExecutionContext, ReasoningPlan, SolutionState


MIXED_SOURCES = ("gsm8k", "humaneval", "swebench", "ruler", "lcr", "mmlu")
MIXED_PRESETS = {"quick": 3, "standard": 10, "large": 25}
DEFAULT_SEED = 20260923

_SOURCE_DATASETS = {
    "gsm8k": "openai/gsm8k",
    "humaneval": "openai/openai_humaneval",
    "swebench": "princeton-nlp/SWE-bench_Lite_bm25_13K",
    "ruler": "simonjegou/ruler",
    "lcr": "ArtificialAnalysis/AA-LCR",
    "mmlu": "cais/mmlu",
}


class MixedBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Grade:
    passed: bool
    expected: Any
    actual: Any
    detail: str | None = None


def _require_hf() -> tuple[Any, Any]:
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MixedBenchmarkError(
            "mixed suite construction requires Hugging Face tooling; "
            "install with: pip install -e '.[benchmarks]'"
        ) from exc
    return load_dataset, hf_hub_download


def _pick_split(dataset: Any, preferred: Sequence[str] = ("test", "validation", "train")) -> tuple[Any, str]:
    if hasattr(dataset, "keys"):
        keys = list(dataset.keys())
        for split in preferred:
            if split in keys:
                return dataset[split], split
        if keys:
            return dataset[keys[0]], str(keys[0])
    return dataset, "unknown"


def _load_hf_rows(
    dataset_id: str,
    *,
    config: str | None = None,
    cache_dir: str | None = None,
    preferred: Sequence[str] = ("test", "validation", "train"),
) -> tuple[Any, str]:
    load_dataset, _ = _require_hf()
    if config is None:
        dataset = load_dataset(dataset_id, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_id, config, cache_dir=cache_dir)
    return _pick_split(dataset, preferred)


def _sample_indices(length: int, count: int, rng: random.Random) -> list[int]:
    if count < 1:
        raise ValueError("case count must be >= 1")
    if length < count:
        raise MixedBenchmarkError(f"requested {count} cases from a split containing only {length}")
    return rng.sample(range(length), count)


def _stratified_indices(rows: Any, count: int, key: str, rng: random.Random) -> list[int]:
    groups: dict[str, list[int]] = {}
    for index in range(len(rows)):
        row = rows[index]
        value = str(row.get(key, "unknown"))
        groups.setdefault(value, []).append(index)
    if len(rows) < count:
        raise MixedBenchmarkError(f"requested {count} cases from a split containing only {len(rows)}")
    keys = sorted(groups)
    for value in keys:
        rng.shuffle(groups[value])
    rng.shuffle(keys)
    selected: list[int] = []
    cursor = 0
    while len(selected) < count:
        key_value = keys[cursor % len(keys)]
        bucket = groups[key_value]
        if bucket:
            selected.append(bucket.pop())
        cursor += 1
        if cursor > count * max(2, len(keys)) * 4 and len(selected) < count:
            remaining = [item for bucket in groups.values() for item in bucket]
            rng.shuffle(remaining)
            selected.extend(remaining[: count - len(selected)])
            break
    return selected[:count]


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return text[:100] or "case"


def _case(
    source: str,
    case_id: str,
    prompt: str,
    evaluation: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] = (),
) -> BenchmarkCase:
    plan = ReasoningPlan(
        id=f"mixed-{source}-{_slug(case_id)}",
        nodes=(),
        outputs=(),
        metadata={"benchmark_source": source, "task": prompt},
    )
    context = ExecutionContext(
        metadata={
            "query": prompt,
            "benchmark_source": source,
            "benchmark_case_id": str(case_id),
        }
    )
    return BenchmarkCase(
        id=f"mixed.{source}.{_slug(case_id)}",
        plan=plan,
        context=context,
        tags=tuple(dict.fromkeys(("mixed", source, *tags))),
        metadata={
            "source": source,
            "evaluation": dict(evaluation),
            **dict(metadata or {}),
        },
    )


def _gsm8k_cases(count: int, rng: random.Random, cache_dir: str | None) -> tuple[list[BenchmarkCase], str]:
    rows, split = _load_hf_rows(_SOURCE_DATASETS["gsm8k"], config="main", cache_dir=cache_dir)
    cases: list[BenchmarkCase] = []
    for index in _sample_indices(len(rows), count, rng):
        row = rows[index]
        answer_text = str(row["answer"])
        reference = answer_text.rsplit("####", 1)[-1].strip()
        prompt = (
            "Solve this grade-school math problem. Return only the final numeric answer "
            "with no explanation.\n\n"
            f"{row['question']}"
        )
        cases.append(
            _case(
                "gsm8k",
                str(index),
                prompt,
                {"kind": "numeric", "reference": reference},
                metadata={"dataset_index": index},
                tags=("math",),
            )
        )
    return cases, split


def _humaneval_cases(count: int, rng: random.Random, cache_dir: str | None) -> tuple[list[BenchmarkCase], str]:
    rows, split = _load_hf_rows(_SOURCE_DATASETS["humaneval"], cache_dir=cache_dir)
    cases: list[BenchmarkCase] = []
    for index in _sample_indices(len(rows), count, rng):
        row = rows[index]
        task_id = str(row.get("task_id", index))
        prompt = (
            "Complete the following Python function. Return only the Python implementation "
            "or completion, without Markdown fences or explanation.\n\n"
            f"{row['prompt']}"
        )
        cases.append(
            _case(
                "humaneval",
                task_id,
                prompt,
                {
                    "kind": "humaneval",
                    "prompt": str(row["prompt"]),
                    "test": str(row["test"]),
                    "entry_point": str(row["entry_point"]),
                },
                metadata={"dataset_index": index, "task_id": task_id},
                tags=("coding", "execution"),
            )
        )
    return cases, split


def _swebench_cases(count: int, rng: random.Random, cache_dir: str | None) -> tuple[list[BenchmarkCase], str]:
    rows, split = _load_hf_rows(_SOURCE_DATASETS["swebench"], cache_dir=cache_dir)
    cases: list[BenchmarkCase] = []
    for index in _sample_indices(len(rows), count, rng):
        row = rows[index]
        instance_id = str(row["instance_id"])
        text = str(row.get("text") or row.get("problem_statement") or "")
        prompt = (
            f"{text}\n\n"
            "Produce the repository patch that resolves the issue. Return only a unified diff "
            "(git patch), without explanation or Markdown fences."
        )
        cases.append(
            _case(
                "swebench",
                instance_id,
                prompt,
                {"kind": "swebench", "instance_id": instance_id},
                metadata={
                    "dataset_index": index,
                    "instance_id": instance_id,
                    "repo": row.get("repo"),
                    "base_commit": row.get("base_commit"),
                    "context_profile": "bm25-13k",
                },
                tags=("coding", "execution", "long-context"),
            )
        )
    return cases, split


def _ruler_cases(
    count: int,
    rng: random.Random,
    cache_dir: str | None,
    context_length: int,
) -> tuple[list[BenchmarkCase], str]:
    rows, split = _load_hf_rows(
        _SOURCE_DATASETS["ruler"],
        config=str(context_length),
        cache_dir=cache_dir,
    )
    indices = _stratified_indices(rows, count, "task", rng)
    cases: list[BenchmarkCase] = []
    for index in indices:
        row = rows[index]
        answers = row.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        prompt = (
            f"{row['context']}\n\n{row['question']}\n\n"
            "Return only the requested answer value or values, without explanation."
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


def _find_lcr_member(names: Sequence[str], category: str, set_id: str, filename: str) -> str:
    normalized = filename.replace("\\", "/").lstrip("/")
    suffix = f"/{category}/{set_id}/{normalized}"
    matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
    if not matches:
        direct = f"{category}/{set_id}/{normalized}"
        matches = [name for name in names if name.replace("\\", "/").endswith(direct)]
    if len(matches) != 1:
        raise MixedBenchmarkError(
            f"could not uniquely locate AA-LCR document {category}/{set_id}/{filename!r}; "
            f"matches={len(matches)}"
        )
    return matches[0]


def _lcr_cases(count: int, rng: random.Random, cache_dir: str | None) -> tuple[list[BenchmarkCase], str]:
    rows, split = _load_hf_rows(_SOURCE_DATASETS["lcr"], cache_dir=cache_dir)
    _, hf_hub_download = _require_hf()
    archive_path = hf_hub_download(
        repo_id=_SOURCE_DATASETS["lcr"],
        filename="extracted_text/AA-LCR_extracted-text.zip",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    indices = _stratified_indices(rows, count, "document_category", rng)
    cases: list[BenchmarkCase] = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for index in indices:
            row = rows[index]
            category = str(row["document_category"])
            set_id = str(row["document_set_id"])
            filenames = [item for item in str(row["data_source_filenames"]).split(";") if item]
            docs: list[str] = []
            for filename in filenames:
                member = _find_lcr_member(names, category, set_id, filename)
                docs.append(archive.read(member).decode("utf-8"))
            documents_text = "\n\n".join(
                f"BEGIN DOCUMENT {doc_index + 1}:\n{doc}\nEND DOCUMENT {doc_index + 1}"
                for doc_index, doc in enumerate(docs)
            )
            prompt = (
                "BEGIN INPUT DOCUMENTS\n\n"
                f"{documents_text}\n\n"
                "END INPUT DOCUMENTS\n\n"
                "Answer the following question using the input documents provided above. "
                "Return only the answer, without explanation.\n\n"
                "START QUESTION\n\n"
                f"{row['question']}\n\n"
                "END QUESTION"
            )
            references = [item.strip() for item in str(row["answer"]).split(";") if item.strip()]
            question_id = str(row.get("question_id", index))
            cases.append(
                _case(
                    "lcr",
                    question_id,
                    prompt,
                    {
                        "kind": "lcr",
                        "question": str(row["question"]),
                        "references": references,
                    },
                    metadata={
                        "dataset_index": index,
                        "question_id": question_id,
                        "document_category": category,
                        "document_set_id": set_id,
                        "input_tokens": int(row.get("input_tokens", 0) or 0),
                        "scoring_version": "AA-LCR-1.1",
                    },
                    tags=("long-context", f"lcr:{_slug(category)}"),
                )
            )
    return cases, split


def _mmlu_cases(count: int, rng: random.Random, cache_dir: str | None) -> tuple[list[BenchmarkCase], str]:
    rows, split = _load_hf_rows(_SOURCE_DATASETS["mmlu"], config="all", cache_dir=cache_dir)
    indices = _stratified_indices(rows, count, "subject", rng)
    letters = "ABCD"
    cases: list[BenchmarkCase] = []
    for index in indices:
        row = rows[index]
        choices = list(row["choices"])
        subject = str(row.get("subject", "unknown"))
        options = "\n".join(f"{letters[i]}. {choice}" for i, choice in enumerate(choices))
        prompt = (
            f"Answer this multiple-choice question from {subject}. Return only one letter: A, B, C, or D.\n\n"
            f"{row['question']}\n{options}"
        )
        expected = letters[int(row["answer"])]
        cases.append(
            _case(
                "mmlu",
                f"{subject}-{index}",
                prompt,
                {"kind": "choice", "reference": expected},
                metadata={"dataset_index": index, "subject": subject},
                tags=("knowledge", f"mmlu:{_slug(subject)}"),
            )
        )
    return cases, split


def build_mixed_suite(
    *,
    preset: str = "quick",
    cases_per_source: int | None = None,
    seed: int = DEFAULT_SEED,
    ruler_context: int = 4096,
    cache_dir: str | None = None,
) -> BenchmarkSuite:
    """Materialize a balanced real-world benchmark into an ordinary LCFA suite."""
    if preset not in MIXED_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {', '.join(MIXED_PRESETS)}")
    count = int(cases_per_source or MIXED_PRESETS[preset])
    if count < 1:
        raise ValueError("cases_per_source must be >= 1")
    if ruler_context not in {4096, 8192, 16384}:
        raise ValueError("ruler_context must be one of 4096, 8192, 16384")

    rng = random.Random(seed)
    builders = (
        ("gsm8k", lambda: _gsm8k_cases(count, rng, cache_dir)),
        ("humaneval", lambda: _humaneval_cases(count, rng, cache_dir)),
        ("swebench", lambda: _swebench_cases(count, rng, cache_dir)),
        ("ruler", lambda: _ruler_cases(count, rng, cache_dir, ruler_context)),
        ("lcr", lambda: _lcr_cases(count, rng, cache_dir)),
        ("mmlu", lambda: _mmlu_cases(count, rng, cache_dir)),
    )
    cases: list[BenchmarkCase] = []
    splits: dict[str, str] = {}
    for source, builder in builders:
        source_cases, split = builder()
        if len(source_cases) != count:
            raise MixedBenchmarkError(
                f"{source} yielded {len(source_cases)} cases; balanced suite requires exactly {count}"
            )
        cases.extend(source_cases)
        splits[source] = split

    rng.shuffle(cases)
    return BenchmarkSuite(
        id=f"lcfa-mixed-real-{preset}-{count}x{len(MIXED_SOURCES)}",
        version="1",
        cases=tuple(cases),
        metadata={
            "family": "mixed-real",
            "preset": preset,
            "cases_per_source": count,
            "source_count": len(MIXED_SOURCES),
            "balanced": True,
            "seed": seed,
            "ruler_context": ruler_context,
            "datasets": dict(_SOURCE_DATASETS),
            "splits": splits,
            "lcr_scoring": "AA-LCR 1.1 local-equivalence unless --lcr-judge-cmd is supplied",
            "swebench_scoring": "official SWE-bench Docker harness",
            "humaneval_scoring": "native tests in isolated Python subprocess",
        },
    )


def _language(solution: SolutionState) -> str:
    value = solution.metadata.get("language", "") if solution.metadata else ""
    return str(value or "").strip()


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:python|diff|patch)?\s*\n?(.*?)\n?```", stripped, flags=re.I | re.S)
    return match.group(1).strip() if match else stripped


def _decimal_from_text(text: str) -> Decimal | None:
    tokens = re.findall(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not tokens:
        return None
    try:
        return Decimal(tokens[-1].replace(",", ""))
    except InvalidOperation:
        return None


def _grade_numeric(candidate: str, reference: str) -> Grade:
    actual = _decimal_from_text(candidate)
    expected = _decimal_from_text(reference)
    passed = actual is not None and expected is not None and actual == expected
    return Grade(passed, reference, str(actual) if actual is not None else candidate)


def _grade_choice(candidate: str, reference: str) -> Grade:
    text = candidate.strip().upper()
    exact = text if text in {"A", "B", "C", "D"} else None
    if exact is None:
        matches = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", text)
        exact = matches[-1] if matches else ""
    return Grade(exact == reference.upper(), reference.upper(), exact or candidate)


def _normalize_text(value: str) -> str:
    value = value.casefold()
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\s\.,;:!?]+$", "", value)
    return value.strip()


def _grade_ruler(candidate: str, references: Sequence[str]) -> Grade:
    normalized_candidate = _normalize_text(candidate)
    normalized_refs = [_normalize_text(item) for item in references if _normalize_text(item)]
    passed = bool(normalized_refs) and all(item in normalized_candidate for item in normalized_refs)
    return Grade(passed, list(references), candidate)


def _lcr_local_equivalent(candidate: str, references: Sequence[str]) -> bool:
    candidate_norm = _normalize_text(candidate)
    if not references:
        return False
    for reference in references:
        ref_norm = _normalize_text(reference)
        if ref_norm and ref_norm in candidate_norm:
            continue
        candidate_number = _decimal_from_text(candidate)
        reference_number = _decimal_from_text(reference)
        if candidate_number is not None and reference_number is not None and candidate_number == reference_number:
            continue
        return False
    return True


def _grade_lcr(
    candidate: str,
    question: str,
    references: Sequence[str],
    judge_cmd: str | None,
) -> Grade:
    if judge_cmd:
        payload = json.dumps(
            {"question": question, "reference": list(references), "candidate": candidate}
        )
        process = subprocess.run(
            shlex.split(judge_cmd),
            input=payload,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if process.returncode != 0:
            raise MixedBenchmarkError(
                f"LCR judge command failed ({process.returncode}): {process.stderr.strip()}"
            )
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise MixedBenchmarkError("LCR judge command did not return JSON") from exc
        if not isinstance(result, Mapping) or "passed" not in result:
            raise MixedBenchmarkError("LCR judge JSON must contain boolean field 'passed'")
        return Grade(bool(result["passed"]), list(references), candidate, "external-judge")
    return Grade(
        _lcr_local_equivalent(candidate, references),
        list(references),
        candidate,
        "local-equivalence; not official AA-LCR LLM-judge parity",
    )


def _grade_humaneval(
    candidate: str,
    prompt: str,
    test: str,
    entry_point: str,
    *,
    timeout: int,
) -> Grade:
    completion = _strip_fences(candidate)
    has_full_def = bool(re.search(rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\(", completion))
    source = completion if has_full_def else prompt + completion
    program = f"{source}\n\n{test}\n\ncheck({entry_point})\n"
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryDirectory(prefix="lcfa-humaneval-") as workdir:
        try:
            process = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=workdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Grade(False, "HumanEval tests pass", "timeout", f"timeout>{timeout}s")
    detail = (process.stderr or process.stdout).strip()
    if len(detail) > 1000:
        detail = detail[-1000:]
    return Grade(
        process.returncode == 0,
        "HumanEval tests pass",
        "pass" if process.returncode == 0 else "fail",
        detail or None,
    )


def _extract_patch(candidate: str) -> str:
    text = _strip_fences(candidate)
    match = re.search(r"<patch>\s*(.*?)\s*</patch>", text, flags=re.I | re.S)
    if match:
        text = match.group(1).strip()
    start_markers = ("diff --git ", "--- a/", "*** Begin Patch")
    starts = [text.find(marker) for marker in start_markers if text.find(marker) >= 0]
    if starts:
        text = text[min(starts):]
    return text.strip()


def _grade_swebench(candidate: str, instance_id: str, *, timeout: int) -> Grade:
    patch = _extract_patch(candidate)
    if not patch:
        return Grade(False, "resolved", "empty patch", "no patch generated")
    try:
        from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR
        from swebench.harness.run_evaluation import main as run_evaluation
    except ImportError as exc:
        raise MixedBenchmarkError(
            "SWE-bench grading requires the official swebench package and Docker"
        ) from exc

    run_id = f"lcfa-{_slug(instance_id)}-{uuid4().hex[:10]}"
    model_name = "lcfa-mixed"
    prediction = {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": model_name,
    }
    with tempfile.TemporaryDirectory(prefix="lcfa-swebench-") as workdir:
        predictions_path = Path(workdir) / "predictions.json"
        predictions_path.write_text(json.dumps([prediction]), encoding="utf-8")
        run_evaluation(
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            instance_ids=[instance_id],
            predictions_path=str(predictions_path),
            max_workers=1,
            open_file_limit=4096,
            run_id=run_id,
            timeout=timeout,
            rewrite_reports=False,
            modal=False,
            task_repo=None,
        )

    report_path = (
        Path(RUN_EVALUATION_LOG_DIR)
        / run_id
        / model_name.replace("/", "__")
        / instance_id
        / LOG_REPORT
    )
    if not report_path.exists():
        raise MixedBenchmarkError(
            f"SWE-bench did not produce a report for {instance_id}; inspect run_id={run_id}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved = bool(report.get(instance_id, {}).get("resolved", False))
    return Grade(resolved, "resolved", "resolved" if resolved else "unresolved", f"run_id={run_id}")


def _truncate_display(value: Any, limit: int = 2000) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + f"...<truncated {len(value) - limit} chars>"


class MixedBenchmarkRunner:
    """Benchmark runner with native graders for the six mixed-source families."""

    def __init__(
        self,
        *,
        repeats: int = 1,
        warmup: int = 0,
        allow_code_exec: bool = False,
        swebench_eval: bool = False,
        code_timeout: int = 10,
        swebench_timeout: int = 1800,
        lcr_judge_cmd: str | None = None,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        self.repeats = repeats
        self.warmup = warmup
        self.allow_code_exec = allow_code_exec
        self.swebench_eval = swebench_eval
        self.code_timeout = max(1, int(code_timeout))
        self.swebench_timeout = max(1, int(swebench_timeout))
        self.lcr_judge_cmd = lcr_judge_cmd

    def _validate(self, suite: BenchmarkSuite) -> None:
        sources = {str(case.metadata.get("source", "")) for case in suite.cases}
        if "humaneval" in sources and not self.allow_code_exec:
            raise MixedBenchmarkError(
                "suite contains HumanEval; pass --allow-code-exec to execute its native tests"
            )
        if "swebench" in sources and not self.swebench_eval:
            raise MixedBenchmarkError(
                "suite contains SWE-bench; pass --swebench-eval to use the official Docker harness"
            )

    def run(self, subject: Any, suite: BenchmarkSuite) -> BenchmarkReport:
        self._validate(suite)
        results = tuple(self._run_case(subject, case) for case in suite.cases)
        tags = sorted({tag for result in results for tag in result.tags})
        return BenchmarkReport(
            subject_name=subject.name,
            suite_id=suite.id,
            suite_version=suite.version,
            summary=_summarize(results),
            cases=results,
            tag_summaries={
                tag: _summarize(tuple(result for result in results if tag in result.tags))
                for tag in tags
            },
            subject_metadata=dict(subject.metadata),
            environment={
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "mixed_runner": "lcfa.mixed.v1",
                "lcr_judge": self.lcr_judge_cmd or "local-equivalence",
                "swebench_eval": self.swebench_eval,
                "humaneval_exec": self.allow_code_exec,
            },
        )

    def _grade(self, solution: SolutionState, case: BenchmarkCase) -> Grade:
        candidate = _language(solution)
        config = case.metadata.get("evaluation", {})
        if not isinstance(config, Mapping):
            raise MixedBenchmarkError(f"{case.id}: evaluation metadata must be a mapping")
        kind = str(config.get("kind", ""))
        if kind == "numeric":
            return _grade_numeric(candidate, str(config["reference"]))
        if kind == "choice":
            return _grade_choice(candidate, str(config["reference"]))
        if kind == "ruler":
            return _grade_ruler(candidate, [str(item) for item in config.get("references", [])])
        if kind == "lcr":
            return _grade_lcr(
                candidate,
                str(config.get("question", "")),
                [str(item) for item in config.get("references", [])],
                self.lcr_judge_cmd,
            )
        if kind == "humaneval":
            return _grade_humaneval(
                candidate,
                str(config["prompt"]),
                str(config["test"]),
                str(config["entry_point"]),
                timeout=self.code_timeout,
            )
        if kind == "swebench":
            return _grade_swebench(
                candidate,
                str(config["instance_id"]),
                timeout=self.swebench_timeout,
            )
        raise MixedBenchmarkError(f"{case.id}: unknown evaluator kind {kind!r}")

    def _run_case(self, subject: Any, case: BenchmarkCase) -> CaseResult:
        for _ in range(self.warmup):
            try:
                subject.reason(case.plan, case.context)
            except Exception as exc:
                return CaseResult(
                    id=case.id,
                    passed=False,
                    assertion_passes=0,
                    assertion_total=0,
                    evidence_tp=0,
                    evidence_fp=0,
                    evidence_fn=0,
                    evidence_checked=False,
                    evidence_passed=None,
                    replay_deterministic=None,
                    repeats_requested=self.repeats,
                    repeats_completed=0,
                    latencies_ms=(),
                    trace_steps=(),
                    error=f"{type(exc).__name__}: {exc}",
                    tags=case.tags,
                )

        solutions: list[SolutionState] = []
        latencies: list[float] = []
        trace_steps: list[int] = []
        assertion_passes = 0
        assertions: tuple[AssertionResult, ...] = ()
        error: str | None = None

        for repeat_index in range(self.repeats):
            started = perf_counter_ns()
            try:
                solution = subject.reason(case.plan, case.context)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            latencies.append((perf_counter_ns() - started) / 1_000_000.0)
            solutions.append(solution)
            trace_steps.append(len(solution.trace.steps) if solution.trace is not None else 0)
            try:
                grade = self._grade(solution, case)
            except Exception as exc:
                error = f"evaluation {type(exc).__name__}: {exc}"
                if repeat_index == 0:
                    assertions = (
                        AssertionResult(
                            path="metadata.language",
                            passed=False,
                            expected="native evaluator completed",
                            actual=_truncate_display(_language(solution)),
                            abs_tol=0.0,
                            rel_tol=0.0,
                        ),
                    )
                break
            assertion_passes += int(grade.passed)
            if repeat_index == 0:
                actual = grade.actual
                if grade.detail:
                    actual = {"answer": _truncate_display(actual), "detail": grade.detail}
                assertions = (
                    AssertionResult(
                        path="metadata.language",
                        passed=grade.passed,
                        expected=_truncate_display(grade.expected),
                        actual=_truncate_display(actual),
                        abs_tol=0.0,
                        rel_tol=0.0,
                    ),
                )

        deterministic: bool | None
        if not solutions:
            deterministic = None
        elif len(solutions) == 1:
            deterministic = True
        else:
            deterministic = len({_solution_signature(item) for item in solutions}) == 1

        completed_all = len(solutions) == self.repeats and error is None
        assertion_total = len(solutions) if error is None else max(1, len(solutions))
        passed = completed_all and assertion_passes == self.repeats

        return CaseResult(
            id=case.id,
            passed=passed,
            assertion_passes=assertion_passes,
            assertion_total=assertion_total,
            evidence_tp=0,
            evidence_fp=0,
            evidence_fn=0,
            evidence_checked=False,
            evidence_passed=None,
            replay_deterministic=deterministic,
            repeats_requested=self.repeats,
            repeats_completed=len(solutions),
            latencies_ms=tuple(latencies),
            trace_steps=tuple(trace_steps),
            assertions=assertions,
            error=error,
            tags=case.tags,
        )


__all__ = [
    "DEFAULT_SEED",
    "MIXED_PRESETS",
    "MIXED_SOURCES",
    "MixedBenchmarkError",
    "MixedBenchmarkRunner",
    "build_mixed_suite",
]
