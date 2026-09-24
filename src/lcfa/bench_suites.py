"""Frozen evaluation-suite registry for LCFA system and milestone testing."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from .bench import BenchmarkSuite
from .bench_mixed import _case, _load_hf_rows, _swebench_cases

GSM8K_SYSTEM_240 = "gsm8k-system-240"
SWEBENCH_LITE = "swebench-lite"
ALE_LCFA_48 = "ale-lcfa-48"
FROZEN_SUITES = (GSM8K_SYSTEM_240, SWEBENCH_LITE, ALE_LCFA_48)
DEFAULT_SEED = 20260924

# Mirrored in benchmarks/suites/gsm8k-system-240.json so the membership is
# reviewable without executing Python.
GSM8K_SYSTEM_240_INDICES = (
    0,6,22,27,29,30,31,36,42,44,46,47,49,54,73,79,82,88,114,116,117,119,123,126,
    141,149,154,158,160,162,172,177,182,191,196,205,207,209,211,212,216,221,223,226,
    239,241,253,261,267,272,282,285,297,309,314,320,321,329,338,341,344,348,353,361,
    387,388,390,392,393,410,422,431,433,444,447,449,450,456,459,462,474,485,488,490,
    491,500,508,521,522,531,540,544,560,567,569,575,588,592,596,597,606,613,619,621,
    628,635,641,644,648,654,655,661,664,678,682,691,694,696,706,710,716,717,737,739,
    751,753,756,760,764,771,773,777,778,790,791,795,803,818,832,835,836,838,844,847,
    849,854,856,861,873,874,884,885,886,888,894,904,907,913,922,925,927,933,937,940,
    946,949,963,971,993,995,996,998,999,1000,1002,1005,1010,1011,1013,1022,1027,
    1028,1029,1030,1031,1037,1039,1053,1056,1057,1061,1075,1080,1082,1086,1093,
    1101,1103,1106,1111,1116,1119,1124,1127,1136,1143,1144,1149,1158,1164,1169,
    1172,1176,1177,1179,1183,1195,1197,1209,1210,1211,1214,1218,1222,1224,1237,
    1240,1244,1245,1252,1258,1284,1288,1289,1292,1294,1295,1299,1312,1317,
)


def _gsm8k_system_240(cache_dir: str | None = None) -> BenchmarkSuite:
    rows, split = _load_hf_rows("openai/gsm8k", config="main", cache_dir=cache_dir, preferred=("test",))
    if len(rows) != 1319:
        raise RuntimeError(f"GSM8K test split changed: expected 1319 rows, found {len(rows)}")
    cases = []
    for index in GSM8K_SYSTEM_240_INDICES:
        row = rows[index]
        answer_text = str(row["answer"])
        reference = answer_text.rsplit("####", 1)[-1].strip()
        prompt = (
            "Solve this grade-school math problem. Return only the final numeric answer "
            "with no explanation.\n\n" + str(row["question"])
        )
        cases.append(_case(
            "gsm8k",
            str(index),
            prompt,
            {"kind": "numeric", "reference": reference},
            metadata={"dataset_index": index, "frozen_suite": GSM8K_SYSTEM_240},
            tags=("math", "system-validation"),
        ))
    return BenchmarkSuite(
        id=GSM8K_SYSTEM_240,
        version="1",
        cases=tuple(cases),
        metadata={
            "family": "frozen-eval",
            "source": "openai/gsm8k",
            "split": split,
            "selection": "fixed-row-indices",
            "seed": DEFAULT_SEED,
            "evaluation_only": True,
        },
    )


def _swebench_lite(cache_dir: str | None = None) -> BenchmarkSuite:
    # The BM25-13K context dataset contains the complete Lite instance set while
    # preserving retrieval context used by the existing mixed benchmark runner.
    cases, split = _swebench_cases(300, random.Random(DEFAULT_SEED), cache_dir)
    if len(cases) != 300:
        raise RuntimeError(f"SWE-bench Lite changed: expected 300 cases, found {len(cases)}")
    return BenchmarkSuite(
        id=SWEBENCH_LITE,
        version="1",
        cases=tuple(cases),
        metadata={
            "family": "frozen-eval",
            "source": "SWE-bench/SWE-bench_Lite",
            "context_source": "princeton-nlp/SWE-bench_Lite_bm25_13K",
            "split": split,
            "selection": "all-rows",
            "evaluation_only": True,
            "official_grader_required": True,
        },
    )


def build_frozen_suite(name: str, *, cache_dir: str | None = None) -> BenchmarkSuite:
    if name == GSM8K_SYSTEM_240:
        return _gsm8k_system_240(cache_dir)
    if name == SWEBENCH_LITE:
        return _swebench_lite(cache_dir)
    if name == ALE_LCFA_48:
        raise ValueError(
            "ale-lcfa-48 is executed by Agents' Last Exam, not lcfa-bench; "
            "use benchmarks/suites/ale-lcfa-48.txt as ALE's selected_tasks file"
        )
    raise ValueError(f"unknown frozen suite {name!r}; choose from {', '.join(FROZEN_SUITES)}")


def suite_catalog() -> tuple[dict[str, Any], ...]:
    return (
        {"id": GSM8K_SYSTEM_240, "cases": 240, "runner": "lcfa", "tier": "system"},
        {"id": SWEBENCH_LITE, "cases": 300, "runner": "lcfa+swebench", "tier": "milestone"},
        {"id": ALE_LCFA_48, "cases": 48, "runner": "ale", "tier": "agent"},
    )


__all__ = [
    "ALE_LCFA_48", "FROZEN_SUITES", "GSM8K_SYSTEM_240", "GSM8K_SYSTEM_240_INDICES",
    "SWEBENCH_LITE", "build_frozen_suite", "suite_catalog",
]
