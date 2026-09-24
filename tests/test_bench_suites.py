from lcfa.bench_suites import (
    ALE_LCFA_48,
    FROZEN_SUITES,
    GSM8K_SYSTEM_240,
    GSM8K_SYSTEM_240_INDICES,
    SWEBENCH_LITE,
    suite_catalog,
)


def test_frozen_suite_registry_is_stable() -> None:
    assert FROZEN_SUITES == (GSM8K_SYSTEM_240, SWEBENCH_LITE, ALE_LCFA_48)
    assert len(GSM8K_SYSTEM_240_INDICES) == 240
    assert len(set(GSM8K_SYSTEM_240_INDICES)) == 240
    assert min(GSM8K_SYSTEM_240_INDICES) >= 0
    assert max(GSM8K_SYSTEM_240_INDICES) < 1319


def test_suite_catalog_declares_external_ale_runner() -> None:
    catalog = {item["id"]: item for item in suite_catalog()}
    assert catalog[GSM8K_SYSTEM_240]["cases"] == 240
    assert catalog[SWEBENCH_LITE]["cases"] == 300
    assert catalog[ALE_LCFA_48]["cases"] == 48
    assert catalog[ALE_LCFA_48]["runner"] == "ale"
