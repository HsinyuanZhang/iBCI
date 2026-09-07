"""Static/no-target contracts for the H1 D-S4/D-Q4 terminal package."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError
from scripts.h1_carrierid_distribution_terminal_checker import _validate_pair


ROOT = Path(__file__).resolve().parents[1]


def _metadata(prefix: str) -> dict[str, str]:
    shared = {
        "fold_date": "19250101",
        "normalizer_sha256": "n" * 64,
        "source_cache_sha256": "c" * 64,
        "normalized_cache_sha256": "z" * 64,
        "source_hashes_sha256": "h" * 64,
        "initial_state_sha256": "i" * 64,
        "source_schedule_sha256": "s" * 64,
        "common_query_samples_sha256": "q" * 64,
        "identity_schedule_sha256": "d" * 64,
        "batch_order_sha256": "b" * 64,
        "s4_effective_carriers_sha256": "4" * 64,
        "q4_effective_carriers_sha256": "5" * 64,
    }
    return {
        **shared,
        "source_manifest_sha256": prefix * 64,
        "effective_source_carriers_sha256": ("a" if prefix == "s" else "b") * 64,
    }


def test_static_terminal_preflight_never_imports_or_opens_target_data():
    source = (ROOT / "scripts/h1_carrierid_distribution_terminal_preflight.py").read_text(encoding="utf-8")
    assert "load_target_records" not in source
    assert "H1CarrierIdDistributionDataModule" not in source
    assert '"target_opened": False' in source
    assert '"cuda_launched": False' in source


def test_evaluator_binds_checker_before_first_target_load():
    source = (ROOT / "scripts/h1_carrierid_distribution_terminal_evaluate.py").read_text(encoding="utf-8")
    assert "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT" in source
    assert 'H1CarrierIdDistributionStrictTargetDataset(target, source.plan, source.normalizer, "s4")' in source
    assert 'H1CarrierIdDistributionStrictTargetDataset(target, source.plan, source.normalizer, "q4")' in source
    assert source.index("_require_checker_binding(") < source.index("target = load_target_records")
    assert source.index("source_closure = _verify_source_closure") < source.index("target = load_target_records")
    assert "optimizer_steps\": 0" in source
    assert "backward_steps\": 0" in source


def test_terminal_pair_requires_shared_everything_except_arm_manifest_and_carrier():
    s4, q4 = _metadata("s"), _metadata("q")
    result = _validate_pair(s4, q4)
    assert result["source_schedule_sha256"] == "s" * 64
    bad = dict(q4)
    bad["batch_order_sha256"] = "x" * 64
    with pytest.raises(NormalizedV2ContractError, match="batch_order_sha256"):
        _validate_pair(s4, bad)
    bad = dict(q4)
    bad["source_manifest_sha256"] = s4["source_manifest_sha256"]
    with pytest.raises(NormalizedV2ContractError, match="source manifests"):
        _validate_pair(s4, bad)


def test_terminal_preflight_hashes_target_evaluator_checker_and_contract_test():
    source = (ROOT / "scripts/h1_carrierid_distribution_terminal_preflight.py").read_text(encoding="utf-8")
    assert "h1_carrierid_distribution_terminal_checker.py" in source
    assert "h1_carrierid_distribution_terminal_evaluate.py" in source
    assert "h1_carrierid_distribution_target.py" in source
    assert "test_h1_carrierid_distribution_terminal_contract.py" in source
