"""Static/no-target contracts for the H1 D-S4e/D-Q4e terminal package."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET
from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError
from scripts.h1_carrierid_distribution_exposure_terminal_checker import _require_finite_state_dict, _validate_pair
from scripts.h1_carrierid_distribution_exposure_terminal_evaluate import _interpret, _require_preflight_binding
from scripts.h1_carrierid_distribution_exposure_terminal_preflight import frozen_terminal_branch_gates
from src.h1_m4_eb_normalized_v2_contract import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _metadata(prefix: str) -> dict[str, object]:
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
        "exposure_sample_plan": {"target_samples_per_epoch": 115_520, "target_batches_per_epoch": 3_610},
    }
    return {**shared, "source_manifest_sha256": prefix * 64}


def _metrics(first: float, second: float, pooled: float) -> dict[str, object]:
    return {
        "pooled_r2": pooled,
        "per_session": {
            H1_M4_FOLD0_TARGET[0]: {"r2": first},
            H1_M4_FOLD0_TARGET[1]: {"r2": second},
        },
    }


def test_static_exposure_terminal_preflight_never_imports_or_opens_target_data():
    source = (ROOT / "scripts/h1_carrierid_distribution_exposure_terminal_preflight.py").read_text(encoding="utf-8")
    assert "load_target_records" not in source
    assert "H1CarrierIdDistributionExposureDataModule" not in source
    assert '"target_opened": False' in source
    assert '"cuda_launched": False' in source


def test_exposure_evaluator_binds_checker_and_source_before_first_target_load():
    source = (ROOT / "scripts/h1_carrierid_distribution_exposure_terminal_evaluate.py").read_text(encoding="utf-8")
    assert "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT" in source
    assert 'H1CarrierIdDistributionStrictTargetDataset(target, source.plan, source.normalizer, "s4")' in source
    assert 'H1CarrierIdDistributionStrictTargetDataset(target, source.plan, source.normalizer, "q4")' in source
    assert source.index("_require_checker_binding(") < source.index("target = load_target_records")
    assert source.index("source_closure = _verify_source_closure") < source.index("target = load_target_records")
    assert source.index("source = _rebuild_s4e_source") < source.index("target = load_target_records")
    assert '"optimizer_steps": 0' in source
    assert '"backward_steps": 0' in source


def test_exposure_pair_requires_common_init_and_exposure_plan():
    s4, q4 = _metadata("s"), _metadata("q")
    result = _validate_pair(s4, q4)
    assert result["exposure_sample_plan"]["target_samples_per_epoch"] == 115_520
    bad = dict(q4)
    bad["batch_order_sha256"] = "x" * 64
    with pytest.raises(NormalizedV2ContractError, match="batch_order_sha256"):
        _validate_pair(s4, bad)
    bad = dict(q4)
    bad["source_manifest_sha256"] = s4["source_manifest_sha256"]
    with pytest.raises(NormalizedV2ContractError, match="source manifests"):
        _validate_pair(s4, bad)


def test_terminal_checker_rejects_nan_inf_and_non_tensor_state_entries():
    _require_finite_state_dict({"finite": torch.tensor([0.0, 1.0])}, "s4")
    with pytest.raises(NormalizedV2ContractError, match="nonfinite tensor 'nan'"):
        _require_finite_state_dict({"nan": torch.tensor([float("nan")])}, "s4")
    with pytest.raises(NormalizedV2ContractError, match="nonfinite tensor 'inf'"):
        _require_finite_state_dict({"inf": torch.tensor([float("inf")])}, "q4")
    with pytest.raises(NormalizedV2ContractError, match="non-tensor 'unexpected'"):
        _require_finite_state_dict({"unexpected": 7}, "q4")


def test_validity_gate_precedes_and_can_block_leakage_delta_interpretation():
    gates = frozen_terminal_branch_gates()
    invalid = _interpret(gates, _metrics(0.8, -0.1, 0.60), _metrics(0.9, 0.0, 0.70))
    assert invalid["decision"] == "invalid_exposure_repair__q4e_minus_s4e_not_interpretable"
    estimator = _interpret(gates, _metrics(0.5, 0.51, 0.505), _metrics(0.54, 0.55, 0.545))
    assert estimator["decision"] == "estimator-limited evidence"
    consumer = _interpret(gates, _metrics(0.5, 0.51, 0.505), _metrics(0.51, 0.52, 0.515))
    assert consumer["decision"] == "consumer-limited evidence"


def test_evaluator_rejects_checker_from_a_different_terminal_preflight_or_source_closure(tmp_path):
    preflight = tmp_path / "terminal_preflight.json"
    preflight.write_text('{"static": true}\n', encoding="utf-8")
    closure = {"scripts/frozen.py": "a" * 64}
    checker = {
        "terminal_preflight": {"path": str(preflight), "sha256": sha256_file(preflight)},
        "source_closure": closure,
    }
    _require_preflight_binding(checker, terminal_preflight=preflight, source_closure=closure)
    wrong_path = tmp_path / "other_preflight.json"
    wrong_path.write_text('{"static": false}\n', encoding="utf-8")
    with pytest.raises(NormalizedV2ContractError, match="different terminal preflight path"):
        _require_preflight_binding(checker, terminal_preflight=wrong_path, source_closure=closure)
    altered = dict(checker)
    altered["source_closure"] = {"scripts/frozen.py": "b" * 64}
    with pytest.raises(NormalizedV2ContractError, match="source closure differs"):
        _require_preflight_binding(altered, terminal_preflight=preflight, source_closure=closure)


def test_exposure_terminal_preflight_hashes_evaluator_checker_and_its_contract_test():
    source = (ROOT / "scripts/h1_carrierid_distribution_exposure_terminal_preflight.py").read_text(encoding="utf-8")
    assert "h1_carrierid_distribution_exposure_terminal_checker.py" in source
    assert "h1_carrierid_distribution_exposure_terminal_evaluate.py" in source
    assert "h1_carrierid_distribution_target.py" in source
    assert "test_h1_carrierid_distribution_exposure_terminal_contract.py" in source
