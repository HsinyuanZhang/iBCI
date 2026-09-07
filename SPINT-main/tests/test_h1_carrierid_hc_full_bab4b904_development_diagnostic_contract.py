"""No-target contracts for the H-C-on-D-S4e-window development diagnostic."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
)
from scripts.h1_carrierid_hc_full_bab4b904_development_diagnostic import (
    D_S4E_CONFIG,
    D_S4E_D_Q4E_TERMINAL,
    EXPOSURE_TERMINAL_STATUS,
    EXPECTED_665_QUERY_HASH,
    EXPECTED_BAB4_QUERY_HASH,
    H_C_FLOAT64_GATE,
    H_C_GATE_STATUS,
    _assert_same_window_identity_and_carrier,
    _metric_delta,
    _require_exposure_terminal_binding,
    _require_hc_gate_binding,
    _require_metric_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _metrics(window_hash: str, first: float = 0.2, second: float = 0.4, pooled: float = 0.3) -> dict[str, object]:
    return {
        "pooled_r2": pooled,
        "r2_accumulator_dtype": "float64",
        "query_window_indices_sha256": window_hash,
        "samples": 5,
        "last_batch_size": 1,
        "state_immutable": True,
        "state_sha256_before": "s" * 64,
        "state_sha256_after": "s" * 64,
        "per_session": {
            H1_M4_FOLD0_TARGET[0]: {"r2": first, "samples": 2},
            H1_M4_FOLD0_TARGET[1]: {"r2": second, "samples": 3},
        },
    }


def test_float64_metric_contract_requires_exact_window_and_immutable_state():
    metric = _metrics(EXPECTED_BAB4_QUERY_HASH)
    _require_metric_contract(metric, expected_query_hash=EXPECTED_BAB4_QUERY_HASH, label="fake")
    bad_hash = dict(metric)
    bad_hash["query_window_indices_sha256"] = EXPECTED_665_QUERY_HASH
    with pytest.raises(NormalizedV2ContractError, match="window hash"):
        _require_metric_contract(bad_hash, expected_query_hash=EXPECTED_BAB4_QUERY_HASH, label="fake")
    bad_state = dict(metric)
    bad_state["state_sha256_after"] = "t" * 64
    with pytest.raises(NormalizedV2ContractError, match="state hash"):
        _require_metric_contract(bad_state, expected_query_hash=EXPECTED_BAB4_QUERY_HASH, label="fake")
    bad_dtype = dict(metric)
    bad_dtype["r2_accumulator_dtype"] = "float32"
    with pytest.raises(NormalizedV2ContractError, match="float64"):
        _require_metric_contract(bad_dtype, expected_query_hash=EXPECTED_BAB4_QUERY_HASH, label="fake")


def test_metric_delta_is_paired_by_named_target_session_not_input_order():
    candidate = _metrics(EXPECTED_BAB4_QUERY_HASH, first=0.55, second=0.25, pooled=0.40)
    reference = _metrics(EXPECTED_665_QUERY_HASH, first=0.50, second=0.10, pooled=0.20)
    delta = _metric_delta(candidate, reference)
    assert delta["pooled_r2"] == pytest.approx(0.20)
    assert delta["per_session_r2"] == {
        H1_M4_FOLD0_TARGET[0]: pytest.approx(0.05),
        H1_M4_FOLD0_TARGET[1]: pytest.approx(0.15),
    }
    assert "not a selection" in str(delta["interpretation_limit"])


def _same_window_views(*, altered_window: bool = False, altered_carrier: bool = False):
    windows = [(H1_M4_FOLD0_TARGET[0], 3), (H1_M4_FOLD0_TARGET[1], 7)]
    exposure_windows = [(H1_M4_FOLD0_TARGET[0], 3), (H1_M4_FOLD0_TARGET[1], 8)] if altered_window else windows
    ordinary_support = {}
    exposure_support = {}
    for index, name in enumerate(H1_M4_FOLD0_TARGET):
        identity = np.full((2, 3), index, dtype=np.float32)
        carrier = np.full((4, 4), index + 1, dtype=np.float64)
        ordinary_support[name] = SimpleNamespace(identity=identity, carriers={"full": carrier})
        exposure_support[name] = SimpleNamespace(
            identity=identity.copy(), s4_carrier=carrier + (1.0 if altered_carrier and index == 0 else 0.0)
        )
    return (
        SimpleNamespace(
            window_indices=windows, window_indices_sha256=EXPECTED_665_QUERY_HASH, support=ordinary_support
        ),
        SimpleNamespace(
            window_indices=exposure_windows, window_indices_sha256=EXPECTED_BAB4_QUERY_HASH, support=exposure_support
        ),
    )


def test_same_window_proof_requires_exact_ordered_indices_and_bitwise_identity_carrier_equality():
    ordinary, exposure = _same_window_views()
    proof = _assert_same_window_identity_and_carrier(ordinary, exposure)
    assert proof["window_index_lists_exactly_equal"] is True
    assert proof["hash_encoding_difference_only"] == {
        "ordinary_h_c_binary_window_hash": EXPECTED_665_QUERY_HASH,
        "d_s4e_canonical_json_window_hash": EXPECTED_BAB4_QUERY_HASH,
    }
    assert all(proof["identity_arrays_bitwise_equal"].values())
    assert all(proof["ordinary_full_vs_exposure_s4_carrier_arrays_bitwise_equal"].values())
    ordinary, exposure = _same_window_views(altered_window=True)
    with pytest.raises(NormalizedV2ContractError, match="window-index lists differ"):
        _assert_same_window_identity_and_carrier(ordinary, exposure)
    ordinary, exposure = _same_window_views(altered_carrier=True)
    with pytest.raises(NormalizedV2ContractError, match="S4 carriers are not bitwise equal"):
        _assert_same_window_identity_and_carrier(ordinary, exposure)


def test_shipped_mode0444_receipts_bind_the_two_real_metric_schemas_without_target_access():
    """Read receipt bytes only: no DataModule, checkpoint load, or target loader."""

    hc_gate = assert_immutable_receipt(H_C_FLOAT64_GATE, H_C_GATE_STATUS)
    exposure_terminal = assert_immutable_receipt(D_S4E_D_Q4E_TERMINAL, EXPOSURE_TERMINAL_STATUS)
    old_hc = _require_hc_gate_binding(hc_gate)
    ds4e = _require_exposure_terminal_binding(exposure_terminal)
    assert old_hc["query_window_indices_sha256"] == EXPECTED_665_QUERY_HASH
    assert ds4e["query_window_indices_sha256"] == EXPECTED_BAB4_QUERY_HASH
    assert exposure_terminal["checkpoints"]["d_s4e"]["config_sha256"] == sha256_file(D_S4E_CONFIG)


def test_static_contract_binds_float64_gate_bab4_receipt_and_ordinary_normalizer_before_target_load():
    source = (
        ROOT / "scripts/h1_carrierid_hc_full_bab4b904_development_diagnostic.py"
    ).read_text(encoding="utf-8")
    assert "H1_CARRIERID_H32_FOLD0_TERMINAL_GATE_FLOAT64_R2.json" in source
    assert "bab4b904c16f91a017159e6296efb04a252fb346359748b7aa6b9f942137596f" in source
    assert "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da" in source
    assert "ordinary_source.normalizer" in source
    assert "ordinary_source.normalizer.normalizer_sha256 == exposure_source.normalizer.normalizer_sha256" in source
    assert "H1M4EBNormalizedV2StrictTargetDataset(\n        target_records, ordinary_source.plan, ordinary_source.normalizer, \"full\"" in source
    assert "H1CarrierIdDistributionStrictTargetDataset(\n        target_records, exposure_source.plan, ordinary_source.normalizer, \"s4\"" in source
    assert "window_index_lists_exactly_equal" in source
    assert "hash_encoding_difference_only" in source
    assert source.index("_require_hc_gate_binding(hc_gate)") < source.index("target_records = load_target_records")
    assert source.index("_require_exposure_terminal_binding(exposure_terminal)") < source.index("target_records = load_target_records")
    assert source.index("ordinary_source = _rebuild_hc_ordinary_source") < source.index("target_records = load_target_records")
    assert source.index("exposure_source = _rebuild_exposure_window_source") < source.index("target_records = load_target_records")
    assert "--execute-opened-fold0-development-diagnostic" in source
    assert "EST4" in source and "CI64" in source
    assert ".backward(" not in source
    assert "torch.optim" not in source


def test_static_contract_writes_a_new_immutable_development_receipt_not_an_existing_terminal_result():
    source = (
        ROOT / "scripts/h1_carrierid_hc_full_bab4b904_development_diagnostic.py"
    ).read_text(encoding="utf-8")
    assert "H1_CARRIERID_HC_FULL_ON_BAB4B904_DEVELOPMENT_DIAGNOSTIC_v1.json" in source
    assert "refusing to overwrite immutable development diagnostic" in source
    assert "write_immutable_json(OUTPUT, receipt)" in source
    assert "OPENED_FOLD0_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_PAPER_ENDPOINT_NOT_MODEL_SELECTION" in source
