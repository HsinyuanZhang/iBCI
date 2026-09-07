"""Support-only parity and fail-closed tests for the active XLSv2 adapter."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from src.data import afc4_xls_v2 as primitive
from src.data.afc4_xls_v2_adapter import (
    AUDIT_SHA256,
    Afc4XlsV2AdapterError,
    afc4_xls_v2_from_support,
    load_immutable_xls_v2_audit,
)
from src.data.falcon_k4_features import (
    collect_k4_support_blocks,
    fit_k4_descriptor_from_blocks,
    k4_from_raw_calibration,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "sua_exploration/results/rt_afc4_ls_null_strength_audit_v2/RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json"


def _support() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins_per_trial = 10
    trials = 26
    time = bins_per_trial * trials
    trial_change = np.zeros(time, dtype=bool)
    trial_change[::bins_per_trial] = True
    trial_index = np.arange(time) // bins_per_trial
    groups = trial_index % 8
    segment_ids = groups.astype(np.int64)
    angles = 2.0 * np.pi * groups / 8.0
    phase = (np.arange(time) % bins_per_trial) * 0.01
    velocity = np.column_stack([np.cos(angles + phase), np.sin(angles + phase)]).astype(np.float32)
    neural = np.column_stack([
        0.2 + (np.arange(time) % 5) * 0.03,
        0.4 + velocity[:, 0] * 0.05,
        0.5 + velocity[:, 1] * 0.04,
    ]).astype(np.float32)
    return neural, velocity, trial_change, segment_ids


def _synthetic_receipt(session_name: str) -> tuple[dict, str]:
    neural, velocity, trial_change, segment_ids = _support()
    blocks = collect_k4_support_blocks(
        neural, velocity, trial_change, calibration_n_trials=24, segment_ids=segment_ids,
    )
    permutation, _ = primitive.deterministic_random_cross_reach_derangement(
        blocks.group_ids, blocks.velocity, session_name=session_name, seed=42,
    )
    receipt = {
        "schema": primitive.V2_AUDIT_SCHEMA,
        "status": primitive.V2_AUDIT_STATUS,
        "fold_rows": [{
            "session_name": session_name,
            "support_trial_index_range": [0, 24],
            "active_blocks": int(blocks.rates.shape[0]),
            "accepted_reaches": int(np.unique(blocks.group_ids).size),
            "num_channels": int(blocks.rates.shape[1]),
            "v2_random_cross_reach_null": {"permutation_sha256": primitive.permutation_sha256(permutation)},
        }],
    }
    return receipt, primitive.permutation_sha256(permutation)


def test_adapter_uses_same_support_blocks_and_ols_with_only_permuted_velocity() -> None:
    neural, velocity, trial_change, segment_ids = _support()
    session = "synthetic-xls-v2"
    receipt, expected_sha = _synthetic_receipt(session)
    features, audit = afc4_xls_v2_from_support(
        neural,
        velocity,
        trial_change,
        segment_ids=segment_ids,
        session_name=session,
        audit_receipt=receipt,
        audit_receipt_sha256=AUDIT_SHA256,
        calibration_n_trials=24,
        seed=42,
    )
    blocks = collect_k4_support_blocks(
        neural, velocity, trial_change, calibration_n_trials=24, segment_ids=segment_ids,
    )
    permutation, _ = primitive.deterministic_random_cross_reach_derangement(
        blocks.group_ids, blocks.velocity, session_name=session, seed=42,
    )
    expected, rank, condition = fit_k4_descriptor_from_blocks(blocks.rates, blocks.velocity[permutation])
    np.testing.assert_array_equal(features, expected)
    assert audit.design_rank == rank and audit.design_condition == condition
    assert audit.as_dict()["label_permutation_sha256"] == expected_sha
    assert audit.as_dict()["query_labels_available_to_generator"] is False
    assert audit.as_dict()["common_inverse_or_alignment_map"] is False


def test_exposed_collection_and_fit_reproduce_legacy_aligned_rt_path_exactly() -> None:
    neural, velocity, trial_change, segment_ids = _support()
    legacy, legacy_audit = k4_from_raw_calibration(
        neural, velocity, trial_change, calibration_n_trials=24, segment_ids=segment_ids,
    )
    blocks = collect_k4_support_blocks(
        neural, velocity, trial_change, calibration_n_trials=24, segment_ids=segment_ids,
    )
    exposed, rank, condition = fit_k4_descriptor_from_blocks(blocks.rates, blocks.velocity)
    np.testing.assert_array_equal(exposed, legacy)
    assert rank == legacy_audit.design_rank
    assert condition == legacy_audit.design_condition
    assert blocks.rates.shape[0] == legacy_audit.active_blocks


def test_post_support_velocity_and_neural_bytes_are_structurally_invisible() -> None:
    neural, velocity, trial_change, segment_ids = _support()
    session = "synthetic-xls-v2"
    receipt, _ = _synthetic_receipt(session)
    baseline, baseline_audit = afc4_xls_v2_from_support(
        neural, velocity, trial_change, segment_ids=segment_ids, session_name=session,
        audit_receipt=receipt, audit_receipt_sha256=AUDIT_SHA256,
    )
    cutoff = int(np.flatnonzero(trial_change)[24])
    changed_neural, changed_velocity = neural.copy(), velocity.copy()
    changed_neural[cutoff:] = np.nan
    changed_velocity[cutoff:] = np.nan
    changed, changed_audit = afc4_xls_v2_from_support(
        changed_neural, changed_velocity, trial_change, segment_ids=segment_ids,
        session_name=session, audit_receipt=receipt,
        audit_receipt_sha256=AUDIT_SHA256,
    )
    np.testing.assert_array_equal(changed, baseline)
    assert changed_audit.as_dict() == baseline_audit.as_dict()


def test_adapter_signature_has_no_query_model_score_or_alignment_surface() -> None:
    parameters = set(inspect.signature(afc4_xls_v2_from_support).parameters)
    assert not parameters.intersection({
        "query_labels", "query_velocity", "decoder", "model", "score", "checkpoint",
        "inverse", "rotation", "procrustes", "ridge", "alignment_map",
    })
    neural, velocity, trial_change, segment_ids = _support()
    receipt, _ = _synthetic_receipt("synthetic-xls-v2")
    with pytest.raises(TypeError, match="query_labels"):
        afc4_xls_v2_from_support(
            neural, velocity, trial_change, segment_ids=segment_ids,
            session_name="synthetic-xls-v2", audit_receipt=receipt,
            audit_receipt_sha256=AUDIT_SHA256, query_labels=np.zeros((3, 2)),
        )


def test_adapter_fails_closed_on_session_or_receipt_sha_drift() -> None:
    neural, velocity, trial_change, segment_ids = _support()
    receipt, _ = _synthetic_receipt("synthetic-xls-v2")
    with pytest.raises((Afc4XlsV2AdapterError, primitive.Afc4XlsV2Error), match="session"):
        afc4_xls_v2_from_support(
            neural, velocity, trial_change, segment_ids=segment_ids,
            session_name="other-session", audit_receipt=receipt,
            audit_receipt_sha256=AUDIT_SHA256,
        )
    with pytest.raises(Afc4XlsV2AdapterError, match="wrong audit SHA"):
        afc4_xls_v2_from_support(
            neural, velocity, trial_change, segment_ids=segment_ids,
            session_name="synthetic-xls-v2", audit_receipt=receipt,
            audit_receipt_sha256="0" * 64,
        )


def test_real_support_audit_loader_binds_mode_schema_and_exact_sha() -> None:
    receipt, digest = load_immutable_xls_v2_audit(AUDIT)
    assert digest == AUDIT_SHA256
    assert AUDIT.stat().st_mode & 0o777 == 0o444
    assert receipt["schema"] == primitive.V2_AUDIT_SCHEMA
    assert receipt["status"] == primitive.V2_AUDIT_STATUS
    assert len(receipt["fold_rows"]) == 15
