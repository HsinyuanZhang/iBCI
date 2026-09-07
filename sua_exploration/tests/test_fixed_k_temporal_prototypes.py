"""Synthetic/no-NWB contracts for the Step-3 fixed-K prototype readiness path."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.fixed_k_temporal_prototypes import (  # noqa: E402
    FIXED_K,
    TEMPORAL_RANK,
    PrototypeAccumulator,
    build_outer_loso_carriers,
    carrier_from_support_trials,
    d4_carrier_padded,
    deterministic_slot_permutation,
    fit_ordered_anchors_source_only,
    label_disclosure_receipt,
    outer_loso_proxy,
    slot_shuffle_carrier,
    streaming_cost_receipt,
)

SCRIPT = ROOT / "sua_exploration" / "scripts" / "audit_m1_fixed_k_prototypes.py"


def _anchors() -> np.ndarray:
    return np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
        dtype=np.float64,
    )


def _support_bins(units: int = 5, bins: int = 14) -> np.ndarray:
    time = np.arange(bins, dtype=np.float64)[:, None]
    scale = np.arange(1, units + 1, dtype=np.float64)[None, :]
    return 0.2 * time + scale


def _support_trials(units: int = 5, bins: int = 14) -> tuple[np.ndarray, ...]:
    values = _support_bins(units, bins)
    # This helper is also used for 8/9-bin synthetic source sessions; unlike
    # fixed slice cutpoints, array_split guarantees that all three trial blocks
    # remain nonempty for every accepted test length.
    return tuple(np.array_split(values, 3, axis=0))


def _script_module():
    spec = importlib.util.spec_from_file_location("m1_fixed_k_prototype_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_representation_is_locked_to_k4_rank4_and_receipts_no_raw_support_state():
    assert FIXED_K == 4 and TEMPORAL_RANK == 4
    receipt = streaming_cost_receipt(64)
    assert receipt.raw_support_matrix_elements == 0
    assert receipt.prototype_count_elements == 64 * 4
    assert receipt.prototype_sum_elements == 64 * 4 * 4
    assert receipt.deployed_state_elements == 64 * (4 + 16 + 4 + 3)
    assert receipt.deployed_state_bytes == receipt.deployed_state_elements * 4
    assert receipt.per_unit_per_bin_multiplications > 0


def test_stream_is_causal_finalizable_and_does_not_accept_query_updates():
    state = PrototypeAccumulator(_anchors(), num_units=3)
    first = state.update(np.asarray([1.0, 2.0, 3.0]))
    state.update(np.asarray([2.0, 3.0, 4.0]))
    final = state.finalize()
    assert first.shape == (3,)
    assert final["prototype"].shape == (3, 20)
    assert final["rate_only"].shape == (3, 20)
    assert final["prototype_blocks"].shape == (3, 4, 5)
    assert state.finalized and final["processed_bins"] == 2
    with pytest.raises(RuntimeError, match="query bins"):
        state.update(np.asarray([9.0, 9.0, 9.0]))
    # The neural-only deployed update surface makes dense behaviour/labels
    # impossible to smuggle in as an argument.
    assert list(inspect.signature(PrototypeAccumulator.update).parameters) == ["self", "neural_rate_bin"]


def test_unit_permutation_equivariance_of_frozen_anchor_operator():
    trials = _support_trials(units=6)
    carrier = carrier_from_support_trials(_anchors(), trials)
    permutation = np.asarray([4, 0, 5, 2, 1, 3])
    permuted = carrier_from_support_trials(_anchors(), tuple(trial[:, permutation] for trial in trials))
    assert np.allclose(permuted["prototype"], carrier["prototype"][permutation])
    assert np.allclose(permuted["rate_only"], carrier["rate_only"][permutation])
    assert np.allclose(permuted["prototype_blocks"], carrier["prototype_blocks"][permutation])


def test_source_only_anchor_fit_excludes_outer_leftout_is_order_invariant_and_reports_offline_workspace():
    source = {"s0": _support_trials(3, 8), "s1": _support_trials(3, 9), "s2": _support_trials(3, 10)}
    anchors, receipt = fit_ordered_anchors_source_only(source, forbidden_session="s3")
    permuted_source = {name: tuple(trial[:, [2, 0, 1]] for trial in trials) for name, trials in reversed(list(source.items()))}
    permuted_anchors, permuted_receipt = fit_ordered_anchors_source_only(permuted_source, forbidden_session="s3")
    assert anchors.shape == (4, 4)
    assert np.allclose(anchors, permuted_anchors)
    assert receipt["source_sessions"] == ["s0", "s1", "s2"]
    assert receipt["forbidden_left_out_session"] == "s3"
    assert receipt["source_support_only"] is True
    assert receipt["source_unit_and_session_row_order_invariant"] is True
    assert receipt["offline_anchor_workspace"]["raw_support_matrix_retained"] is False
    assert receipt["offline_anchor_workspace"]["not_deployed_streaming_state"] is True
    assert permuted_receipt["trial_filter_reset"] is True
    with pytest.raises(ValueError, match="must not enter"):
        fit_ordered_anchors_source_only({**source, "s3": _support_trials(3, 8)}, forbidden_session="s3")


def test_trial_boundaries_reset_causal_filter_and_rank2_continuous_input_is_rejected():
    trials = _support_trials(2, 9)
    helper = carrier_from_support_trials(_anchors(), trials)
    direct = PrototypeAccumulator(_anchors(), num_units=2)
    for trial in trials:
        direct.update_trial(trial)
    expected = direct.finalize()
    assert np.allclose(helper["prototype"], expected["prototype"])
    with pytest.raises(ValueError, match="trialized"):
        carrier_from_support_trials(_anchors(), np.concatenate(trials, axis=0))


def test_complete_slot_block_shuffle_preserves_blocks_not_rows_or_coordinates():
    blocks = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5)
    order = deterministic_slot_permutation(session_name="leftout", seed=42)
    shuffled = slot_shuffle_carrier(blocks, order)
    assert not np.array_equal(order, np.arange(4))
    assert shuffled.shape == (2, 20)
    assert np.array_equal(shuffled.reshape(2, 4, 5), blocks[:, order, :])
    with pytest.raises(ValueError, match="non-identity"):
        slot_shuffle_carrier(blocks, np.arange(4))


def test_rate_and_d4_are_width_matched_and_label_disclosure_is_explicit():
    state = PrototypeAccumulator(_anchors(), num_units=2)
    state.update_trial(_support_bins(2, 7))
    rate = state.finalize()["rate_only"]
    d4 = d4_carrier_padded(np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64))
    assert rate.shape == d4.shape == (2, 20)
    assert np.array_equal(d4[:, :4], np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]))
    assert not d4[:, 4:].any()
    disclosure = label_disclosure_receipt()
    assert disclosure["prototype"]["support_labels"] == "none"
    assert "obj_id" in disclosure["D4"]["support_labels"]
    assert "offline score" in disclosure["later_neural_rate_target"]


def test_outer_loso_requires_four_sessions_exactly_four_arms_and_never_selects_hyperparameters():
    sessions = ["s0", "s1", "s2", "s3"]
    rng = np.random.default_rng(3)
    targets = {name: rng.normal(size=(6, 1)) for name in sessions}
    carriers = {
        left_out: {
            arm: {name: rng.normal(size=(6, 20)) for name in sessions}
            for arm in ("D4", "rate_only", "prototype", "slot_shuffle")
        }
        for left_out in sessions
    }
    result = outer_loso_proxy(carriers, targets, ridge=1.0)
    assert result["source_only_outer_loso"] is True
    assert result["endpoint"] == "later_neural_rate_oracle_proxy"
    assert all(len(rows) == 4 for rows in result["arms"].values())
    with pytest.raises(ValueError, match="exactly four"):
        outer_loso_proxy(carriers, {key: value for key, value in targets.items() if key != "s3"})
    incomplete = {key: dict(value) for key, value in carriers.items()}; incomplete["s0"].pop("slot_shuffle")
    with pytest.raises(ValueError, match="requires exactly"):
        outer_loso_proxy(incomplete, targets)


def test_fold_level_builder_refits_anchors_and_shuffles_every_session_in_each_fold():
    sessions = ["s0", "s1", "s2", "s3"]
    support = {name: _support_trials(3, 10) for name in sessions}
    d4 = {name: np.arange(12, dtype=np.float64).reshape(3, 4) + index for index, name in enumerate(sessions)}
    folds, receipts = build_outer_loso_carriers(support, d4, seed=12)
    assert set(folds) == set(sessions) == set(receipts)
    for left_out in sessions:
        receipt = receipts[left_out]
        assert left_out not in receipt["anchor_receipt"]["source_sessions"]
        assert set(receipt["session_slot_permutations"]) == set(sessions)
        for session in sessions:
            assert folds[left_out]["slot_shuffle"][session].shape == (3, 20)
            assert not np.array_equal(receipt["session_slot_permutations"][session], list(range(4)))


def test_prelaunch_receipt_is_no_data_fail_closed_and_refuses_manifest_mode(tmp_path):
    module = _script_module()
    receipt = module.build_prelaunch_receipt(example_units=8)
    assert receipt["gate_status"]["gpu_authorization"] is False
    assert receipt["gate_status"]["not_a_gate_a_result"] is True
    assert receipt["oracle_boundary"]["formal_test"] == "unopened"
    assert receipt["comparators"]["required_exact_set"] == ["D4", "rate_only", "prototype", "slot_shuffle"]
    path = module.run_dry_run(tmp_path / "receipt")
    loaded = json.loads(path.read_text())
    assert loaded["locked_representation"]["k"] == 4
    assert (path.parent / "prelaunch_receipt.sha256").is_file()
    with pytest.raises(FileExistsError):
        module.run_dry_run(path.parent)
