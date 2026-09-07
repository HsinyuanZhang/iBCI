from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.falcon_h1_afc4_features import (  # noqa: E402
    H1_ARMS,
    H1_NUM_NEURONS,
    H1_Q,
    H1_VELOCITY_DIM,
    H1AFC4SourcePlan,
    deterministic_row_permutation,
    fit_source_velocity_basis,
    record_from_arrays,
)


def _record(name: str, *, seed: int = 7, split: str = "calib"):
    rng = np.random.default_rng(seed)
    lengths = (3, 61, 2, 53, 47, 4)
    n_bins = sum(lengths) + 20
    velocity = rng.normal(0.0, 0.1, size=(n_bins, H1_VELOCITY_DIM))
    neural = rng.normal(0.5, 0.3, size=(n_bins, H1_NUM_NEURONS))
    # Trial 0 is pretrial and invalid; first valid TrialNum is 1.  Its active
    # support has several contiguous segments separated by still-time gaps.
    trial_num = np.full(n_bins, 2.0)
    trial_num[: sum(lengths)] = 1.0
    eval_mask = np.ones(n_bins, dtype=bool)
    cursor = 0
    for length in lengths:
        cursor += length
        if cursor < n_bins:
            velocity[cursor : cursor + 2] = 0.0
            eval_mask[cursor : cursor + 2] = True
            cursor += 2
    eval_mask[:2] = False
    # Keep the first trial's active segments non-degenerate.
    trial_change = np.zeros(n_bins, dtype=bool)
    trial_change[2] = True
    return record_from_arrays(
        session_name=name,
        split=split,
        neural=neural,
        covariates=velocity,
        trial_change=trial_change,
        eval_mask=eval_mask,
        trial_num=trial_num,
    )


def test_h1_source_basis_is_raw_covariance_whitened_and_sign_stable():
    records = {name: _record(name, seed=index) for index, name in enumerate(("ses-a", "ses-b", "ses-c"))}
    basis = fit_source_velocity_basis(records)
    assert basis.components.shape == (H1_Q, H1_VELOCITY_DIM)
    assert basis.eigenvalues.shape == (H1_Q,)
    assert np.all(basis.eigenvalues > 0.0)
    assert np.all(basis.components[np.arange(H1_Q), basis.sign_anchor_indices] >= 0.0)
    np.testing.assert_allclose(
        basis.project(records["ses-a"].support_velocity).std(axis=0),
        np.ones(H1_Q),
        atol=0.25,
    )


def test_h1_primary_support_uses_first_valid_trial_and_all_active_segments():
    record = _record("ses-a")
    assert record.support_bins == sum(len(index) for index in record.support_segment_indices)
    assert len(record.support_segment_indices) >= 2
    assert np.all(record.trial_num[record.support_mask] == 1.0)
    assert np.all(record.eval_mask[record.support_mask])
    assert np.all(np.any(np.abs(record.covariates[record.support_mask]) >= 1.0e-3, axis=1))


def test_h1_arms_have_fixed_width_and_source_only_normalizer():
    records = {name: _record(name, seed=index) for index, name in enumerate(("ses-a", "ses-b", "ses-c"))}
    plan = H1AFC4SourcePlan(records)
    target = _record("ses-target", seed=99)
    for arm in H1_ARMS:
        values, audit = plan.descriptor_for(target, arm)
        assert values.shape == (H1_NUM_NEURONS, 4)
        assert np.isfinite(values).all()
        assert audit.get("normalizer_fit_sessions") == list(records)
    full, _ = plan.descriptor_for(target, "afc4_h1q3")
    b4, _ = plan.descriptor_for(target, "afc4_h1_b4")
    zero, _ = plan.descriptor_for(target, "zero4")
    rs, _ = plan.descriptor_for(target, "afc4_h1_rs")
    np.testing.assert_array_equal(b4[:, :3], np.zeros((H1_NUM_NEURONS, 3), dtype=np.float32))
    np.testing.assert_array_equal(zero, np.zeros_like(zero))
    assert not np.array_equal(full, rs)


def test_h1_xs_is_global_cross_segment_bijection_without_self_pairs():
    records = {name: _record(name, seed=index) for index, name in enumerate(("ses-a", "ses-b"))}
    plan = H1AFC4SourcePlan(records)
    target = _record("ses-target", seed=99)
    values, audit = plan.descriptor_for(target, "afc4_h1_xs")
    assert values.shape == (H1_NUM_NEURONS, 4)
    assert audit["label_shuffle"] is True
    segment_ids = np.concatenate(
        [np.full(index.size, segment, dtype=np.int64) for segment, index in enumerate(target.support_segment_indices)]
    )
    source_indices = np.asarray(audit["source_index_permutation"], dtype=np.int64)
    np.testing.assert_array_equal(np.sort(source_indices), np.arange(target.support_bins))
    assert np.all(segment_ids[source_indices] != segment_ids)


def test_h1_row_permutation_is_deterministic_complete_and_nonidentity():
    first = deterministic_row_permutation(H1_NUM_NEURONS, session_name="ses-a", seed=42)
    second = deterministic_row_permutation(H1_NUM_NEURONS, session_name="ses-a", seed=42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(H1_NUM_NEURONS))
    assert np.all(first != np.arange(H1_NUM_NEURONS))


def test_h1_xs_rejects_a_single_segment_instead_of_faking_a_null():
    rng = np.random.default_rng(3)
    velocity = rng.normal(size=(20, H1_VELOCITY_DIM))
    neural = rng.normal(size=(20, H1_NUM_NEURONS))
    record = record_from_arrays(
        session_name="ses-a", split="calib", neural=neural, covariates=velocity,
    )
    plan = H1AFC4SourcePlan({"ses-a": record})
    with pytest.raises(ValueError, match="at least two active segments"):
        plan.descriptor_for(record, "afc4_h1_xs")
