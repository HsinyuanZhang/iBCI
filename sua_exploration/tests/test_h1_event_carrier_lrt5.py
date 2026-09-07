from __future__ import annotations

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_lrt5 as lrt5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


def _subspace(rank: int = 2) -> lrt5.ChannelSlopeSubspace:
    # Put an explicitly channel-correspondent orthonormal basis in the first
    # rows; a large source mean makes it easy to prove it is not deployed.
    u = np.zeros((v1.EXPECTED_NEURONS, rank), dtype=np.float64)
    u[:rank, :rank] = np.eye(rank)
    return lrt5.ChannelSlopeSubspace(
        outer_date="19250101", source_sessions=("ses-19250108T110520", "ses-19250113T120811"), rank=rank,
        u=u, source_slope_mean=np.arange(v1.EXPECTED_NEURONS, dtype=np.float64) + 100.0,
        source_prior_slopes=np.full((lrt5.RANK, v1.EXPECTED_NEURONS), 7.0),
        source_slope_carrier_sha256=("a", "b"), source_slope_row_count=8, subspace_sha256="synthetic",
    )


def _arrays() -> tuple[np.ndarray, np.ndarray, lrt5.ChannelSlopeSubspace]:
    rng = np.random.default_rng(220)
    z = rng.normal(size=(19, lrt5.RANK))
    assert np.linalg.matrix_rank(z - z.mean(axis=0, keepdims=True)) == lrt5.RANK
    subspace = _subspace()
    a = np.asarray([[0.5, -0.2], [0.3, 0.15], [-0.1, 0.4], [0.2, -0.25]])
    slopes = a @ subspace.u.T
    response = z @ slopes + np.linspace(-0.5, 0.5, v1.EXPECTED_NEURONS)[None, :]
    return z, response, subspace


def test_lrt5_literal_grid_and_carrier_width_are_frozen() -> None:
    assert lrt5.RANK == 4 and lrt5.CARRIER_DIM == 5
    assert lrt5.TARGET_RANK_GRID == (1, 2, 4, 8)
    assert lrt5.TARGET_RIDGE_GRID == (0.1, 1.0, 3.0, 10.0)
    assert lrt5.SUPPORT_BUDGETS == (3, 4)


def test_lrt5_closed_form_matches_explicit_centered_equation_and_ignores_source_mean() -> None:
    z, response, subspace = _arrays()
    carrier, metadata = lrt5.fit_lrt_carrier(z, response, subspace, ridge_lambda=1.0)
    zc = z - z.mean(axis=0, keepdims=True)
    yc = response - response.mean(axis=0, keepdims=True)
    expected_a = np.linalg.solve(
        zc.T @ zc + np.eye(lrt5.RANK) * len(z), zc.T @ (yc @ subspace.u),
    )
    expected_slopes = expected_a @ subspace.u.T
    expected_intercept = response.mean(axis=0) - z.mean(axis=0) @ expected_slopes
    np.testing.assert_allclose(carrier[:, :lrt5.RANK].T, expected_slopes, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(carrier[:, -1], expected_intercept, rtol=0.0, atol=1.0e-12)
    assert metadata["target_pairing_required"] is True
    # Altering the stored source mean cannot alter the deployable correct fit.
    altered = lrt5.ChannelSlopeSubspace(
        **{**subspace.__dict__, "source_slope_mean": np.full(v1.EXPECTED_NEURONS, -1e9)}
    )
    repeated, _ = lrt5.fit_lrt_carrier(z, response, altered, ridge_lambda=1.0)
    np.testing.assert_array_equal(carrier, repeated)


def test_lrt5_target_pairing_changes_a_and_carrier() -> None:
    z, response, subspace = _arrays()
    correct, meta = lrt5.fit_lrt_carrier(z, response, subspace, ridge_lambda=0.1)
    order = np.roll(np.arange(len(z)), 1)
    wrong, wrong_meta = lrt5.fit_lrt_carrier(z[order], response, subspace, ridge_lambda=0.1)
    assert meta["target_coefficient_sha256"] != wrong_meta["target_coefficient_sha256"]
    assert not np.array_equal(correct, wrong)


def test_lrt5_u_row_shuffle_is_fixed_point_free_and_changes_attachment() -> None:
    z, response, subspace = _arrays()
    shuffled, manifest = lrt5.row_shuffle_subspace(subspace, session="ses-19250101T111740", budget=3)
    assert manifest["fixed_points"] == 0
    assert not np.array_equal(subspace.u, shuffled.u)
    correct, _ = lrt5.fit_lrt_carrier(z, response, subspace, ridge_lambda=0.1)
    wrong, _ = lrt5.fit_lrt_carrier(z, response, shuffled, ridge_lambda=0.1)
    assert not np.array_equal(correct, wrong)


def test_lrt5_source_prior_control_is_mean_aligned_without_target_label_pairing() -> None:
    z, response, subspace = _arrays()
    first = lrt5.carrier_from_source_prior(z, response, subspace)
    # A static nonzero W must be accompanied by b=mean(Y)-mean(Z)W, otherwise
    # its predicted target support mean is wrong and the comparison is unfair.
    predicted = lrt5.predict(first, z)
    np.testing.assert_allclose(predicted.mean(axis=0), response.mean(axis=0), rtol=0.0, atol=1.0e-12)
    # A hypothetical response-row relabelling preserves the no-pairing arm:
    # slopes are source fixed and the unpaired response mean is unchanged.
    second = lrt5.carrier_from_source_prior(z, response[::-1], subspace)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, :lrt5.RANK], subspace.source_prior_slopes.T)


def test_lrt5_outer_source_allowlist_excludes_outer_future_date() -> None:
    names = lrt5.source_names_for_outer("19250101")
    assert len(names) == 11
    assert all(not name.startswith("ses-19250101") for name in names)
    assert set(names) | {name for name in v1.H1_HELDIN_SESSIONS if name.startswith("ses-19250101")} == set(v1.H1_HELDIN_SESSIONS)


def test_lrt5_gate_requires_material_gain_and_every_pairing_static_control() -> None:
    good = {"defined_sessions": 13, "mean": 0.0201, "median": 0.0101, "positive": 10,
            "leave_largest_absolute_out_mean": 0.0001}
    aggregate = {
        "correct_minus_hse5": good, "correct_minus_label_shuffle": good,
        "correct_minus_u_row_shuffle": good, "correct_minus_intercept": good,
        "correct_minus_source_prior": good,
    }
    assert lrt5.gate(aggregate)["passed"] is True
    aggregate["correct_minus_u_row_shuffle"] = {**good, "mean": -0.01}
    assert lrt5.gate(aggregate)["passed"] is False
