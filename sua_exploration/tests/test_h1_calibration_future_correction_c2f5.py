from __future__ import annotations

import inspect

import numpy as np

from sua_exploration.mc_maze import h1_calibration_future_correction_c2f5 as c2f5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


def _latent(rows: int = 20) -> np.ndarray:
    rng = np.random.default_rng(809)
    values = rng.normal(size=(rows, c2f5.RANK))
    assert np.linalg.matrix_rank(np.c_[np.ones(rows), values]) == c2f5.CARRIER_DIM
    return values


def _examples(seed: int = 14, sessions: int = 3) -> c2f5.SourceExamples:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(sessions, v1.EXPECTED_NEURONS, c2f5.CARRIER_DIM))
    quality = rng.normal(size=(sessions, v1.EXPECTED_NEURONS, c2f5.QUALITY_DIM))
    # A nontrivial shared correction that has no channel-index term.
    correction = np.empty_like(base)
    correction[..., 0] = 0.15 * base[..., 1] - 0.10 * quality[..., 0]
    correction[..., 1] = -0.11 * base[..., 3] + 0.08 * quality[..., 3]
    correction[..., 2] = 0.07 * base[..., 0] + 0.03 * quality[..., 5]
    correction[..., 3] = 0.12 * quality[..., 1]
    correction[..., 4] = -0.04 * base[..., 4] + 0.05 * quality[..., 6]
    return c2f5.SourceExamples(
        session_names=tuple(f"ses-19250108T0000{index:02d}" for index in range(sessions)),
        outer_date="19250101",
        budget=3,
        support_carriers=base,
        quality=quality,
        future_teachers=base + correction,
        manifests=tuple({"session": f"synthetic-{index}"} for index in range(sessions)),
    )


def test_c2f5_literal_contract_is_five_wide_and_frozen() -> None:
    assert c2f5.RANK == 4
    assert c2f5.CARRIER_DIM == 5
    assert len(c2f5.QUALITY_NAMES) == c2f5.QUALITY_DIM == 7
    assert c2f5.C2F_RIDGE_GRID == (0.01, 0.1, 1.0, 10.0)
    assert c2f5.SUPPORT_BUDGETS == (3, 4)


def test_exact_hse5_raw_carrier_recovery() -> None:
    rng = np.random.default_rng(15)
    z = _latent(18)
    response = rng.normal(size=(len(z), v1.EXPECTED_NEURONS))
    observed, quality, metadata = c2f5._exact_hse5_carrier_and_quality(z, response)
    expected = v2.fit_carrier_arrays(z, response)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert quality.shape == (v1.EXPECTED_NEURONS, c2f5.QUALITY_DIM)
    assert metadata["design_rank"] == c2f5.CARRIER_DIM


def test_analytic_quality_is_finite_and_carries_no_future_argument() -> None:
    rng = np.random.default_rng(16)
    z = _latent(17)
    response = rng.normal(size=(len(z), v1.EXPECTED_NEURONS))
    _carrier, quality, _meta = c2f5._exact_hse5_carrier_and_quality(z, response)
    assert np.isfinite(quality).all()
    signature = inspect.signature(c2f5.fit_operator)
    assert tuple(signature.parameters) == ("examples", "input_kind", "ridge_lambda")
    assert "target" not in " ".join(signature.parameters).lower()


def test_shared_operator_recovers_known_affine_residual_map() -> None:
    examples = _examples()
    operator = c2f5.fit_operator(examples, input_kind=c2f5.PRIMARY_INPUT_KIND, ridge_lambda=0.01)
    predicted = operator.apply(examples.support_carriers[0], examples.quality[0])
    expected = examples.future_teachers[0]
    # Ridge=0.01 intentionally causes a tiny shrinkage, so this verifies the
    # direction and close recovery rather than an impossible exact inverse.
    assert np.mean(np.square(predicted - expected)) < 2.0e-3
    assert operator.input_dim == c2f5.CARRIER_DIM + c2f5.QUALITY_DIM
    assert operator.manifest()["has_channel_index_feature"] is False


def test_quality_only_control_does_not_receive_carrier_inside_correction_map() -> None:
    examples = _examples()
    operator = c2f5.fit_operator(examples, input_kind=c2f5.QUALITY_ONLY_INPUT_KIND, ridge_lambda=0.1)
    quality = examples.quality[0]
    carrier_a = examples.support_carriers[0]
    carrier_b = carrier_a + 5.0
    # The final corrected carrier changes because the raw H-SE5 base is always
    # retained, but the fitted residual must be identical: B is absent from F.
    np.testing.assert_allclose(
        operator.predict_residual(carrier_a, quality),
        operator.predict_residual(carrier_b, quality),
        rtol=0.0, atol=1.0e-12,
    )


def test_channel_permutation_equivariance_and_no_channel_lookup() -> None:
    examples = _examples()
    operator = c2f5.fit_operator(examples, input_kind=c2f5.PRIMARY_INPUT_KIND, ridge_lambda=0.1)
    rng = np.random.default_rng(17)
    order = rng.permutation(v1.EXPECTED_NEURONS)
    permuted = c2f5.SourceExamples(
        session_names=examples.session_names,
        outer_date=examples.outer_date,
        budget=examples.budget,
        support_carriers=examples.support_carriers[:, order],
        quality=examples.quality[:, order],
        future_teachers=examples.future_teachers[:, order],
        manifests=examples.manifests,
    )
    permuted_operator = c2f5.fit_operator(permuted, input_kind=c2f5.PRIMARY_INPUT_KIND, ridge_lambda=0.1)
    np.testing.assert_allclose(operator.standardized_coefficients, permuted_operator.standardized_coefficients,
                               rtol=0.0, atol=1.0e-12)
    direct = operator.apply(examples.support_carriers[0], examples.quality[0])
    through_permuted = permuted_operator.apply(permuted.support_carriers[0], permuted.quality[0])
    np.testing.assert_allclose(through_permuted, direct[order], rtol=0.0, atol=1.0e-12)


def test_source_examples_fail_closed_if_outer_date_leaks() -> None:
    # The function checks source names before it attempts to inspect any data;
    # this is the direct target-future leakage guard.
    try:
        c2f5.build_source_examples({}, object(), source_names=("ses-19250101T111740",),
                                   outer_date="19250101", budget=3)  # type: ignore[arg-type]
    except v1.SparseEventEndpointError as error:
        assert "leaked" in str(error)
    else:
        raise AssertionError("outer-date source leakage was accepted")


def test_lambda_tie_break_is_smallest_and_gate_does_not_lower_material_threshold() -> None:
    # Selection's tie policy is structural and explicitly declared; gate proof
    # below guards against a +0.0199 post-hoc relaxation.
    selected = c2f5._select_lambda_from_source_scores([
        {"ridge_lambda": 0.01, "mean_score": 0.50},
        {"ridge_lambda": 0.1, "mean_score": 0.50 + 0.5e-12},
        {"ridge_lambda": 1.0, "mean_score": 0.49},
        {"ridge_lambda": 10.0, "mean_score": 0.48},
    ])
    assert selected == 0.01
    row = {
        "delta_c2f5_minus_hse5": 0.0199,
        "delta_c2f5_minus_label_shuffled": 0.03,
        "delta_c2f5_minus_row_shuffled": 0.03,
        "delta_c2f5_minus_quality_only": 0.03,
        "delta_c2f5_minus_intercept": 0.03,
    }
    rows = {f"s{index}": {key: value for key, value in row.items()} for index in range(13)}
    gate = c2f5.budget_gate(rows)
    assert gate["material_gain_vs_hse5"] is False
    assert gate["passed"] is False
