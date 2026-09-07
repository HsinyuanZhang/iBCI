from __future__ import annotations

import numpy as np

from sua_exploration.mc_maze import h1_calibration_future_quadratic_c2f5 as qc2f5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


def _examples(seed: int = 204, sessions: int = 3) -> qc2f5.SourceExamples:
    rng = np.random.default_rng(seed)
    support = rng.normal(size=(sessions, v1.EXPECTED_NEURONS, qc2f5.CARRIER_DIM))
    quality = rng.normal(size=(sessions, v1.EXPECTED_NEURONS, qc2f5.QUALITY_DIM))
    future = support.copy()
    # Fixed interactions exercise square and lexicographic cross terms.
    future[..., 0] += .13 * support[..., 0] * quality[..., 1] - .08 * support[..., 2] ** 2
    future[..., 1] += .07 * quality[..., 0] * quality[..., 6]
    future[..., 4] += .04 * support[..., 4] ** 2
    return qc2f5.SourceExamples(
        session_names=tuple(f"ses-19250108T0000{item:02d}" for item in range(sessions)), outer_date="19250101", budget=3,
        support_carriers=support, quality=quality, future_teachers=future,
        manifests=tuple({"session": str(item)} for item in range(sessions)),
    )


def test_degree2_design_is_exactly_linear_squares_then_pairwise_crosses() -> None:
    values = np.array([[2., 3., 5.], [7., 11., 13.]])
    observed = qc2f5.degree2_features(values)
    expected = np.array([[2., 3., 5., 4., 9., 25., 6., 10., 15.], [7., 11., 13., 49., 121., 169., 77., 91., 143.]])
    np.testing.assert_allclose(observed, expected, rtol=0., atol=0.)
    names = qc2f5.polynomial_feature_names(("a", "b", "c"))
    assert names == ("linear:a", "linear:b", "linear:c", "square:a^2", "square:b^2", "square:c^2", "cross:a*b", "cross:a*c", "cross:b*c")


def test_quadratic_operator_recovers_synthetic_interaction() -> None:
    examples = _examples()
    operator = qc2f5.fit_quadratic_operator(examples, input_kind=qc2f5.PRIMARY_INPUT_KIND, ridge_lambda=.01)
    recovered = operator.apply(examples.support_carriers[0], examples.quality[0])
    assert np.mean(np.square(recovered - examples.future_teachers[0])) < 4e-3
    manifest = operator.manifest()
    assert manifest["raw_input_dim"] == 12
    assert manifest["polynomial_feature_dim"] == 90
    assert manifest["design_width_with_intercept"] == 91
    assert manifest["has_channel_index_feature"] is False


def test_source_teacher_pairing_shuffle_preserves_marginal_and_destroys_pairs() -> None:
    examples = _examples()
    shuffled, manifest = qc2f5.source_teacher_pairing_shuffle(examples)
    assert manifest["fixed_points"] == 0
    assert manifest["teacher_marginal_sha256_before"] == manifest["teacher_marginal_sha256_after"]
    assert not np.array_equal(shuffled.future_teachers, examples.future_teachers)
    assert not np.array_equal(shuffled.residuals, examples.residuals)


def test_quality_only_f2_has_no_B_argument_and_is_deterministic() -> None:
    examples = _examples()
    first = qc2f5.fit_quadratic_operator(examples, input_kind=qc2f5.QUALITY_ONLY_INPUT_KIND, ridge_lambda=.1)
    second = qc2f5.fit_quadratic_operator(examples, input_kind=qc2f5.QUALITY_ONLY_INPUT_KIND, ridge_lambda=.1)
    carrier = examples.support_carriers[0]
    quality = examples.quality[0]
    np.testing.assert_allclose(first.predict_residual(carrier, quality), first.predict_residual(carrier + 7., quality), rtol=0., atol=1e-12)
    np.testing.assert_allclose(first.standardized_coefficients, second.standardized_coefficients, rtol=0., atol=0.)
    assert first.operator_sha256 == second.operator_sha256


def test_outer_date_exclusion_is_checked_before_source_data_access() -> None:
    try:
        qc2f5.linear.build_source_examples({}, object(), source_names=("ses-19250101T111740",), outer_date="19250101", budget=3)  # type: ignore[arg-type]
    except v1.SparseEventEndpointError as error:
        assert "leaked" in str(error)
    else:
        raise AssertionError("outer-date source leakage accepted")


def test_strongest_ridge_tie_break_is_frozen() -> None:
    rows = [{"ridge_lambda": .01, "mean_score": .5}, {"ridge_lambda": .1, "mean_score": .5 + .5e-12}, {"ridge_lambda": 1., "mean_score": .49}, {"ridge_lambda": 10., "mean_score": .48}]
    assert qc2f5.linear._select_lambda_from_source_scores(rows) == .01  # Linear receipt stays unchanged.
    assert qc2f5._select_lambda_from_source_scores(rows) == .1
