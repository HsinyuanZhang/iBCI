import numpy as np
import pytest
from dataclasses import dataclass

from dandi688_bench_v2.baselines import (
    CanonicalAlignedFA, CanonicalCoral, CanonicalDiagZ, fit_baseline, fit_ridge,
    fit_ridge_grid_adaptive, fit_ridge_sufficient, fit_wf_fss, fitstatic_adapter, predict_baseline, predict_wf_fss,
)


@dataclass
class Record:
    session_id: str
    neural: np.ndarray
    velocity: np.ndarray
    query_indices: np.ndarray
    support_indices: np.ndarray
    carrier_indices: np.ndarray
    channel_indices: np.ndarray
    split: str = "train"
    representation: str = "PMUA"


def test_diag_z_embeds_only_observed_hardware_rows_and_fills_centered_missing_with_zero():
    support = np.array([[1., 10.], [3., 14.], [5., 18.]])
    fitted = CanonicalDiagZ.fit(support, np.array([3, 7]), canonical_size=10)
    got = fitted.transform(np.array([[3., 14.]]))
    assert got.shape == (1, 10)
    np.testing.assert_allclose(got[0, [3, 7]], 0)
    np.testing.assert_array_equal(got[0, [0, 1, 2, 4, 5, 6, 8, 9]], 0)


def test_coral_uses_only_common_observed_electrodes_and_returns_canonical_z_features():
    base = np.array([[-2., -1.], [-1., 1.], [1., -1.], [2., 1.]])
    source = base * np.array([2., 3.]) + np.array([5., -4.])
    target = base * np.array([4., .5]) + np.array([-2., 8.])
    # The third target column is genuinely observed but absent from the source;
    # it must not enter the two-dimensional covariance estimate.
    target = np.c_[target[:, 1], target[:, 0], np.array([100., -50., 20., 80.])]
    fitted = CanonicalCoral.fit(source, np.array([2, 5]), target, np.array([5, 2, 9]),
                                ridge=0, shrinkage=0, canonical_size=12)
    got = fitted.transform(target)
    np.testing.assert_allclose(got[:, [2, 5]].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(got[:, [2, 5]].std(axis=0), 1, atol=1e-12)
    np.testing.assert_array_equal(got[:, 9], 0)
    # Per-session affine re-expression is removed before CORAL covariance fit.
    source_relabelled = source * np.array([3., 7.]) + np.array([11., -9.])
    target_relabelled = target.copy()
    target_relabelled[:, :2] = target[:, :2] * np.array([5., 2.]) + np.array([-3., 4.])
    relabelled = CanonicalCoral.fit(source_relabelled, np.array([2, 5]), target_relabelled,
                                    np.array([5, 2, 9]), ridge=0, shrinkage=0, canonical_size=12)
    np.testing.assert_allclose(relabelled.transform(target_relabelled), got, atol=1e-12, rtol=1e-12)


def test_aligned_fa_rejects_rank_deficient_common_loading_alignment():
    rng = np.random.RandomState(0)
    source = rng.normal(size=(30, 4))
    target = rng.normal(size=(30, 4))
    # K=3 with only two common electrodes cannot pass the required stable-row gate.
    with pytest.raises(ValueError, match="common"):
        CanonicalAlignedFA.fit(source, np.array([0, 1, 2, 3]), target, np.array([0, 1, 6, 7]),
                               latent_dim=3, stable_fraction=.5, posterior="stable", n_init=1, max_iter=20)


def test_ridge_is_closed_form_with_an_unpenalized_intercept():
    x = np.array([[0.], [1.], [2.]])
    y = 2 * x + 5
    readout = fit_ridge(x, y, alpha=1e-12)
    np.testing.assert_allclose(readout.predict(x), y, atol=1e-9)


def test_chunked_ridge_matches_direct_grid_fit():
    rng = np.random.RandomState(4)
    x, y = rng.normal(size=(31, 5)), rng.normal(size=(31, 2))
    direct = fit_ridge(x, y, 3.)
    chunked = fit_ridge_sufficient(((x[:7], y[:7]), (x[7:19], y[7:19]), (x[19:], y[19:])), 3.)
    np.testing.assert_allclose(chunked.predict(x), direct.predict(x), atol=1e-11, rtol=1e-11)


def test_dual_ridge_grid_matches_primal_for_each_alpha_when_labels_fewer_than_features():
    rng = np.random.RandomState(14)
    x, y = rng.normal(size=(7, 19)), rng.normal(size=(7, 2))
    for alpha, dual in fit_ridge_grid_adaptive(x, y, (1., 10., 100.)).items():
        primal = fit_ridge_sufficient(((x, y),), alpha)
        np.testing.assert_allclose(dual.predict(x), primal.predict(x), rtol=1e-10, atol=1e-10)


def _record(session_id, neural, *, split="train"):
    t = len(neural)
    velocity = np.c_[np.arange(t, dtype=float), -np.arange(t, dtype=float)]
    return Record(session_id, neural, velocity, np.arange(4, t), np.arange(4), np.arange(4),
                  np.array([2, 5, 7]), split=split)


def test_fsu_fit_never_needs_target_velocity_and_wf_fss_uses_only_carrier_indices():
    source = _record("20150716", np.arange(30, dtype=float).reshape(10, 3))
    target = _record("dev", np.arange(30, 60, dtype=float).reshape(10, 3), split="dev")
    model = fit_baseline("diag_z_wf", [source], {"smooth": False, "history_bins": 1, "alpha": 1.})
    # If predict touched target Y/velocity this property would fail; its target
    # velocity is deliberately replaced by a non-array sentinel.
    target.velocity = "target labels must not be read"
    got = predict_baseline(model, target)
    assert got.shape == (6, 2)

    target.velocity = np.c_[np.arange(10, dtype=float), np.arange(10, dtype=float)]
    local = fit_wf_fss(target, {"smooth": False, "history_bins": 1, "alpha": 1.})
    assert predict_wf_fss(local, target).shape == (6, 2)


def test_source_guard_accepts_real_2015_cache_id_and_rejects_2013():
    source = _record("sub-C_ses-CO-20150716", np.arange(30, dtype=float).reshape(10, 3))
    fit_baseline("diag_z_wf", [source], {"smooth": False, "history_bins": 1, "alpha": 1.}, reference_session=source.session_id)
    source.session_id = "sub-C_ses-CO-20131003"
    with pytest.raises(ValueError, match="2015-only"):
        fit_baseline("diag_z_wf", [source], {"smooth": False, "history_bins": 1, "alpha": 1.}, reference_session=source.session_id)


def test_fss_m33_calibration_is_invariant_to_all_support_external_neural_values():
    record = _record("20150716", np.arange(60, dtype=float).reshape(20, 3))
    record.support_indices = np.arange(5, 15)
    record.carrier_indices = np.arange(7, 13)
    first = fit_wf_fss(record, {"smooth": True, "history_bins": 5, "alpha": 10.})
    changed = _record("20150716", record.neural.copy())
    changed.support_indices, changed.carrier_indices = record.support_indices, record.carrier_indices
    changed.neural[:5] += 1e6; changed.neural[15:] -= 1e6
    second = fit_wf_fss(changed, {"smooth": True, "history_bins": 5, "alpha": 10.})
    np.testing.assert_allclose(first.readout.coef_, second.readout.coef_, rtol=0, atol=1e-12)
    np.testing.assert_allclose(first.readout.intercept_, second.readout.intercept_, rtol=0, atol=1e-12)


def test_diag_uses_the_same_reference_overlap_as_coral():
    source = _record("20150716", np.arange(30, dtype=float).reshape(10, 3))
    target = _record("dev", np.arange(30, 60, dtype=float).reshape(10, 3), split="dev")
    target.channel_indices = np.array([2, 5, 8])
    model = fit_baseline("diag_z_wf", [source], {"smooth": False, "history_bins": 1, "alpha": 1.})
    assert model.source_diagnostics["20150716"]["observed_count"] == 3
    # At target inference only the reference overlap {2, 5} is embedded.
    from dandi688_bench_v2.baselines import _target_adapter
    adapter = _target_adapter("diag_z_wf", model.reference_support, model.reference_indices, target,
                              target.neural, model.config)
    assert adapter.diagnostics["observed_count"] == 2


def test_static_adapter_preserves_target_compact_order_and_reports_partial_reference_coverage():
    source = _record("20150716", np.arange(30, dtype=float).reshape(10, 3))
    target = _record("dev", np.arange(30, 60, dtype=float).reshape(10, 3), split="dev")
    adapter = fitstatic_adapter([source], target, kind="diag_z")
    got = adapter.transform(target.neural)
    assert got.shape == target.neural.shape
    target.channel_indices = np.array([2, 5, 8])
    partial = fitstatic_adapter([source], target, kind="coral")
    transformed = partial.transform(target.neural)
    assert partial.diagnostics["coverage_fraction"] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(transformed[:, 2], target.neural[:, 2])
