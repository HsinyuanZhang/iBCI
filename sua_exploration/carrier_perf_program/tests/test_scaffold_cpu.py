"""CPU-only tests for carrier_perf_program scaffold."""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pytest

from carrier_perf.gpu_guard import GUARD_ENV, force_cpu, require_reviewed_cpu
from carrier_perf.p0n_noise_floor import (
    build_p0n_report,
    discover_epoch_curves,
    lightning_to_protocol_epoch,
    parse_lightning_epoch_r2,
    protocol_window_from_lightning,
)
from carrier_perf.p1_estimators import (
    compare_estimators_on_arrays,
    fit_estimator_a_bin_means,
    fit_estimator_b_per_trial,
    nearest_canonical_direction_index,
    synthesize_cosine_unit,
)
from carrier_perf.p3_corruption import (
    CorruptionConfig,
    apply_corruption,
    corrupt_batch,
    sample_kind,
)
from carrier_perf.p5a_degeneracy import apply_degeneracy_policy, decide_direction_degeneracy
from carrier_perf.protocol import PRACTICAL_EFFECT_FLOOR, sigma_delta_paired


def test_force_cpu_blanks_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    force_cpu()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_require_reviewed_cpu_blocks_without_guard(monkeypatch):
    monkeypatch.delenv(GUARD_ENV, raising=False)
    with pytest.raises(RuntimeError, match="CARRIER_PERF_REVIEWED_CPU"):
        require_reviewed_cpu(True)


def test_no_prewritten_effect_floor():
    assert PRACTICAL_EFFECT_FLOOR is None


def test_sigma_delta_paired_requires_two_seeds():
    with pytest.raises(ValueError, match="at least 2 seeds"):
        sigma_delta_paired([0.1])


def test_sigma_delta_paired_value():
    se = sigma_delta_paired([0.0, 0.2])
    assert se == pytest.approx(math.sqrt(0.02) / math.sqrt(2.0))


def test_estimator_a_recovers_noiseless_canonical():
    angles, rates = synthesize_cosine_unit(
        n_trials=40, a=1.5, c=-0.5, b=2.0, seed=1, rate_noise_std=0.0
    )
    fit = fit_estimator_a_bin_means(angles, rates)
    assert fit.status == "ok"
    assert fit.a == pytest.approx(1.5, abs=1e-6)
    assert fit.c == pytest.approx(-0.5, abs=1e-6)
    assert fit.b == pytest.approx(2.0, abs=1e-6)


def test_estimator_b_recovers_noiseless_continuous():
    angles, rates = synthesize_cosine_unit(
        n_trials=40,
        a=1.5,
        c=-0.5,
        b=2.0,
        seed=2,
        angle_mode="continuous_uniform",
        rate_noise_std=0.0,
    )
    fit = fit_estimator_b_per_trial(angles, rates)
    assert fit.status == "ok"
    assert fit.a == pytest.approx(1.5, abs=1e-6)
    assert fit.c == pytest.approx(-0.5, abs=1e-6)
    assert fit.b == pytest.approx(2.0, abs=1e-6)


def test_balanced_canonical_a_equals_b():
    angles, rates = synthesize_cosine_unit(
        n_trials=40, a=1.2, c=-0.7, b=3.0, seed=0, rate_noise_std=0.0
    )
    cmp = compare_estimators_on_arrays(angles, rates)
    assert cmp["comparable"] is True
    assert cmp["delta_A_minus_B"]["d_ac_l2"] == pytest.approx(0.0, abs=1e-9)


def test_estimators_diverge_under_imbalanced_off_canonical():
    angles, rates = synthesize_cosine_unit(
        n_trials=80,
        a=2.0,
        c=0.5,
        b=1.0,
        seed=3,
        angle_mode="continuous_imbalanced",
        rate_noise_std=0.05,
    )
    cmp = compare_estimators_on_arrays(angles, rates)
    assert cmp["comparable"] is True
    assert cmp["delta_A_minus_B"]["d_ac_l2"] > 0.0


def test_nan_angles_dropped_shared_mask_not_snapped_to_bin0():
    with pytest.raises(ValueError, match="non-finite"):
        nearest_canonical_direction_index(float("nan"))
    angles = np.asarray(
        [np.nan, np.nan, math.pi / 2, math.pi / 2, 0.0, 0.0, -math.pi / 2, -math.pi / 2],
        dtype=np.float64,
    )
    rates = np.asarray([99.0, 99.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float64)
    cmp = compare_estimators_on_arrays(angles, rates)
    assert cmp["shared_n_dropped"] == 2
    assert cmp["A"]["n_dropped_nonfinite"] == 2
    assert cmp["B"]["n_dropped_nonfinite"] == 2
    # Without NaN pollution, A should not explode toward the 99 rates.
    assert abs(cmp["A"]["a"]) < 10.0


def test_estimator_a_rank1_returns_zeros():
    angles = np.zeros(10)
    rates = np.ones(10)
    fit = fit_estimator_a_bin_means(angles, rates)
    assert fit.status == "degenerate_zeros"
    assert fit.a == 0.0 and fit.c == 0.0 and fit.m == 0.0
    assert fit.present_directions == 1


def test_estimator_b_two_directions_undefined_not_raise():
    angles = np.concatenate([np.zeros(5), np.full(5, math.pi / 2)])
    rates = np.ones(10)
    fit = fit_estimator_b_per_trial(angles, rates, strict_rank=True)
    assert fit.status == "b_undefined"
    cmp = compare_estimators_on_arrays(angles, rates)
    assert cmp["comparable"] is False
    assert cmp["delta_A_minus_B"] is None


def test_corruption_probs_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        CorruptionConfig(p_identity=0.5, p_row_permute=0.5, p_gaussian_noise=0.5, p_zero=0.5).validate()


def test_row_permute_changes_order_for_n_ge_2():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    rng = np.random.RandomState(0)
    y = apply_corruption(x, "row_permute", rng=rng, noise_std=1.0)
    assert y.shape == x.shape
    assert not np.array_equal(y, x)
    assert sorted(map(tuple, y.tolist())) == sorted(map(tuple, x.tolist()))


def test_zero_corruption():
    x = np.ones((4, 4), dtype=np.float32)
    y = apply_corruption(x, "zero", rng=np.random.RandomState(0), noise_std=1.0)
    assert np.all(y == 0)


def test_corrupt_batch_receipt():
    x = np.ones((5, 4), dtype=np.float32)
    out = corrupt_batch(x, CorruptionConfig(), seed=123)
    assert out["kind"] in {"identity", "row_permute", "gaussian_noise", "zero"}
    assert out["m_sampled"] in CorruptionConfig().m_grid
    assert out["features"].shape == x.shape


def test_sample_kind_respects_identity_mass():
    cfg = CorruptionConfig(
        p_identity=1.0, p_row_permute=0.0, p_gaussian_noise=0.0, p_zero=0.0
    )
    rng = np.random.RandomState(0)
    kinds = {sample_kind(cfg, rng) for _ in range(20)}
    assert kinds == {"identity"}


def test_p5a_raise_on_rank1():
    decision = decide_direction_degeneracy(
        present_directions=1, num_channels=64, mode="raise"
    )
    assert decision.action == "raise"
    assert decision.insufficient_direction == 64
    with pytest.raises(ValueError, match="degenerate"):
        apply_degeneracy_policy(decision)


def test_p5a_ok_when_enough_directions():
    decision = decide_direction_degeneracy(
        present_directions=8, num_channels=64, mode="raise"
    )
    assert decision.action == "proceed"
    assert decision.message == "ok"
    apply_degeneracy_policy(decision)


def test_p5a_warn_and_fill_emits_warning():
    decision = decide_direction_degeneracy(
        present_directions=1, num_channels=8, mode="warn_and_fill"
    )
    assert decision.action == "warn_and_fill"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_degeneracy_policy(decision)
    assert caught
    assert "degenerate" in str(caught[0].message)


def test_p0n_fails_closed_without_logs(tmp_path):
    arts = {}
    for arm in ("f0", "t4", "ts4"):
        arts[arm] = {}
        for cell in ("fold1_seed42", "fold1_seed43", "fold2_seed42"):
            d = tmp_path / f"{arm}_{cell}"
            d.mkdir()
            (d / "metrics_summary.csv").write_text("run_id\n", encoding="utf-8")
            arts[arm][cell] = str(d)
    aggregate = {
        "artifacts": {"m2": arts},
        "scores": {
            "m2": {
                "f0": {"fold1_seed42": 0.5, "fold1_seed43": 0.4, "fold2_seed42": 0.45},
                "t4": {"fold1_seed42": 0.6, "fold1_seed43": 0.55, "fold2_seed42": 0.5},
                "ts4": {"fold1_seed42": 0.52, "fold1_seed43": 0.42, "fold2_seed42": 0.4},
            }
        },
    }
    report = build_p0n_report(aggregate, logs_dir=None)
    assert report["epoch_curves_found"] is False
    assert report["thresholds_frozen"] is False
    assert report["fail_closed_epoch_claim"] is True
    assert report["summary_score_deltas"]["not_v4_comparable"] is True
    assert report["missing_cell"] == "fold2_seed43"


def test_parse_lightning_log_and_protocol_window(tmp_path):
    log = tmp_path / "m2_t4_f1_s42.log"
    lines = []
    for lightning in range(12):
        # protocol epoch = lightning + 1; invent monotone values
        r2 = 0.4 + 0.02 * lightning
        lines.append(f"Epoch {lightning}: 100%|████| 10/10 [00:01<00:00, 1it/s] val_heldin/r2_mean={r2:.3f}\n")
    log.write_text("".join(lines), encoding="utf-8")
    by = parse_lightning_epoch_r2(log)
    assert sorted(by) == list(range(12))
    assert lightning_to_protocol_epoch(4) == 5
    protocol = protocol_window_from_lightning(by)
    assert list(protocol) == list(range(5, 13))


def test_discover_epoch_curves_empty(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    found, source, notes = discover_epoch_curves(d)
    assert found is False
    assert source is None
    assert notes
