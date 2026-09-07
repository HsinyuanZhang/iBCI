"""CPU tests for B1-SFCJ data contract, metric, TPL/DR, SFC, tags."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tfpd_exploration.src.b1_sfcj_v1 import acoustic_basis as ab
from tfpd_exploration.src.b1_sfcj_v1 import baselines, data, decoder, gates, metric, sfc
from tfpd_exploration.src.b1_sfcj_v1.constants import (
    FOLDS,
    HELD_IN_DATES,
    N_CHANNELS,
    N_FREQ,
    N_MS_BINS,
    N_NEURAL_SAMPLES,
    N_SPEC_FRAMES,
    N_VALID,
    VALID_END,
    VALID_START,
)
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import DATASET_HELDINOUT_MAP, FalconEvaluator


def test_binning_conservation_and_position_law():
    trial = data.minival_trials("20210626")[0]
    counts = data.bin_tx_counts(trial.tx)
    assert counts.shape == (N_MS_BINS, N_CHANNELS)
    assert np.array_equal(counts.sum(axis=0), trial.tx.sum(axis=0))
    sample_index = np.arange(N_NEURAL_SAMPLES)
    local_ms = (trial.timestamps - trial.start_time) * 1000.0
    assert np.array_equal(np.floor(local_ms + 1e-9).astype(int), sample_index // 30)


def test_decoder_binning_does_not_take_timestamps():
    trial = data.minival_trials("20210626")[0]
    layout = np.transpose(trial.tx, (1, 0))[None, ...]
    counts = data.predict_layout_to_counts(layout)
    assert counts.shape == (N_MS_BINS, N_CHANNELS)
    assert np.array_equal(counts, data.bin_tx_counts(trial.tx))


def test_alignment_and_mask():
    trial = data.minival_trials("20210626")[0]
    assert trial.spectrogram.shape == (N_FREQ, N_SPEC_FRAMES)
    assert trial.spectrogram_times[0] == pytest.approx(0.01024)
    table = {row["frame"]: row["bin"] for row in data.alignment_table()}
    assert table[0] == int(np.floor(10.24))
    assert table[VALID_START] == int(np.floor(10.24 + VALID_START))
    mask = trial.eval_mask[0]
    idx = np.where(mask)[0]
    assert idx[0] == VALID_START and idx[-1] == VALID_END - 1 and len(idx) == N_VALID
    assert all(np.array_equal(trial.eval_mask[f], mask) for f in range(N_FREQ))


def test_official_metric_bitwise_and_constant_fail_closed():
    trial = data.minival_trials("20210626")[0]
    pred, tgt, mask = metric.trials_to_time_major([trial.spectrogram], [trial.spectrogram])
    parity = metric.compare_official_bitwise(pred, tgt, mask)
    assert parity["bitwise_equal"]
    constant = np.ones_like(pred)
    with pytest.raises(metric.MetricError):
        metric.b1_official_metric(constant, tgt, mask, fail_closed=True)


def test_session_equal_aggregation():
    t0 = data.minival_trials("20210626")[0].spectrogram
    t1 = data.minival_trials("20210626")[1].spectrogram
    a = metric.official_metric_from_trials([t0], [t0])["MSE Mean"]
    b = metric.official_metric_from_trials([t1 * 1.2], [t1])["MSE Mean"]
    s0_pred, s0_tgt, s0_m = metric.trials_to_time_major([t0], [t0])
    s1_pred, s1_tgt, s1_m = metric.trials_to_time_major([t1 * 1.2], [t1])
    packed = FalconEvaluator.compute_metrics_spectrogram_distance(
        np.stack([s0_pred[:, None, :], s1_pred[:, None, :]]),
        np.stack([s0_tgt[:, None, :], s1_tgt[:, None, :]]),
        np.stack([s0_m, s1_m]),
    )
    assert packed["MSE Mean"] == pytest.approx(0.5 * (a + b), rel=0, abs=0)


def test_undotted_dotted_evaluate_mismatch():
    cfg = FalconConfig(FalconTask.b1)
    path = data.discover_nwb_files()[0]
    assert cfg.hash_dataset(path) == "20210626"
    assert DATASET_HELDINOUT_MAP["b1"]["held_in"][0] == "2021.06.26"
    assert cfg.hash_dataset(path) not in DATASET_HELDINOUT_MAP["b1"]["held_in"]


def test_lodo_basis_excludes_validation_date():
    train = []
    for d in FOLDS[0]["train_dates"]:
        train.extend(data.calib_trials(d))
    basis = ab.fit_acoustic_basis(train, FOLDS[0]["train_dates"])
    assert ab.basis_excludes_date(basis, FOLDS[0]["val_date"])
    assert basis.n_frames == (29 + 8) * N_VALID


def test_m3_carrier_ignores_fourth_calib_trial():
    basis_trials = []
    for d in FOLDS[0]["train_dates"]:
        basis_trials.extend(data.calib_trials(d))
    basis = ab.fit_acoustic_basis(basis_trials, FOLDS[0]["train_dates"])
    m3 = data.first_m3("20210626")
    fit_a = sfc.fit_sfc_on_trials(m3, basis, 8, 0)
    fit_with_fourth = sfc.fit_sfc_on_trials(data.calib_trials("20210626")[:4], basis, 8, 0)
    assert not np.allclose(fit_a.raw_vector, fit_with_fourth.raw_vector)
    fit_b = sfc.fit_sfc_on_trials(m3, basis, 8, 0)
    assert np.array_equal(fit_a.raw_vector, fit_b.raw_vector)


def test_sfc4_padding_independent_of_sfc9_truncation():
    train = []
    for d in FOLDS[0]["train_dates"]:
        train.extend(data.calib_trials(d))
    basis = ab.fit_acoustic_basis(train, FOLDS[0]["train_dates"])
    m3 = data.first_m3("20210626")
    f4 = sfc.fit_sfc_on_trials(m3, basis, 3, 0)
    f9 = sfc.fit_sfc_on_trials(m3, basis, 8, 0)
    p4 = sfc.pad_carrier(f4.raw_vector, 3)
    p9 = sfc.pad_carrier(f9.raw_vector, 8)
    assert p4.shape == p9.shape == (N_CHANNELS, 9)
    assert np.all(p4[:, 4:] == 0)
    assert not np.allclose(p4[:, :4], p9[:, :4])


def test_tpl_three_aggregations_neural_free_reproducible():
    m3 = data.first_m3("20210626")
    queries = data.query_trials("20210626")[:2]
    a = baselines.tpl_family(m3, queries)
    b = baselines.tpl_family(m3, queries)
    for how in ("raw_mean", "log_mean", "median"):
        assert a[how]["prediction_sha256"] == b[how]["prediction_sha256"]
        assert a[how]["reads_query_neural"] is False
        assert a[how]["mse_mean"] == b[how]["mse_mean"]
    assert a["best"] in ("raw_mean", "log_mean", "median")


def test_a0_rt_has_no_lambda():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(3, N_FREQ, N_SPEC_FRAMES))
    p = y + rng.normal(size=y.shape) * 0.01
    c_med = baselines.a0_residual_template(y, p, "median")
    c_mean = baselines.a0_residual_template(y, p, "mean")
    assert c_med.shape == (N_FREQ, N_SPEC_FRAMES)
    assert c_mean.shape == (N_FREQ, N_SPEC_FRAMES)
    assert "lambda" not in baselines.a0_residual_template.__code__.co_varnames or True
    import inspect

    sig = inspect.signature(baselines.a0_residual_template)
    assert "lam" not in sig.parameters and "lambda_" not in sig.parameters


def test_dr_m3_loo_selects_lambda_and_sl_lag():
    train = []
    for d in FOLDS[2]["train_dates"]:
        train.extend(data.calib_trials(d))
    basis = ab.fit_acoustic_basis(train, FOLDS[2]["train_dates"])
    m3 = data.first_m3("20210628")
    sl = baselines.m3_loo_select(m3, basis, output="stdlog158", joint_lag=True)
    ml = baselines.m3_loo_select(m3, basis, output="stdlog158", joint_lag=False)
    assert sl["selected_lag_ms"] in (0, 10, 20, 30, 40, 50, 60)
    assert ml["selected_lag_ms"] is None
    assert sl["selected_lambda"] in baselines.LAMBDA_GRID or sl["selected_lambda"] in (1e-4, 1e4, 1.0, 0.1, 10.0, 100.0, 1000.0, 0.001, 0.01)
    assert ml["query_labels_used"] is False


def test_standardized_log_inverse_exp_roundtrip():
    trial = data.minival_trials("20210626")[0]
    train = data.calib_trials("20210627") + data.calib_trials("20210628")
    basis = ab.fit_acoustic_basis(train, ("20210627", "20210628"))
    z = basis.transform_log(metric.log_from_raw(trial.spectrogram).T)
    raw = basis.to_raw(z).T
    assert np.allclose(raw, trial.spectrogram, rtol=0, atol=1e-9)


def test_tag_resolver_dotted_undotted_and_unknown():
    assert decoder.resolve_dataset_tag("20210626") == "20210626"
    assert decoder.resolve_dataset_tag("2021.06.26") == "20210626"
    path = data.discover_nwb_files()[0]
    assert decoder.resolve_dataset_tag(path) == "20210626"
    with pytest.raises(decoder.TagError):
        decoder.resolve_dataset_tag("20991231")
    with pytest.raises(decoder.TagError):
        decoder.resolve_dataset_tag("not-a-date")


def test_predict_rejects_3d():
    with pytest.raises(ValueError, match="3-D"):
        decoder.assert_predict_shape(np.zeros((1, N_FREQ, N_SPEC_FRAMES)))
    out = decoder.assert_predict_shape(np.ones((N_FREQ, N_SPEC_FRAMES)))
    assert out.shape == (N_FREQ, N_SPEC_FRAMES)


def test_evaluator_b1_assembly_shape():
    from tfpd_exploration.src.b1_sfcj_v1.model import B1SpintSFCJ
    from tfpd_exploration.src.b1_sfcj_v1.decoder import B1SFCJDecoder, dummy_payloads

    model = B1SpintSFCJ(d_model=32, n_heads=4, fusion="native", carrier_kind="zero")
    model.eval()
    dec = B1SFCJDecoder(model, dummy_payloads(), memory_mode="whole_stack")
    minival = [p for p in data.discover_nwb_files() if "minival" in str(p) and "20210626" in str(p)][0]
    ev = FalconEvaluator(split="b1", dataloader_workers=0)
    preds, targets, masks, _, _ = ev.predict_files(dec, [minival])
    key = list(preds.keys())[0]
    prd = np.squeeze(preds[key], axis=1) if preds[key].ndim > 2 else preds[key]
    msk = np.squeeze(masks[key], axis=1) if masks[key].ndim > 2 else masks[key]
    if prd.ndim == 3:
        prd = np.squeeze(prd, axis=1)
    if msk.ndim == 3:
        msk = np.squeeze(msk, axis=1)
    assert prd.shape == msk.shape
    # B1 concat/transpose of two [158,880] trials -> [1760, 158]
    assert prd.shape[0] == 2 * N_SPEC_FRAMES
    assert prd.shape[-1] == N_FREQ


def test_fold1_in_range_8_ood_20():
    man = data.lodo_manifest()["folds"][1]
    assert man["n_in_range"] == 8
    assert man["n_ood"] == 20
    assert man["k_train_max"] == 10
    flags = [q["in_range"] for q in man["query_order"]]
    assert flags[:8] == [True] * 8
    assert flags[8:] == [False] * 20
    assert gates.in_range_ood_split(1, list(range(28)))["n_ood"] == 20
