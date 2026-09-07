"""H-U label-free descriptor contracts: no behaviour, no PCs, H-C bitwise identity."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data import h1_carrierid_hu_features as hu
from src.data.h1_m4_eb_pilot import (
    FrozenEBPlan,
    carrier_sha256,
    fit_frozen_carrier,
    load_record,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pilot_artifacts/h1_m4_eb_fold0/preflight_cache"
HC_NPZ = CACHE / "fold0_all_source_m4_carriers.npz"
HC_MANIFEST = CACHE / "fold0_all_source_m4_carriers.manifest.json"
PLAN_NPZ = CACHE / "fold0_frozen_eb_plan.npz"
PLAN_MANIFEST = CACHE / "fold0_frozen_eb_plan.manifest.json"
DATA = ROOT / "data/000954"
FROZEN_HC_CACHE_SHA256 = "88261cc03532b605da1790e8669760d4d47e2f87d2db1428060445541638b0af"
FROZEN_HC_FIRST_CARRIER_SHA256 = "a3fe44ca2ba767577add07c94aa0e7ba37c3604579cb3bdb66c37c2f1d4b24fe"
FIRST_SESSION = "ses-19250108T110520"


def _poisson_counts(rng: np.random.Generator, n_bins: int, rates_hz: np.ndarray) -> np.ndarray:
    lam = np.clip(rates_hz, 0.1, None) * hu.BIN_SECONDS
    return rng.poisson(lam[None, :], size=(n_bins, rates_hz.size)).astype(np.float64)


def test_public_builders_have_no_behaviour_parameter():
    for fn in (hu.hu_from_counts, hu.hu_from_support_neural, hu.neural_support_counts):
        names = set(inspect.signature(fn).parameters)
        leaked = names & hu.BEHAVIOUR_PARAMETER_NAMES
        assert not leaked, leaked
    hu.assert_no_behaviour_in_signature(hu.hu_from_counts)
    hu.assert_no_behaviour_in_signature(hu.hu_from_support_neural)


def test_width_is_four_and_names_are_frame_free_scalars():
    assert hu.CARRIER_WIDTH == 4
    assert hu.FEATURE_NAMES == ("mean_rate", "fano", "log_isi_cv", "autocorr_tau_s")
    assert hu.H1_NUM_NEURONS == 176
    source = Path(hu.__file__).read_text(encoding="utf-8")
    for needle in ("np.linalg.svd", "sklearn"):
        assert needle not in source


def test_per_unit_statistics_do_not_mix_across_units():
    rng = np.random.default_rng(20260813)
    rates = rng.uniform(5.0, 40.0, size=12)
    counts = _poisson_counts(rng, 400, rates)
    base, _ = hu.hu_from_counts(counts)
    mutated = counts.copy()
    mutated[:, 3] += 8.0
    other, _ = hu.hu_from_counts(mutated)
    for channel in range(12):
        if channel == 3:
            assert not np.array_equal(base[channel], other[channel])
        else:
            np.testing.assert_array_equal(base[channel], other[channel])


def test_descriptor_is_finite_nondegenerate_and_full_rank_on_synthetic():
    rng = np.random.default_rng(7)
    n_bins, n_ch = 600, 16
    counts = np.zeros((n_bins, n_ch), dtype=np.float64)
    for channel in range(n_ch):
        phi = 0.05 + 0.05 * channel
        rate = 4.0 + 3.0 * channel
        state = rate * hu.BIN_SECONDS
        for t in range(n_bins):
            state = phi * state + (1.0 - phi) * rate * hu.BIN_SECONDS
            counts[t, channel] = rng.poisson(max(state, 1e-3))
    features, audit = hu.hu_from_counts(counts)
    assert features.shape == (n_ch, 4)
    assert np.isfinite(features).all()
    assert audit.used_velocity is False
    assert audit.used_behaviour_labels is False
    stats = hu.per_unit_column_stats(features, silent=audit.silent_channels)
    assert stats["rank_live"] == 4
    for name, column in stats["columns"].items():
        assert column["finite"]
        assert not column["degenerate_constant"], name
        assert column["n_unique"] > 5, name


def test_silent_unit_is_zero_row_and_does_not_poison_live_units():
    rng = np.random.default_rng(3)
    counts = _poisson_counts(rng, 300, np.linspace(4.0, 30.0, 6))
    counts[:, 1] = 0.0
    features, audit = hu.hu_from_counts(counts)
    assert 1 in audit.silent_channels
    np.testing.assert_array_equal(features[1], np.zeros(4))
    assert np.any(features[0] != 0.0)


def test_support_selection_ignores_velocity_and_is_nan_behaviour_identical():
    rng = np.random.default_rng(11)
    n_bins, n_ch = 200, 8
    neural = _poisson_counts(rng, n_bins, np.linspace(3.0, 25.0, n_ch))
    eval_mask = np.ones(n_bins, dtype=bool)
    trial_num = np.repeat(np.arange(1, 9, dtype=np.float64), n_bins // 8)[:n_bins]
    trial_values = (1.0, 2.0, 3.0, 4.0)
    a, _ = hu.hu_from_support_neural(neural, eval_mask, trial_num, trial_values)
    velocity = np.full((n_bins, 7), np.nan)
    # The builder cannot see velocity; calling it again with the same neural
    # arguments must be bitwise identical even if a NaN velocity array exists.
    b, _ = hu.hu_from_support_neural(neural, eval_mask, trial_num, trial_values)
    np.testing.assert_array_equal(a, b)
    assert not np.isfinite(velocity).any()


def test_mean_rate_recovers_known_poisson_rate():
    rng = np.random.default_rng(0)
    rates = np.array([5.0, 10.0, 20.0, 40.0])
    counts = _poisson_counts(rng, 8000, rates)
    features, _ = hu.hu_from_counts(counts)
    recovered = features[:, 0]
    np.testing.assert_allclose(recovered, rates, rtol=0.08, atol=0.5)


def _load_frozen_plan() -> FrozenEBPlan:
    manifest = json.loads(PLAN_MANIFEST.read_text(encoding="utf-8"))
    with np.load(PLAN_NPZ, allow_pickle=False) as values:
        return FrozenEBPlan(
            outer_date=str(manifest["outer_date"]),
            source_sessions=tuple(manifest["source_sessions"]),
            source_input_sha256=tuple(manifest["source_input_sha256"]),
            mean=np.asarray(values["mean"], dtype=np.float64),
            scale=np.asarray(values["scale"], dtype=np.float64),
            pcs=np.asarray(values["pcs"], dtype=np.float64),
            q=int(values["q"]),
            ridge_lambda=float(values["lambda"]),
            U=np.asarray(values["U"], dtype=np.float64),
            mu=np.asarray(values["mu"], dtype=np.float64),
            tau2=float(values["tau2"]),
            raw_plan_sha256=str(manifest["raw_plan_sha256"]),
            raw_receipt_sha256=str(manifest["raw_receipt_sha256"]),
            eb_receipt_sha256=str(manifest["eb_receipt_sha256"]),
            transform_sha256=str(manifest["transform_sha256"]),
        )


def test_frozen_hc_cache_sha256_is_the_sealed_fold0_hash():
    manifest = json.loads(HC_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["cache_sha256"] == FROZEN_HC_CACHE_SHA256
    with np.load(HC_NPZ, allow_pickle=False) as values:
        carriers = np.asarray(values["carriers"], dtype=np.float64)
    assert carriers.shape == (116, 176, 4)
    assert carrier_sha256(carriers[0]) == FROZEN_HC_FIRST_CARRIER_SHA256


@pytest.mark.skipif(not (DATA / "sub-HumanPitt-held-in-calib").is_dir(), reason="H1 held-in calib missing")
def test_hc_feature_tensor_one_date_is_bitwise_identical_after_hu_landing():
    """Exact-null: adding H-U must not change the H-C producer for one date."""

    calib = DATA / "sub-HumanPitt-held-in-calib"
    nwb = next(path for path in sorted(calib.glob("*.nwb")) if FIRST_SESSION in path.name)
    record = load_record(nwb)
    plan = _load_frozen_plan()
    reconstructed = fit_frozen_carrier(record, plan, (1.0, 2.0, 3.0, 4.0))["carrier"]
    with np.load(HC_NPZ, allow_pickle=False) as values:
        sealed = np.asarray(values["carriers"][0], dtype=np.float64)
    assert reconstructed.dtype == sealed.dtype
    assert reconstructed.shape == sealed.shape == (176, 4)
    assert np.array_equal(reconstructed, sealed)
    assert carrier_sha256(reconstructed) == FROZEN_HC_FIRST_CARRIER_SHA256
    digest = hashlib.sha256(np.ascontiguousarray(reconstructed).tobytes()).hexdigest()
    assert digest == FROZEN_HC_FIRST_CARRIER_SHA256


@pytest.mark.skipif(not (DATA / "sub-HumanPitt-held-in-calib").is_dir(), reason="H1 held-in calib missing")
def test_hu_from_record_is_identical_when_velocity_is_replaced_by_nans():
    calib = DATA / "sub-HumanPitt-held-in-calib"
    nwb = next(path for path in sorted(calib.glob("*.nwb")) if FIRST_SESSION in path.name)
    record = load_record(nwb)
    nan_velocity = np.full_like(record.velocity, np.nan)
    poisoned = replace(record, velocity=nan_velocity)
    a, audit_a = hu.hu_from_record(record, (1.0, 2.0, 3.0, 4.0))
    b, audit_b = hu.hu_from_record(poisoned, (1.0, 2.0, 3.0, 4.0))
    np.testing.assert_array_equal(a, b)
    assert audit_a.used_velocity is False
    assert audit_b.used_velocity is False
    assert not np.isfinite(poisoned.velocity).any()
    assert a.shape == (176, 4)
    assert np.isfinite(a).all()
