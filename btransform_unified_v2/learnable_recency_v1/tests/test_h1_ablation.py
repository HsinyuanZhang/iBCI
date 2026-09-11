"""Identity-ablation unit tests for the H1 learnable-recency runner (CPU only).

Covers the three load-bearing laws of the ACTIVITY_ONLY / NORM_ONLY bank
transforms: carrier literally zero, NORM_ONLY E0 rows equal the z broadcast,
and the shared base bank object is never mutated.  All cases run on small
synthetic arrays; no NWB record, materializer checkpoint, or GPU is touched.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "h1_ablation_train.py"
_spec = importlib.util.spec_from_file_location("_h1_identity_ablation_train", RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def _toy_bank(n_units: int = 4, e0_dim: int = 8):
    rng = np.random.default_rng(7)
    return runner.signed.TaskBank(
        "toy-session",
        rng.normal(size=(n_units, e0_dim)).astype(np.float32),
        rng.normal(size=(n_units, 4)).astype(np.float32),
        np.ones(n_units, bool),
        rng.normal(size=(3, 5, n_units)).astype(np.float32),
        rng.normal(size=(3, 2)).astype(np.float32),
        np.arange(3, dtype=np.int64),
        {"shape": (n_units, e0_dim), "trial_count": 3, "budget": 3,
         "estimator": "toy", "array_sha256": "a", "carrier_sha256": "b"},
    )


def test_pooled_support_rate_totals_over_counts():
    activity = np.zeros((2, 3, 4), np.float32)
    activity[0, :, 0] = 1.0          # 3 counts over 6 bins -> 0.5
    activity[1, 1, 1] = 6.0          # 6 counts over 6 bins -> 1.0
    activity[:, :, 2] = 0.5          # 3 counts over 6 bins -> 0.5
    rate = runner.pooled_support_rate(activity)
    assert rate.shape == (4,)
    assert rate[0] == pytest.approx(0.5)
    assert rate[1] == pytest.approx(1.0)
    assert rate[2] == pytest.approx(0.5)
    assert rate[3] == pytest.approx(0.0)


def test_pooled_support_rate_rejects_degenerate_tensor():
    with pytest.raises(ValueError):
        runner.pooled_support_rate(np.zeros((0, 3, 4)))
    with pytest.raises(ValueError):
        runner.pooled_support_rate(np.zeros((3, 4)))


def test_fit_source_rate_stats_population_moments_and_floor_count():
    rows = {"s0": np.array([1.0, 0.0, 4.0]), "s1": np.array([3.0, 0.0, 6.0]), "s2": np.array([2.0, 0.0, 8.0])}
    stats = runner.fit_source_rate_stats(rows)
    assert stats["n_source_sessions"] == 3
    assert np.allclose(stats["mu_src"], [2.0, 0.0, 6.0])
    # ddof=0 population std, per unit
    assert np.allclose(stats["sigma_src"], [np.sqrt(2.0 / 3.0), 0.0, np.sqrt(8.0 / 3.0)])
    assert stats["n_sigma_floored"] == 1  # the always-zero unit


def test_norm_only_identity_row_equals_z_broadcast_and_carrier_zero():
    activity = np.full((3, 10, 4), 0.25, np.float32)
    activity[:, :, 1] = 0.75
    mu = np.array([0.2, 0.5, 0.25, 0.25])
    sigma = np.array([0.1, 0.25, 1e-9, 0.05])  # unit 2 must hit the eps floor
    e0, carrier, info = runner.norm_only_identity(activity, mu, sigma, e0_dim=6, n_units=4)
    rate = np.full(4, 0.25)
    rate[1] = 0.75
    expected_z = (rate - mu) / np.maximum(sigma, runner.SIGMA_FLOOR)
    assert e0.shape == (4, 6) and e0.dtype == np.float32 and e0.flags["C_CONTIGUOUS"]
    for unit in range(4):
        assert e0[unit].tolist() == pytest.approx([expected_z[unit]] * 6, rel=1e-6)
    assert carrier.shape == (4, 4)
    assert np.array_equal(carrier, np.zeros((4, 4), np.float32))
    assert info["max_abs_z"] == pytest.approx(float(np.abs(expected_z).max()))
    assert info["n_sigma_floored_slots"] == 1


def test_norm_only_identity_rejects_width_drift():
    with pytest.raises(RuntimeError):
        runner.norm_only_identity(np.zeros((2, 5, 3)), np.zeros(4), np.ones(4), e0_dim=6, n_units=4)
    with pytest.raises(RuntimeError):
        runner.norm_only_identity(np.zeros((2, 5, 4)), np.zeros(3), np.ones(3), e0_dim=6, n_units=4)


def test_activity_only_identity_passes_zero_side_and_fails_closed():
    seen = {}

    def fake_materialize(activity, carrier):
        seen["activity"] = np.array(activity, copy=True)
        seen["carrier_in"] = np.array(carrier, copy=True)
        out = np.arange(activity.shape[2] * 700, dtype=np.float32).reshape(activity.shape[2], 700)
        return out, np.array(carrier, copy=True)

    activity = np.random.default_rng(0).normal(size=(3, 1024, 4)).astype(np.float32)
    e0, carrier = runner.activity_only_identity(activity, fake_materialize, n_units=4)
    assert np.array_equal(seen["carrier_in"], np.zeros((4, 4), np.float32)), "side carrier input must be literal zero"
    assert np.array_equal(seen["activity"], activity)
    assert np.array_equal(e0, np.arange(4 * 700, dtype=np.float32).reshape(4, 700))
    assert np.array_equal(carrier, np.zeros((4, 4), np.float32))

    def bad_materialize(activity, carrier):
        return np.zeros((4, 700), np.float32), np.ones((4, 4), np.float32)

    with pytest.raises(RuntimeError, match="zero carrier"):
        runner.activity_only_identity(activity, bad_materialize, n_units=4)
    with pytest.raises(RuntimeError, match="width"):
        runner.activity_only_identity(np.zeros((3, 1024, 5), np.float32), fake_materialize, n_units=4)


def test_replace_bank_identity_never_mutates_base():
    base = _toy_bank()
    e0_before = base.E0.copy()
    carrier_before = base.carrier.copy()
    meta_before = dict(base.calibration_meta)
    new_e0 = np.full((4, 8), 2.5, np.float32)
    new_carrier = np.zeros((4, 4), np.float32)
    bank = runner.replace_bank_identity(base, new_e0, new_carrier, {"identity_ablation": "norm_only", "estimator": "toy-z"})
    # base untouched
    assert np.array_equal(base.E0, e0_before)
    assert np.array_equal(base.carrier, carrier_before)
    assert base.calibration_meta == meta_before
    # new bank carries the ablated identity, fresh arrays, merged meta
    assert np.array_equal(bank.E0, new_e0)
    assert np.array_equal(bank.carrier, new_carrier)
    assert not np.shares_memory(bank.E0, base.E0)
    assert not np.shares_memory(bank.carrier, base.carrier)
    assert bank.calibration_meta is not base.calibration_meta
    assert bank.calibration_meta["identity_ablation"] == "norm_only"
    assert bank.calibration_meta["estimator"] == "toy-z"
    assert bank.calibration_meta["array_sha256"] == runner.array_sha256(new_e0)
    assert bank.calibration_meta["carrier_sha256"] == runner.array_sha256(new_carrier)
    assert bank.session_id == base.session_id and np.array_equal(bank.unit_mask, base.unit_mask)


def test_norm_stats_meta_round_trip_preserves_frozen_statistics():
    rows = {f"s{i}": np.arange(6, dtype=np.float64) * (1.0 + i) for i in range(3)}
    stats = runner.fit_source_rate_stats(rows)
    stats["source_sessions"] = list(rows)
    stats["support"] = "toy"
    stats["rate_units"] = "counts per resampled bin"
    block = runner.norm_stats_meta_block(stats)
    # exact float64 mirror through JSON text
    round_trip = json.loads(json.dumps(block))
    loaded = runner.load_norm_stats_from_meta({"norm_only_source": round_trip})
    assert np.array_equal(loaded["mu_src"], stats["mu_src"])
    assert np.array_equal(loaded["sigma_src"], stats["sigma_src"])
    assert loaded["sigma_floor"] == runner.SIGMA_FLOOR


def test_load_norm_stats_from_meta_fails_closed_on_digest_drift():
    rows = {f"s{i}": np.arange(6, dtype=np.float64) + i for i in range(3)}
    stats = runner.fit_source_rate_stats(rows)
    stats["source_sessions"] = list(rows)
    stats["support"] = "toy"
    stats["rate_units"] = "counts per resampled bin"
    block = runner.norm_stats_meta_block(stats)
    block["mu_src"] = (np.asarray(block["mu_src"]) + 0.5).tolist()
    with pytest.raises(RuntimeError, match="digest drift"):
        runner.load_norm_stats_from_meta({"norm_only_source": block})
    with pytest.raises(RuntimeError):
        runner.load_norm_stats_from_meta({})


def test_identity_flag_is_required_and_ladder_defaults_to_default():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([])
    args = runner.build_parser().parse_args(["--identity", "activity_only"])
    assert args.identity == "activity_only"
    assert args.ladder == "default" and args.tier == "learned_slope" and args.layers == 4
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--identity", "full"])
