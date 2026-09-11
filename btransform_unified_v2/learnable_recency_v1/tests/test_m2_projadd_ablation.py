"""Identity-ablation unit tests: carrier zeroing, norm z-broadcast, no bank mutation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

from btransform_unified_v1.bank import make_synthetic_bank

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m2_projadd_ablation_train as train_mod  # noqa: E402
import m2_projadd_ablation_score as score_mod  # noqa: E402
from m2_projadd_ablation_train import ensure_frozen_recipe  # noqa: E402


def _bank(seed: int = 3):
    return make_synthetic_bank("m2", n_windows=4, seed=seed)


def _stats(n_sessions: int = 7, seed: int = 0):
    rng = np.random.default_rng(seed)
    rates = {f"ses-{i}": rng.uniform(0.2, 2.0, size=96) for i in range(n_sessions)}
    return rates


def test_script_constants_and_identity_choices() -> None:
    assert train_mod.IDENTITIES == ("full", "activity_only", "norm_only")
    assert train_mod.ABLATION_IDENTITIES == ("activity_only", "norm_only")
    assert train_mod.SIGMA_EPS == 1e-6
    assert train_mod.SUPPORT_SHAPE == (33, 100, 96)
    assert train_mod.SCHEMA == "m2_rift_projadd_ablation_train_v1"
    assert train_mod.CHECKPOINT_SCHEMA == "m2_rift_projadd_ablation_epoch_checkpoint_v1"
    assert train_mod.SMOKE_RECEIPT_SCHEMA == "m2_rift_projadd_ablation_smoke_receipt_v1"
    assert train_mod.TRAIN_RECEIPT_SCHEMA == "m2_rift_projadd_ablation_train_receipt_v1"
    assert score_mod.SELECTION_SCHEMA == "m2_rift_projadd_ablation_ext6_epoch_pick_v1_selection"
    assert score_mod.PROJ_DIM == 16
    assert score_mod.EXT6_RAW_M33.name == "m2_joint_ext6_raw_m33_v1"


def test_parser_rejects_full_and_freezes_recipe() -> None:
    parser = train_mod.build_parser()
    args = parser.parse_args([])
    assert args.identity == "full"  # rejected by ensure_ablation_identity / main
    with pytest.raises(ValueError):
        train_mod.ensure_ablation_identity("full")
    train_mod.ensure_ablation_identity("activity_only")
    train_mod.ensure_ablation_identity("norm_only")
    args = parser.parse_args(["--identity", "norm_only"])
    assert args.identity == "norm_only"
    assert args.tier == "learned_slope"
    assert args.ladder == "default"
    assert args.layers == 4
    ensure_frozen_recipe(parser.parse_args(["--identity", "norm_only"]))
    with pytest.raises(ValueError):
        ensure_frozen_recipe(parser.parse_args(["--identity", "activity_only", "--tier", "cable"]))
    with pytest.raises(ValueError):
        ensure_frozen_recipe(parser.parse_args(["--identity", "norm_only", "--ladder", "scaled"]))
    with pytest.raises(ValueError):
        ensure_frozen_recipe(parser.parse_args(["--identity", "norm_only", "--layers", "3"]))


def test_activity_only_zeroes_carrier_keeps_e0_and_does_not_mutate_original() -> None:
    bank = _bank()
    e0_before = bank.E0.copy()
    carrier_before = bank.carrier.copy()
    meta_before = dict(bank.calibration_meta)
    transformed = train_mod.transform_bank(bank, "activity_only")
    assert transformed is not bank
    assert np.all(transformed.carrier == 0.0)
    assert transformed.carrier.shape == bank.carrier.shape == (96, 4)
    assert np.array_equal(transformed.E0, e0_before)
    assert transformed.E0 is not bank.E0
    assert np.array_equal(bank.E0, e0_before)
    assert np.array_equal(bank.carrier, carrier_before)
    assert bank.calibration_meta == meta_before
    record = transformed.calibration_meta["identity_ablation"]
    assert record["identity"] == "activity_only"
    assert record["input_e0_sha256"] == bank.calibration_meta["array_sha256"]


def test_norm_only_broadcasts_z_and_matches_rate_formula() -> None:
    bank = _bank()
    rate = np.full(96, 1.7, dtype=np.float64)
    rate[5] = 0.3
    mu = np.full(96, 1.0, dtype=np.float64)
    sigma = np.full(96, 0.5, dtype=np.float64)
    transformed = train_mod.transform_bank(bank, "norm_only", rate=rate, mu=mu, sigma=sigma)
    assert np.all(transformed.carrier == 0.0)
    expected_z = (rate - mu) / np.maximum(sigma, train_mod.SIGMA_EPS)
    expected = expected_z.astype(np.float32)
    expected[~bank.unit_mask] = 0.0  # padding/masked rows carry E0 = 0
    assert transformed.E0.shape == (96, 50)
    # every row equals the broadcast z, all 50 columns identical
    np.testing.assert_array_equal(transformed.E0[:, 0], expected)
    np.testing.assert_array_equal(transformed.E0, np.broadcast_to(expected[:, None], (96, 50)))
    # original bank untouched
    assert not np.array_equal(bank.E0, transformed.E0)
    assert np.isfinite(transformed.E0).all()
    record = transformed.calibration_meta["identity_ablation"]["norm_only"]
    assert record["mu_sha256"] == train_mod.float64_sha256(mu)
    assert record["sigma_sha256"] == train_mod.float64_sha256(sigma)
    assert record["z_sha256"] == train_mod.float64_sha256(expected_z)
    assert record["rate_sha256"] == train_mod.float64_sha256(rate)


def test_norm_only_sigma_floor() -> None:
    bank = _bank()
    rate = np.full(96, 2.0, dtype=np.float64)
    mu = np.full(96, 1.0, dtype=np.float64)
    sigma = np.full(96, 1e-12, dtype=np.float64)  # below eps -> floored
    transformed = train_mod.transform_bank(bank, "norm_only", rate=rate, mu=mu, sigma=sigma)
    expected = np.float32((rate - mu) / train_mod.SIGMA_EPS)
    expected[~bank.unit_mask] = 0.0
    np.testing.assert_array_equal(transformed.E0[:, 0], expected)


def test_norm_only_zeroes_padding_rows() -> None:
    bank = _bank()
    mask = bank.unit_mask.copy()
    mask[:10] = False
    from btransform_unified_v1.bank import TaskBank

    padded = TaskBank(
        session_id=bank.session_id,
        E0=bank.E0,
        carrier=bank.carrier,
        unit_mask=mask,
        X_store=bank.X_store,
        target_store=bank.target_store,
        window_ids=bank.window_ids,
        calibration_meta=dict(bank.calibration_meta),
    )
    rate = np.linspace(0.1, 2.0, 96)
    mu = np.zeros(96)
    sigma = np.ones(96)
    transformed = train_mod.transform_bank(padded, "norm_only", rate=rate, mu=mu, sigma=sigma)
    assert np.all(transformed.E0[~mask] == 0.0)
    assert np.all(transformed.E0[mask] != 0.0)


def test_pooled_rate_matches_counts_over_duration() -> None:
    activity = np.zeros((33, 100, 96), dtype=np.float32)
    activity[:, :, 0] = 3.0
    activity[0, 0, 1] = 3300.0
    rate = train_mod.pooled_rate_from_activity(activity)
    assert rate.shape == (96,)
    assert rate[0] == pytest.approx(3.0)
    assert rate[1] == pytest.approx(3300.0 / 3300.0)
    assert rate[2] == pytest.approx(0.0)
    # total-counts / total-duration equivalence on a common grid
    counts = activity.sum(axis=(0, 1), dtype=np.float64)
    np.testing.assert_allclose(rate, counts / 3300.0, rtol=0, atol=0)


def test_source_z_stats_uses_only_source_sessions() -> None:
    rates = _stats(n_sessions=7)
    stats = train_mod.source_z_stats(rates)
    matrix = np.stack([rates[name] for name in sorted(rates)], axis=0)
    np.testing.assert_allclose(stats["mu_src"], matrix.mean(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(stats["sigma_src"], matrix.std(axis=0), rtol=0, atol=0)
    assert stats["source_sessions"] == sorted(rates)
    assert stats["mu_sha256"] == train_mod.float64_sha256(stats["mu_src"])
    assert stats["sigma_sha256"] == train_mod.float64_sha256(stats["sigma_src"])
    with pytest.raises(RuntimeError):
        train_mod.source_z_stats({k: v for k, v in list(rates.items())[:3]})  # wrong session count


def test_transform_bank_rejects_full_and_missing_stats() -> None:
    bank = _bank()
    with pytest.raises(ValueError):
        train_mod.transform_bank(bank, "full")
    with pytest.raises(ValueError):
        train_mod.transform_bank(bank, "norm_only")  # missing rate/mu/sigma


def test_score_side_stats_round_trip() -> None:
    rates = _stats(n_sessions=7, seed=1)
    stats = train_mod.source_z_stats(rates)
    meta = {
        "identity_ablation": {
            "identity": "norm_only",
            "norm_only": {
                "sigma_eps": train_mod.SIGMA_EPS,
                "mu_src": [float(v) for v in stats["mu_src"]],
                "sigma_src": [float(v) for v in stats["sigma_src"]],
                "mu_sha256": stats["mu_sha256"],
                "sigma_sha256": stats["sigma_sha256"],
                "rate_sha256": {"ext4": {}},
            }
        }
    }
    block, mu, sigma = score_mod.norm_stats_from_meta(meta)
    np.testing.assert_allclose(mu, stats["mu_src"], rtol=0, atol=0)
    np.testing.assert_allclose(sigma, stats["sigma_src"], rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        tampered = {"identity_ablation": {"norm_only": {**meta["identity_ablation"]["norm_only"], "sigma_eps": 1e-3}}}
        score_mod.norm_stats_from_meta(tampered)
    with pytest.raises(RuntimeError):
        drift = {"identity_ablation": {"norm_only": {**meta["identity_ablation"]["norm_only"], "mu_sha256": "0" * 64}}}
        score_mod.norm_stats_from_meta(drift)


@pytest.mark.skipif(not score_mod.EXT6_RAW_M33.is_dir(), reason="frozen ext6 raw-M33 directory not present")
def test_ext6_raw_m33_manifest_verifies() -> None:
    digests = score_mod.verify_ext6_m33_root(score_mod.EXT6_RAW_M33)
    assert set(digests) == set(__import__("scripts.rift_v1.m2_ext6_epoch_pick", fromlist=["SIX"]).SIX)
