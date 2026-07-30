"""Tests for aggregate_swa_windows_v1.py (E1 SWA-vs-plain-window-average aggregation).

No GPU, no NWB data, no torch: this aggregator only reads JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_swa_windows_v1 as agg  # noqa: E402

VAL_SESSIONS = [f"sub-X_ses-CO-2015110{i}" for i in range(1, 7)]
TRAIN_SESSIONS = [f"sub-X_ses-CO-2013100{i}" for i in range(1, 4)]
TEST_SESSIONS = [f"sub-X_ses-CO-2015120{i}" for i in range(1, 7)]
SESSION_SPLITS = {"train": TRAIN_SESSIONS, "val": VAL_SESSIONS, "test": TEST_SESSIONS}


def test_plain_window_average_matches_hand_computation():
    per_epoch_mean_r2 = {str(e): 0.30 + 0.01 * e for e in range(1, 41)}
    result = agg.plain_window_average(per_epoch_mean_r2, window=5, max_epoch=40)
    expected = sum(0.30 + 0.01 * e for e in range(36, 41)) / 5
    assert result == pytest.approx(expected)


def test_plain_window_average_window_20():
    per_epoch_mean_r2 = {str(e): float(e) for e in range(1, 41)}
    result = agg.plain_window_average(per_epoch_mean_r2, window=20, max_epoch=40)
    assert result == pytest.approx(sum(range(21, 41)) / 20)


def _write_swa(results_dir: Path, seed: int, scores: dict[int, float], run_dir: str) -> Path:
    payload = {
        "run_dir": run_dir,
        "variant": "B3",
        "seed": seed,
        "session_splits": SESSION_SPLITS,
        "per_window": {
            str(window): {
                "mean_r2": score,
                "swa_checkpoint_sha256": "0" * 64,
                "epochs_averaged": list(range(40 - window + 1, 41)),
            }
            for window, score in scores.items()
        },
    }
    path = results_dir / f"swa_s{seed}.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _write_curve(results_dir: Path, seed: int, constant_value: float, run_dir: str) -> Path:
    epochs = list(range(5, 41))
    payload = {
        "run_dir": run_dir,
        "variant": "B3",
        "seed": seed,
        "session_splits": SESSION_SPLITS,
        "epoch_list": epochs,
        "per_epoch_mean_r2": {str(e): constant_value for e in epochs},
    }
    path = results_dir / f"curve_s{seed}.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def test_full_pipeline_swa_reduces_std(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # SWA scores are nearly identical across seeds (small sigma_seed); the plain-average
    # reference (constant curve per seed, but each seed's constant differs) has larger
    # across-seed spread -- this is the "SWA wins" pattern E1 is looking for.
    swa_scores_by_seed = {42: {5: 0.360, 10: 0.361, 20: 0.362}, 43: {5: 0.359, 10: 0.360, 20: 0.361}, 44: {5: 0.361, 10: 0.362, 20: 0.360}}
    curve_constants = {42: 0.30, 43: 0.40, 44: 0.35}  # wide spread -> large plain-average std
    for seed in agg.SEEDS:
        _write_swa(results_dir, seed, swa_scores_by_seed[seed], run_dir=f"/fake/run_{seed}")
        _write_curve(results_dir, seed, curve_constants[seed], run_dir=f"/fake/run_{seed}")

    payload = agg.run_aggregation(results_dir)
    for window in agg.WINDOWS:
        w = payload["per_window"][str(window)]
        assert w["swa"]["across_seed_std"] < w["plain_epoch_window_average"]["across_seed_std"]
        assert w["across_seed_std_ratio_swa_over_plain"] < 1.0

    w5 = payload["per_window"]["5"]
    assert w5["swa"]["per_seed"]["42"] == pytest.approx(0.360)
    assert w5["swa"]["mean"] == pytest.approx((0.360 + 0.359 + 0.361) / 3)
    assert w5["plain_epoch_window_average"]["per_seed"]["42"] == pytest.approx(0.30)


def test_rejects_seeds_sharing_a_run_dir(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    scores = {5: 0.36, 10: 0.36, 20: 0.36}
    _write_swa(results_dir, 42, scores, run_dir="/fake/shared")
    _write_swa(results_dir, 43, scores, run_dir="/fake/shared")
    _write_swa(results_dir, 44, scores, run_dir="/fake/run_44")
    for seed in agg.SEEDS:
        _write_curve(results_dir, seed, 0.3, run_dir="/fake/whatever")
    with pytest.raises(ValueError, match="share a run directory"):
        agg.run_aggregation(results_dir)


def test_rejects_missing_window(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for seed in agg.SEEDS:
        scores = {5: 0.36, 10: 0.36} if seed == 44 else {5: 0.36, 10: 0.36, 20: 0.36}
        _write_swa(results_dir, seed, scores, run_dir=f"/fake/run_{seed}")
        _write_curve(results_dir, seed, 0.3, run_dir=f"/fake/run_{seed}")
    with pytest.raises(ValueError, match="windows"):
        agg.run_aggregation(results_dir)


def test_rejects_missing_artifact(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    scores = {5: 0.36, 10: 0.36, 20: 0.36}
    _write_swa(results_dir, 42, scores, run_dir="/fake/run_42")
    _write_curve(results_dir, 42, 0.3, run_dir="/fake/run_42")
    with pytest.raises(FileNotFoundError, match="swa_s43"):
        agg.run_aggregation(results_dir)
