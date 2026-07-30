"""Tests for aggregate_convergence_swa_v1.py (E2 convergence-curve aggregation): the pure
arithmetic helpers (``ols_slope``, ``window_epochs``, ``summarize_window``) and one full
synthetic 3-seed pipeline through ``run_aggregation``.

No GPU, no NWB data, no torch: this aggregator only reads JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_convergence_swa_v1 as agg  # noqa: E402


def test_ols_slope_exact_line():
    xs = [5, 6, 7, 8, 9, 10, 11, 12]
    ys = [5 + 0.5 * (x - 5) for x in xs]
    assert agg.ols_slope(xs, ys) == pytest.approx(0.5)


def test_ols_slope_zero_for_constant_ys():
    xs = [5, 6, 7, 8]
    ys = [0.3, 0.3, 0.3, 0.3]
    assert agg.ols_slope(xs, ys) == pytest.approx(0.0)


def test_ols_slope_negative():
    xs = [1, 2, 3, 4]
    ys = [10, 8, 6, 4]
    assert agg.ols_slope(xs, ys) == pytest.approx(-2.0)


def test_ols_slope_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        agg.ols_slope([1, 2, 3], [1, 2])


def test_ols_slope_rejects_fewer_than_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        agg.ols_slope([1], [1])


def test_window_epochs_inclusive():
    assert agg.window_epochs((5, 12)) == [5, 6, 7, 8, 9, 10, 11, 12]
    assert agg.window_epochs((37, 40)) == [37, 38, 39, 40]


def test_summarize_window_matches_hand_computation():
    per_epoch_mean_r2 = {str(e): 0.30 + 0.01 * (e - 5) for e in range(5, 13)}
    summary = agg.summarize_window(per_epoch_mean_r2, (5, 12))
    assert summary["n_epochs"] == 8
    assert summary["partial"] is False
    assert summary["slope_per_epoch"] == pytest.approx(0.01)
    assert summary["cumulative_change"] == pytest.approx(0.01 * 7)
    assert summary["mean_r2"] == pytest.approx(agg.mean(list(per_epoch_mean_r2.values())))


def test_summarize_window_flags_partial_window():
    per_epoch_mean_r2 = {str(e): 0.5 for e in range(37, 41)}
    summary = agg.summarize_window(per_epoch_mean_r2, (37, 40))
    assert summary["n_epochs"] == 4
    assert summary["partial"] is True


# ----------------------------------------------------------------------------------------
# Full synthetic 3-seed pipeline.
# ----------------------------------------------------------------------------------------
VAL_SESSIONS = [f"sub-X_ses-CO-2015110{i}" for i in range(1, 7)]
TRAIN_SESSIONS = [f"sub-X_ses-CO-2013100{i}" for i in range(1, 4)]
TEST_SESSIONS = [f"sub-X_ses-CO-2015120{i}" for i in range(1, 7)]
SESSION_SPLITS = {"train": TRAIN_SESSIONS, "val": VAL_SESSIONS, "test": TEST_SESSIONS}
PROTOCOL = {
    "name": "fixed_forward_calibration_protocol_full_epoch_range",
    "expected_max_epochs": 40,
    "epoch_start": 5,
    "epoch_end": 40,
    "selection_mode": "first",
    "calibration_n": 30,
    "pool_size": 50,
}


def _write_curve(results_dir: Path, seed: int, base: float, slope: float, run_dir: str) -> Path:
    epochs = list(range(5, 41))
    per_epoch = {}
    per_epoch_mean_r2 = {}
    for epoch in epochs:
        value = base + slope * (epoch - 5)
        per_epoch[str(epoch)] = {
            "checkpoint_path": f"/fake/{seed}/epoch_{epoch - 1:03d}.ckpt",
            "checkpoint_sha256": "0" * 64,
            "per_session_r2": {s: value for s in VAL_SESSIONS},
            "mean_r2": value,
        }
        per_epoch_mean_r2[str(epoch)] = value
    payload = {
        "schema_version": 1,
        "run_dir": run_dir,
        "variant": "B3",
        "seed": seed,
        "epoch_list": epochs,
        "protocol": PROTOCOL,
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": per_epoch_mean_r2,
        "session_splits": SESSION_SPLITS,
    }
    path = results_dir / f"curve_s{seed}.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def test_full_pipeline_plateaued_curve(tmp_path):
    # Constant curve (slope 0 everywhere): every window's slope must be ~0, matching a
    # plateaued run.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for seed in agg.SEEDS:
        _write_curve(results_dir, seed, base=0.35, slope=0.0, run_dir=f"/fake/run_{seed}")

    payload = agg.run_aggregation(results_dir)
    assert payload["final_8_epoch_window_33_40"]["mean_slope_per_epoch"] == pytest.approx(0.0, abs=1e-9)
    for window_payload in payload["successive_8_epoch_windows"].values():
        assert window_payload["mean_slope_per_epoch"] == pytest.approx(0.0, abs=1e-9)
    assert payload["overall_5_to_40"]["mean_slope_per_epoch"] == pytest.approx(0.0, abs=1e-9)
    # Cross-seed std at every epoch is 0 (all 3 seeds identical curves here).
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in payload["cross_seed_std_curve"].values())


def test_full_pipeline_still_improving_curve(tmp_path):
    # Linear +0.002/epoch curve for all seeds: still improving at epoch 40.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for seed in agg.SEEDS:
        _write_curve(results_dir, seed, base=0.30, slope=0.002, run_dir=f"/fake/run_{seed}")

    payload = agg.run_aggregation(results_dir)
    assert payload["final_8_epoch_window_33_40"]["mean_slope_per_epoch"] == pytest.approx(0.002)
    assert payload["overall_5_to_40"]["mean_slope_per_epoch"] == pytest.approx(0.002)


def test_rejects_seeds_sharing_a_run_dir(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_curve(results_dir, 42, base=0.30, slope=0.0, run_dir="/fake/shared")
    _write_curve(results_dir, 43, base=0.30, slope=0.0, run_dir="/fake/shared")
    _write_curve(results_dir, 44, base=0.30, slope=0.0, run_dir="/fake/run_44")
    with pytest.raises(ValueError, match="share a run directory"):
        agg.run_aggregation(results_dir)


def test_rejects_missing_curve_artifact(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_curve(results_dir, 42, base=0.30, slope=0.0, run_dir="/fake/run_42")
    _write_curve(results_dir, 43, base=0.30, slope=0.0, run_dir="/fake/run_43")
    with pytest.raises(FileNotFoundError, match="curve_s44"):
        agg.run_aggregation(results_dir)


def test_rejects_mismatched_epoch_list(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_curve(results_dir, 42, base=0.30, slope=0.0, run_dir="/fake/run_42")
    _write_curve(results_dir, 43, base=0.30, slope=0.0, run_dir="/fake/run_43")
    _write_curve(results_dir, 44, base=0.30, slope=0.0, run_dir="/fake/run_44")
    payload = json.loads((results_dir / "curve_s44.json").read_text())
    payload["epoch_list"] = payload["epoch_list"][:-1]
    (results_dir / "curve_s44.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="epoch_list differs"):
        agg.run_aggregation(results_dir)
