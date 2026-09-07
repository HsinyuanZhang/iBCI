from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tfpd_exploration"), str(ROOT / "tfpd_exploration/src"), str(ROOT / "sua_exploration")]

from src import low_cost_calibration_v1 as route


def test_scattered_exactly_matches_cue_multiset_and_spreads_time() -> None:
    directions = np.tile(np.arange(8, dtype=np.int64), 10)
    early = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=np.int64)
    result = route.cue_matched_scattered_indices(directions, early)
    assert sorted(directions[result].tolist()) == sorted(directions[early].tolist())
    assert len(set(result.tolist())) == len(early)
    assert int(result.max() - result.min()) > int(early.max() - early.min())


def test_support_rules_and_carrier_rows_are_deterministic() -> None:
    directions = np.tile(np.arange(8, dtype=np.int64), 10)
    theta = np.asarray([-3 * np.pi / 4 + int(item) * np.pi / 4 for item in directions])
    times = np.arange(theta.size, dtype=np.float64)
    rates = np.stack([
        5.0 + (unit + 1) * np.cos(theta) + 0.5 * np.sin(theta)
        for unit in range(5)
    ], axis=1)
    first = route.carrier_rows(rates=rates, thetas=theta, directions=directions, trial_times=times, budget=4)
    second = route.carrier_rows(rates=rates, thetas=theta, directions=directions, trial_times=times, budget=4)
    assert first["C1_cue_balanced_early"]["selected_indices"] == second["C1_cue_balanced_early"]["selected_indices"]
    assert first["C1_cue_balanced_early"]["selected_direction_counts"] == first["C2_cue_matched_scattered"]["selected_direction_counts"]
    assert first["C3_full_session_oracle"]["median_ac_cosine_vs_full"] == pytest.approx(1.0)


def test_score_decomposition_matches_manual_variance_weighted_r2() -> None:
    target = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 5.0], [3.0, 9.0]], dtype=np.float32)
    prediction = target + np.asarray([[0.1, -0.2], [0.0, 0.1], [-0.2, 0.0], [0.1, 0.1]], dtype=np.float32)
    result = route.score_decomposition(prediction, target)
    residual = np.sum((prediction - target) ** 2)
    total = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    assert result["variance_weighted_r2"] == pytest.approx(1.0 - residual / total)
    assert 1.0 <= result["target_participation_ratio"] <= 2.0


def test_aggregate_is_equal_session_not_window_weighted() -> None:
    rows = []
    for value in (0.1, 0.9):
        rows.append({
            "variance_weighted_r2": value,
            "coordinate_r2": [value, value],
            "eigenbasis_r2_descending": [value, value],
            "target_participation_ratio": 1.5,
        })
    result = route.aggregate_session_rows(rows)
    assert result["equal_session_mean_r2"] == pytest.approx(0.5)
    assert result["equal_session_median_r2"] == pytest.approx(0.5)


def test_dry_cli_is_static_and_does_not_import_torch() -> None:
    cli = ROOT / "tfpd_exploration/scripts/run_low_cost_calibration_v1.py"
    code = (
        "import runpy,sys,json; p=sys.argv.pop(1); runpy.run_path(p,run_name='__main__'); "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code, str(cli)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE_NO_LAUNCH"


def test_one_public_flag_fails_closed() -> None:
    cli = ROOT / "tfpd_exploration/scripts/run_low_cost_calibration_v1.py"
    completed = subprocess.run(
        [sys.executable, str(cli), "--execute"], cwd=ROOT, text=True, capture_output=True,
    )
    assert completed.returncode != 0
    assert "requires both" in completed.stderr


def test_actual_failed_v1_graph_is_exactly_bound() -> None:
    payload = route.validate_failed_v1(ROOT)
    assert payload["attempt_sha256"] == route.FAILED_V1_SHA256["attempt.json"]
    assert payload["failure_sha256"] == route.FAILED_V1_SHA256["failure.json"]
    assert payload["no_experimental_forward"] is True


def test_actual_failed_v2_graph_is_exactly_bound() -> None:
    payload = route.validate_failed_v2(ROOT)
    assert payload["attempt_sha256"] == route.FAILED_V2_SHA256["attempt.json"]
    assert payload["failure_sha256"] == route.FAILED_V2_SHA256["failure.json"]
    assert payload["prediction_sha_parity_passed_before_metric_check"] is True
