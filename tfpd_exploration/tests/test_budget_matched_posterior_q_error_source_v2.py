from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tfpd_exploration/src", ROOT / "sua_exploration"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from budget_matched_posterior_cal_aug_c3_v1 import q_error_source as v1
from budget_matched_posterior_cal_aug_c3_v1 import q_error_source_v2 as v2


def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(-np.pi, np.pi, 50, endpoint=False, dtype=np.float64)
    unit = np.arange(16, dtype=np.float64)[:, None]
    rates = 2.0 + 0.4 * np.cos(theta)[None, :] + 0.2 * np.sin(theta)[None, :]
    rates = rates + 0.001 * unit * np.cos(2.0 * theta)[None, :]
    return np.ascontiguousarray(rates), theta


def test_failed_v1_graph_is_exact() -> None:
    binding = v2.validate_failed_v1(ROOT)
    assert binding["exact_leaf_count"] == 4
    assert binding["attempt_sha256"] == v2.FAILED_V1_ATTEMPT_SHA256
    assert binding["failure_sha256"] == v2.FAILED_V1_FAILURE_SHA256


def test_masked_route_matches_v1_when_all_directions_are_finite() -> None:
    rates, theta = _synthetic()
    expected = v1.session_rows("synthetic", rates, theta, prior_variance=1.0)
    actual = v2.session_rows_masked("synthetic", rates, theta, prior_variance=1.0)
    for left, right in zip(expected, actual, strict=True):
        assert left["budget"] == right["budget"]
        assert left["rho_q_vs_absolute_angular_error"] == right["rho_q_vs_absolute_angular_error"]
        assert right["nominal_support_trial_count"] == right["effective_support_trial_count"]
        assert right["invalid_support_positions"] == []


def test_position27_nan_preserves_m4_m10_and_discloses_effective_m29() -> None:
    rates, theta = _synthetic()
    theta[27] = np.nan
    rows = v2.session_rows_masked("synthetic", rates, theta, prior_variance=1.0)
    by_budget = {row["budget"]: row for row in rows}
    assert by_budget[4]["effective_support_trial_count"] == 4
    assert by_budget[10]["effective_support_trial_count"] == 10
    assert by_budget[30]["effective_support_trial_count"] == 29
    assert by_budget[30]["invalid_support_positions"] == [27]
    assert all(row["no_later_trial_refill"] is True for row in rows)


def test_execution_review_split() -> None:
    execution = v2.execution_closure(ROOT)
    review = v2.review_closure(ROOT)
    assert set(execution["files"]).isdisjoint(review["files"])


def test_static_v2_dry_cli() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_budget_matched_posterior_q_error_source_v2.py"
    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    assert payload["finite_direction_policy"] == "FIXED_FIRST50_MASK_NO_REFILL"
    assert payload["m30_disposition"].startswith("DESCRIPTIVE_ONLY")

