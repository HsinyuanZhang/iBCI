from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tfpd_exploration/src", ROOT / "sua_exploration"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from budget_matched_posterior_cal_aug_c3_v1 import q_error_source as qes


def test_average_ranks_and_spearman_ties() -> None:
    assert qes._average_ranks(np.array([4.0, 1.0, 1.0, 9.0])).tolist() == [3.0, 1.5, 1.5, 4.0]
    assert qes.spearman_rho(np.arange(8.0), -np.arange(8.0)) == pytest.approx(-1.0)
    with pytest.raises(qes.QErrorSourceError, match="undefined"):
        qes.spearman_rho(np.ones(8), np.arange(8.0))


def _synthetic_session(seed: int, units: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = np.tile(np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False), 5)
    baseline = rng.uniform(3.0, 8.0, size=(units, 1))
    ac = rng.normal(size=(units, 2))
    clean = baseline + ac[:, :1] * np.cos(theta)[None, :] + ac[:, 1:] * np.sin(theta)[None, :]
    noise_scale = np.linspace(0.05, 1.5, units)[:, None]
    return np.ascontiguousarray(clean + rng.normal(size=clean.shape) * noise_scale), theta


def test_session_rows_are_disjoint_and_complete() -> None:
    rates, theta = _synthetic_session(7)
    rows = qes.session_rows("synthetic", rates, theta, prior_variance=1.0)
    assert [row["budget"] for row in rows] == [4, 10, 30]
    assert all(row["reference_rates_sha256"] == rows[0]["reference_rates_sha256"] for row in rows)
    assert len({row["support_rates_sha256"] for row in rows}) == 3
    assert all(row["defined_unit_count"] >= 8 for row in rows)
    changed = rates.copy()
    changed[:, 30:50] += np.linspace(-2.0, 2.0, changed.shape[0])[:, None] * np.sin(theta[30:50])
    changed_rows = qes.session_rows("synthetic", changed, theta, prior_variance=1.0)
    assert [row["posterior_raw_t4_sha256"] for row in changed_rows] == [
        row["posterior_raw_t4_sha256"] for row in rows
    ]
    assert changed_rows[0]["held_reference_ac_sha256"] != rows[0]["held_reference_ac_sha256"]


def test_summary_gate_and_m30_cannot_rescue() -> None:
    rows = []
    for session_index in range(27):
        for budget in qes.BUDGETS:
            rho = -0.4 if budget == 4 else (0.05 if budget == 10 else -0.8)
            rows.append({
                "session": f"s{session_index:02d}",
                "budget": budget,
                "rho_q_vs_absolute_angular_error": rho,
                "rho_defined": True,
                "defined_unit_count": 32,
                "defined_unit_fraction": 1.0,
            })
    summary = qes.summarize(rows)
    assert summary["by_budget"]["m4"]["mechanism_gate_passed"] is True
    assert summary["by_budget"]["m10"]["mechanism_gate_passed"] is False
    assert summary["by_budget"]["m30"]["mechanism_gate_passed"] is True
    assert summary["low_budget_any_passed"] is True
    assert summary["low_budget_both_passed"] is False
    assert summary["m30_disposition"].startswith("DESCRIPTIVE_ONLY")

    rows[0]["rho_q_vs_absolute_angular_error"] = None
    rows[0]["rho_defined"] = False
    rows[0]["rho_undefined_reason"] = "constant_q_or_angular_error_ranks"
    incomplete = qes.summarize(rows)
    assert incomplete["by_budget"]["m4"]["defined_correlation_session_count"] == 26
    assert incomplete["by_budget"]["m4"]["coverage_passed"] is False
    assert incomplete["by_budget"]["m4"]["mechanism_gate_passed"] is False


def test_review_drift_is_non_numerical() -> None:
    start = {"closure_sha256": "a", "files": {}}
    final = {"closure_sha256": "b", "files": {}}
    disposition = qes.review_drift(start, final)
    assert disposition["status"] == "ACCEPTED_NON_NUMERIC_DRIFT"
    assert disposition["numerical_acceptance_affected"] is False


def test_dry_plan_is_inert_and_torch_free() -> None:
    before = set(sys.modules)
    payload = qes.dry_plan()
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"
    assert payload["budgets"] == [4, 10, 30]
    assert payload["held_reference_positions"] == [30, 50]
    assert "torch" not in set(sys.modules) - before
    json.dumps(payload, allow_nan=False)


def test_static_cli_dry_run_needs_no_site_packages() -> None:
    completed = subprocess.run(
        [sys.executable, "-S", str(ROOT / "tfpd_exploration/scripts/run_budget_matched_posterior_q_error_source_v1.py"), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_GPU_NO_WRITE"


def test_current_source_authority_descriptor_graph() -> None:
    binding = qes.validate_source_authority(ROOT)
    assert binding["source_authority_sha256"] == qes.SOURCE_AUTHORITY_BODY_SHA256
    assert binding["exact_leaf_count"] == 6
    qes.source_prior_from_payload(binding["source_prior"])


def test_closures_are_disjoint_and_current() -> None:
    execution = qes.execution_closure(ROOT)
    review = qes.review_closure(ROOT)
    assert set(execution["files"]).isdisjoint(review["files"])
    assert execution["closure_sha256"] != review["closure_sha256"]
