"""Tests for the calibration-gap route foundation (ledger + coverage).

CPU-only; no torch. Every assertion loads real sealed receipts through the
SHA-verified ledger — no hardcoded performance numbers anywhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_pkg():
    spec = importlib.util.spec_from_file_location(
        "calibration_gap_v1", ROOT / "src/calibration_gap_v1/__init__.py",
        submodule_search_locations=[str(ROOT / "src/calibration_gap_v1")],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["calibration_gap_v1"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


@pytest.fixture(scope="module")
def pkg():
    return _load_pkg()


@pytest.fixture(scope="module")
def ledger(pkg):
    return pkg.load_ledger()


# -- ledger ---------------------------------------------------------------
def test_ledger_loads_and_verifies_all_pinned_receipts(ledger, pkg):
    assert len(ledger.bodies) == len(pkg.RECEIPTS)
    for rel in pkg.RECEIPTS:
        assert "cells" in ledger.bodies[rel] or "status" in ledger.bodies[rel]


def test_ledger_refuses_tampered_receipt(pkg, tmp_path, monkeypatch):
    (tmp_path / "results").mkdir(parents=True)
    for rel in pkg.RECEIPTS:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    victim = tmp_path / "results/calibration_budget_comparators_v1/receipt.json"
    body = victim.read_text().replace("equal_session_mean_r2", "tampered_r2", 1)
    victim.write_text(body)
    with pytest.raises(pkg.LedgerError, match="SHA drift"):
        pkg.load_ledger(tmp_path)


def test_cell_resolution_requires_unique_match(ledger):
    with pytest.raises(Exception):
        ledger.cell("protocol_factorial_v2", surface="external")  # ambiguous


def test_ladder_m4_matches_sealed_structure(ledger):
    m4 = ledger.ladder_external(4)
    # structural identities computed FROM loaded rungs, not hardcoded values
    assert m4["carrier_term"] == pytest.approx(
        m4["oracle_full_session_labels"]
        - m4["best_label_limited_dopt_ridge"], abs=1e-12)
    assert m4["activity_term"] == pytest.approx(
        m4["best_label_limited_dopt_ridge"]
        - m4["honest_total_calibration_dopt_ridge"], abs=1e-12)
    assert m4["total_gap"] == pytest.approx(
        m4["carrier_term"] + m4["activity_term"], abs=1e-12)
    # the decomposition ordering must hold at M4 on the external surface
    assert m4["oracle_full_session_labels"] > m4["best_label_limited_dopt_ridge"]
    assert m4["best_label_limited_dopt_ridge"] > m4["honest_total_calibration_dopt_ridge"]
    assert m4["carrier_term"] > m4["activity_term"]


def test_ladder_m30_headroom_is_small_relative_to_m4(ledger):
    m4, m30 = ledger.ladder_external(4), ledger.ladder_external(30)
    # D-opt exists only at M4; M30 rungs source from the comparators receipt
    assert m30["support_used"]["label_limited"] == "comparators/chronological"
    assert m30["support_used"]["honest"].startswith("comparators/label_limited(total==full@M30)")
    assert m4["support_used"]["label_limited"] == "factorial/doptimal_first30"
    # C0 stays chronological even where a D-opt OLS cell exists
    assert m4["c0_source"] == "factorial/chronological"
    # C3 was recorded at budgets {4, 30}; M10 reuses the budget-agnostic rung
    m10 = ledger.ladder_external(10)
    assert m10["oracle_cell_budget"] in (4, 30)
    assert m30["total_gap"] < m4["total_gap"] / 10


def test_missing_cells_registry_covers_z1_and_z5(ledger):
    ids = {m["id"] for m in ledger.missing_cells()}
    assert {"honest_M4_oracle", "honest_M10_oracle", "honest_M30_oracle",
            "h1_query_oracle_full_surface"} <= ids
    assert all(m["status"] == "MISSING" for m in ledger.missing_cells())


# -- coverage -------------------------------------------------------------
def test_coverage_covariates_extract_real_sessions(pkg, ledger):
    rec = pkg.coverage_covariates(ledger)
    assert rec.rows, "no coverage rows extracted"
    m4_ext_c0 = rec.frame("external", "C0_contiguous")
    assert any(r["budget"] == 4 for r in m4_ext_c0)
    row = next(r for r in m4_ext_c0 if r["budget"] == 4)
    for key in ("log_design_condition", "distinct_directions",
                "participation_ratio", "r2", "eigenbasis_r2"):
        assert key in row


def test_coverage_regression_runs_and_reports_rank(pkg, ledger):
    rec = pkg.coverage_covariates(ledger)
    result = pkg.coverage_regression(rec, "external", "C0_contiguous")
    assert result["n_sessions"] >= 6
    assert set(result["coef"]) == {"log_design_condition",
                                   "distinct_directions",
                                   "participation_ratio"}
    assert result["design_rank"] <= result["n_covariates"] + 1


def test_outlier_report_flags_low_performers(pkg, ledger):
    rec = pkg.coverage_covariates(ledger)
    report = pkg.outlier_report(rec, "external", "C0_contiguous",
                                threshold_std=1.5)
    assert "mean" in report and "flagged" in report
    for f in report["flagged"]:
        assert f["z"] <= -1.5


def test_per_dof_attribution_returns_eigenbasis(pkg, ledger):
    rec = pkg.coverage_covariates(ledger)
    rows = pkg.per_dof_attribution(rec, "external", "C0_contiguous", budget=4)
    assert rows
    assert all(len(r["eigenbasis_r2"]) == 2 for r in rows)  # 2D behavior


def test_pearson_math_on_synthetic_fixture(pkg):
    rows = [
        {"surface": "s", "support": "sup", "session": f"n{i}",
         "r2": float(i), "n_windows": 1,
         "log_design_condition": float(i), "distinct_directions": 2,
         "participation_ratio": 1.9, "index_span": 1,
         "coordinate_r2": [0.0, 0.0], "eigenbasis_r2": [0.0, 0.0]}
        for i in range(8)
    ]
    rec = pkg.CoverageRecords(rows=rows)
    result = pkg.coverage_regression(rec, "s", "sup")
    assert result["pearson_r"]["log_design_condition"] == pytest.approx(
        1.0, abs=1e-9)
    assert result["regression_r2"] > 0.99
