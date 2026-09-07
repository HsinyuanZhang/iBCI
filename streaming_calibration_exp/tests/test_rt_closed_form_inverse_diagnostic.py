"""CPU contracts for the reviewer-only RT closed-form inverse diagnostic."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rt_closed_form_inverse_diagnostic.py"
PREFLIGHT = ROOT.parent / "sua_exploration/results/rt_closed_form_inverse_diagnostic_v1/RT_CLOSED_FORM_INVERSE_CPU_PREFLIGHT_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("rt_closed_form_inverse_diagnostic", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closed_form_inverse_matches_explicit_diagonal_formula() -> None:
    module = _module()
    rng = np.random.default_rng(4)
    w = rng.normal(size=(7, 2))
    b = rng.normal(size=7)
    sigma = rng.uniform(0.1, 2.0, size=7)
    rates = rng.normal(size=(11, 7))
    observed = module.inverse_velocity(rates, b, w, sigma, ridge_lambda=0.25)
    precision = np.diag(1.0 / sigma)
    expected = np.linalg.solve(w.T @ precision @ w + 0.25 * np.eye(2), w.T @ precision @ (rates - b).T).T
    assert np.allclose(observed, expected)


def test_full_covariance_and_target_tuned_lambda_are_not_an_inverse_surface() -> None:
    module = _module()
    with pytest.raises(ValueError, match="diagonal"):
        module.inverse_velocity(np.ones((3, 2)), np.zeros(2), np.ones((2, 2)), np.eye(2))
    with pytest.raises(ValueError, match="non-negative"):
        module.inverse_velocity(np.ones((3, 2)), np.zeros(2), np.ones((2, 2)), np.ones(2), ridge_lambda=-1.0)


def test_preflight_binds_m24_data_split_query_and_non_gate_scope() -> None:
    payload = json.loads(PREFLIGHT.read_text())
    assert payload["status"] == "CPU_PREFLIGHT_READY_NOT_A_GATE"
    assert payload["support_contract"]["trials"] == [0, 24]
    assert payload["query_contract"]["trials"] == [24, "end"]
    assert payload["inverse_contract"]["sigma"].startswith("support-residual diagonal")
    assert payload["formal_heldout_opened"] is False and payload["gpu_launched"] is False
    assert len(payload["sessions"]) == 15
    assert all(len(row["data_sha256"]) == 64 and len(row["split_manifest_sha256"]) == 64 and len(row["matched_mb4_outer_eval_sha256"]) == 64 for row in payload["sessions"])
