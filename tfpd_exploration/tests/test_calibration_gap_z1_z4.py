"""Tests for the Z1 honest-M oracle cells and the Z4 U-stability diagnostic.

Synthetic fixtures for all new math (principal angles, Ledoit-Wolf, effective
rank, honest-oracle activity restriction through the real B3S encoder,
C3-anchor comparison, ladder resolution).  The only real-artifact dependency
is the sealed receipt set loaded through the SHA-verified ledger (same
convention as test_calibration_gap_v1.py).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
STREAMING_ENCODERS = (
    REPO_ROOT / "streaming_calibration_exp/src/models/components/streaming_encoders.py"
)


def _load_pkg():
    spec = importlib.util.spec_from_file_location(
        "calibration_gap_v1_z", ROOT / "src/calibration_gap_v1/__init__.py",
        submodule_search_locations=[str(ROOT / "src/calibration_gap_v1")],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["calibration_gap_v1_z"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def _load_z1():
    spec = importlib.util.spec_from_file_location(
        "calibration_gap_z1_oracle_cells", ROOT / "src/calibration_gap_v1/z1_oracle_cells.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibration_gap_z1_oracle_cells"] = module
    spec.loader.exec_module(module)
    return module


def _load_z4():
    spec = importlib.util.spec_from_file_location(
        "calibration_gap_z4_subspace", ROOT / "src/calibration_gap_v1/z4_subspace.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibration_gap_z4_subspace"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def z1():
    return _load_z1()


@pytest.fixture(scope="module")
def z4():
    return _load_z4()


@pytest.fixture(scope="module")
def ledger():
    pkg = _load_pkg()
    return pkg.load_ledger()


# ---------------------------------------------------------------------------
# Z4: principal angles
# ---------------------------------------------------------------------------


def test_principal_angles_identical_and_orthogonal(z4):
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.normal(size=(12, 5)))[0]
    assert z4.principal_angles_deg(Q, Q).max() < 1e-4
    identity = np.eye(12)
    angles = z4.principal_angles_deg(identity[:, :5], identity[:, 5:10])
    assert angles.shape == (5,)
    assert np.allclose(angles, 90.0, atol=1e-6)


def test_principal_angles_known_rotation(z4):
    identity = np.eye(10)
    base = identity[:, :4]
    theta = np.radians(30.0)
    rotated = base.copy()
    rotated[:, 0] = np.cos(theta) * identity[:, 0] + np.sin(theta) * identity[:, 4]
    rotated = np.linalg.qr(rotated)[0]
    angles = z4.principal_angles_deg(base, rotated)
    assert angles.shape == (4,)
    assert angles[0] < 1e-6 and angles[1] < 1e-6 and angles[2] < 1e-6
    assert abs(angles[3] - 30.0) < 1e-4


def test_principal_angles_rectangular_bases(z4):
    rng = np.random.default_rng(1)
    Q = np.linalg.qr(rng.normal(size=(15, 6)))[0]
    angles = z4.principal_angles_deg(Q, Q[:, :2])
    assert angles.shape == (2,)
    assert angles.max() < 1e-4


def test_topk_subspace_recovers_generating_subspace(z4):
    rng = np.random.default_rng(2)
    V = np.linalg.qr(rng.normal(size=(40, 6)))[0]
    scale = np.diag([10.0, 6.0, 3.0, 1.5, 0.7, 0.3])
    X = (rng.normal(size=(6000, 6)) @ scale) @ V.T + 0.05 * rng.normal(size=(6000, 40))
    U, eigenvalues = z4.topk_subspace(z4.sample_covariance(X), 6)
    assert np.all(np.diff(eigenvalues) <= 1e-12)  # descending
    assert z4.principal_angles_deg(U, V).max() < 2.0


def test_ledoit_wolf_matches_sklearn_and_is_bounded(z4):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 18)) @ np.diag(np.linspace(0.2, 4.0, 18))
    Sigma, shrinkage = z4.ledoit_wolf_covariance(X)
    assert 0.0 <= shrinkage <= 1.0
    centered = X - X.mean(axis=0)
    S = centered.T @ centered / X.shape[0]
    mu = float(np.trace(S)) / X.shape[1]
    assert np.allclose(Sigma, (1 - shrinkage) * S + shrinkage * mu * np.eye(X.shape[1]))
    sklearn = pytest.importorskip("sklearn.covariance", reason="sklearn cross-check")
    _, expected = sklearn.ledoit_wolf(X)
    assert abs(shrinkage - float(expected)) < 1e-10


def test_ledoit_wolf_isotropic_data_shrinks_fully(z4):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 10)) * 2.0  # isotropic
    _, shrinkage = z4.ledoit_wolf_covariance(X)
    assert shrinkage > 0.9


def test_effective_rank_measures(z4):
    flat = z4.effective_rank_measures(np.ones(4))
    assert flat["participation_ratio"] == pytest.approx(4.0)
    assert flat["entropy_effective_rank"] == pytest.approx(4.0)
    decay = z4.effective_rank_measures(np.array([4.0, 2.0, 1.0, 0.5]))
    assert 1.0 < decay["participation_ratio"] < 4.0
    assert decay["entropy_effective_rank"] > decay["participation_ratio"]


# ---------------------------------------------------------------------------
# Z4: the per-session cell on a synthetic fixture
# ---------------------------------------------------------------------------


def _synthetic_calib(rng, *, n_trials=30, n_bins=100, n_units=30, latent=6, drift=0.05):
    V = np.linalg.qr(rng.normal(size=(n_units, latent)))[0]
    scale = np.diag(np.linspace(5.0, 0.5, latent))
    calib = np.empty((n_trials, n_bins, n_units))
    for trial in range(n_trials):
        latents = rng.normal(size=(n_bins, latent)) @ scale
        calib[trial] = latents @ V.T + drift * rng.normal(size=(n_bins, n_units))
    return calib, V


def test_subspace_cell_m10_beats_m4_and_bootstraps_populate(z4):
    rng = np.random.default_rng(5)
    calib, V = _synthetic_calib(rng)
    cell = z4.subspace_cell(calib, ks=(4, 6), ms=(4, 10), n_bootstrap=25)
    assert cell["n_units"] == 30 and cell["n_trials"] == 30
    assert cell["by_m"]["4"]["windows"] == 400
    assert cell["by_m"]["10"]["windows"] == 1000
    # effective rank of a latent-6 + small-noise stream is well below N
    assert cell["reference"]["effective_rank"]["participation_ratio"] < 20
    for m in ("4", "10"):
        entry = cell["by_m"][m]
        assert entry["effective_rank"]["entropy_effective_rank"] <= 30.0 + 1e-9
        for k in ("4", "6"):
            node = entry["by_k"][k]
            assert node["raw"]["max_angle_deg"] >= node["raw"]["min_angle_deg"]
            assert 0.0 <= node["bootstrap_max_angle_deg_percentiles"]["p2_5"]
            assert (
                node["bootstrap_max_angle_deg_percentiles"]["p2_5"]
                <= node["bootstrap_max_angle_deg_percentiles"]["p50"]
                <= node["bootstrap_max_angle_deg_percentiles"]["p97_5"]
            )
    # more activity -> (weakly) better aligned top subspaces on this fixture
    a4 = cell["by_m"]["4"]["by_k"]["6"]["raw"]["max_angle_deg"]
    a10 = cell["by_m"]["10"]["by_k"]["6"]["raw"]["max_angle_deg"]
    assert a10 < a4


def test_subspace_cell_reference_angles_to_generating_subspace(z4):
    rng = np.random.default_rng(6)
    calib, V = _synthetic_calib(rng)
    cell = z4.subspace_cell(calib, ks=(6,), ms=(10,), n_bootstrap=5)
    X_ref = calib.reshape(-1, 30)
    U_ref = z4.topk_subspace(z4.sample_covariance(X_ref), 6)[0]
    assert z4.principal_angles_deg(U_ref, V).max() < 25.0


def test_summarize_and_verdict_paths(z4):
    rng = np.random.default_rng(7)
    cells = {
        f"sess-{index}": z4.subspace_cell(
            _synthetic_calib(rng, n_units=20 + index)[0], ks=(6,), ms=(4, 10), n_bootstrap=5,
        )
        for index in range(3)
    }
    summary = z4.summarize_sessions(cells)
    assert summary["session_count"] == 3
    assert set(summary["by_m"]) == {"4", "10"}
    verdict = z4.verdict(summary)
    assert verdict["primary_scalar"].startswith("median over external-15 sessions")
    assert "reading" in verdict and isinstance(verdict["u_starves_at_m4"], bool)


def test_verdict_threshold_semantics(z4):
    def summary_with(m4: float, m10: float) -> dict:
        node = {
            "median_max_angle_deg": None,
            "median_mean_angle_deg": 0.0,
            "median_max_angle_deg_ledoit_wolf": 0.0,
            "median_bootstrap_p50_max_angle_deg": 0.0,
            "median_bootstrap_p97_5_max_angle_deg": 0.0,
        }
        base = {
            "session_count": 1,
            "reference": {},
            "by_m": {
                "4": {"by_k": {"6": {**node, "median_max_angle_deg": m4}}},
                "10": {"by_k": {"6": {**node, "median_max_angle_deg": m10}}},
            },
        }
        return base

    good = z4.verdict(summary_with(15.0, 12.0))
    assert "Path 1 proceed" in good["reading"] and good["u_starves_at_m4"] is False
    mixed = z4.verdict(summary_with(80.0, 12.0))
    assert "MIXED" in mixed["reading"] and mixed["u_starves_at_m4"] is True
    starved = z4.verdict(summary_with(85.0, 75.0))
    assert "starves" in starved["reading"] and starved["u_starves"] is True


# ---------------------------------------------------------------------------
# Z1: honest-oracle activity restriction through the REAL B3S encoder
# ---------------------------------------------------------------------------


def _load_streaming_encoders():
    spec = importlib.util.spec_from_file_location("z_test_streaming_encoders", STREAMING_ENCODERS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["z_test_streaming_encoders"] = module
    spec.loader.exec_module(module)
    return module


def test_b3s_activity_restriction_is_exact_trial_prefix_mean_pool():
    """first-M restriction = mean-pool over exactly the first M trials."""
    torch = pytest.importorskip("torch")
    encoders = _load_streaming_encoders()
    rng = np.random.default_rng(8)
    n_units, side_dim = 7, 4
    encoder = encoders.build_encoder(
        "B3S", window_size=50, trial_length=100, id_hidden_dim=8, hidden_dim=4,
        side_dim=side_dim,
    )
    encoder.eval()
    calib = torch.from_numpy(rng.normal(size=(30, 100, n_units)).astype(np.float32))
    side = torch.from_numpy(rng.normal(size=(1, n_units, side_dim)).astype(np.float32))
    with torch.no_grad():
        identity_m = {
            m: encoder.forward_batch(calib[:m].unsqueeze(0), side_features=side)
            for m in (4, 10, 30)
        }
        manual = {}
        for m in (4, 10, 30):
            feats = torch.stack([encoder.pre_pool(calib[i].unsqueeze(0).permute(0, 2, 1)) for i in range(m)])
            mean_feat = feats.mean(dim=0)
            pooled = torch.cat([mean_feat, side], dim=-1)
            manual[m] = encoder.post_pool(pooled)
    for m in (4, 10, 30):
        assert torch.allclose(identity_m[m], manual[m], atol=1e-5), f"M={m} prefix semantics"
    # the honest-oracle cells differ across budgets (activity is the new factor)
    assert not torch.allclose(identity_m[4], identity_m[30], atol=1e-4)
    assert not torch.allclose(identity_m[10], identity_m[30], atol=1e-4)
    # prefix property: identity(4) is a component of identity(10)'s mean pool
    with torch.no_grad():
        def _feat(i):
            return encoder.pre_pool(calib[i].unsqueeze(0).permute(0, 2, 1))
        feats4 = sum(_feat(i) for i in range(4))
        feats10 = feats4 + sum(_feat(i) for i in range(4, 10))
        assert torch.allclose(
            encoder.post_pool(torch.cat([feats10 / 10.0, side], dim=-1)),
            identity_m[10], atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Z1: anchors and ladder
# ---------------------------------------------------------------------------


def test_cell_matrix_is_the_six_executed_honest_oracle_cells(z1):
    """3 budgets x 2 surfaces x the single C3 support = 6 executed cells."""
    cells = z1.cell_matrix()
    assert len(cells) == 6
    assert {(c["budget"], c["surface"]) for c in cells} == {
        (b, s) for b in (4, 10, 30) for s in ("external", "within")
    }
    assert all(c["support"] == "C3_full_session_oracle" for c in cells)
    assert all(c["activity"] == f"b3s_first_{c['budget']}_trials" for c in cells)
    assert all(c["leakage_diagnostic"] is True for c in cells)


def test_anchor_table_and_check_accepts_near_exact(z1):
    ledger = _FakeLedger(
        {
            "results/low_cost_calibration_diagnostics_v3/receipt.json": {
                "cells": [
                    {"surface": "external", "budget": b, "support": "C3_full_session_oracle",
                     "sessions": [
                         {"session": "s1", "n_windows": 10, "variance_weighted_r2": 0.5,
                          "prediction_sha256": "a", "normalized_side_sha256": "side1"},
                     ]}
                    for b in (4, 30)
                ]
            }
        }
    )
    anchors = z1.anchor_table(ledger)
    assert set(anchors) == {("external", "s1")}
    rows = [
        {"surface": "external", "session": "s1", "budget": 30, "n_windows": 10,
         "variance_weighted_r2": 0.5 + 1e-7, "normalized_side_sha256": "side1"},
        {"surface": "external", "session": "s1", "budget": 4, "variance_weighted_r2": 0.4,
         "normalized_side_sha256": "side1"},
    ]
    report = z1.check_anchor(rows, ledger)
    assert report["sessions_checked"] == 1
    assert report["max_abs_delta_r2"] <= 1e-6


def test_check_anchor_fails_on_drift_and_carrier_mismatch(z1):
    ledger = _FakeLedger(
        {
            "results/low_cost_calibration_diagnostics_v3/receipt.json": {
                "cells": [
                    {"surface": "external", "budget": b, "support": "C3_full_session_oracle",
                     "sessions": [
                         {"session": "s1", "n_windows": 10, "variance_weighted_r2": 0.5,
                          "prediction_sha256": "a", "normalized_side_sha256": "side1"},
                     ]}
                    for b in (4, 30)
                ]
            }
        }
    )
    drifted = [
        {"surface": "external", "session": "s1", "budget": 30, "n_windows": 10,
         "variance_weighted_r2": 0.42, "normalized_side_sha256": "side1"},
    ]
    with pytest.raises(z1.Z1Error):
        z1.check_anchor(drifted, ledger)
    wrong_carrier = [
        {"surface": "external", "session": "s1", "budget": 30, "n_windows": 10,
         "variance_weighted_r2": 0.5, "normalized_side_sha256": "other"},
    ]
    with pytest.raises(z1.Z1Error):
        z1.check_anchor(wrong_carrier, ledger)


def test_anchor_table_rejects_budget_disagreement(z1):
    ledger = _FakeLedger(
        {
            "results/low_cost_calibration_diagnostics_v3/receipt.json": {
                "cells": [
                    {"surface": "external", "budget": 4, "support": "C3_full_session_oracle",
                     "sessions": [{"session": "s1", "n_windows": 10, "variance_weighted_r2": 0.5,
                                   "prediction_sha256": "a", "normalized_side_sha256": "side1"}]},
                    {"surface": "external", "budget": 30, "support": "C3_full_session_oracle",
                     "sessions": [{"session": "s1", "n_windows": 10, "variance_weighted_r2": 0.6,
                                   "prediction_sha256": "b", "normalized_side_sha256": "side1"}]},
                ]
            }
        }
    )
    with pytest.raises(z1.Z1Error):
        z1.anchor_table(ledger)


def test_budget_ladder_resolves_every_rung_from_receipts(z1, ledger):
    for surface in ("external", "within"):
        for budget in (4, 10, 30):
            ladder = z1.budget_ladder(ledger, surface, budget, honest_oracle_r2=0.40)
            for key in (
                "c0_chronological_ols", "best_label_limited_dopt_ridge",
                "honest_total_calibration_dopt_ridge", "oracle_full_session_labels_m30_activity",
                "carrier_term_m30_activity", "activity_term", "total_gap",
            ):
                assert isinstance(ladder[key], float), (surface, budget, key)
            assert ladder["honest_m_budget_oracle"] == 0.40
            assert ladder["activity_cost_of_oracle_at_m_budget"] == pytest.approx(
                ladder["oracle_full_session_labels_m30_activity"] - 0.40
            )
            assert ladder["carrier_term_at_m_budget_activity"] == pytest.approx(
                0.40 - ladder["honest_total_calibration_dopt_ridge"]
            )


def test_readings_verdict_thresholds(z1):
    def payload_with(m4: float, oracle: float) -> dict:
        cells = [
            {"surface": "external", "budget": b, "summary": {"equal_session_mean_r2": m4 if b == 4 else 0.0}}
            for b in (4, 10, 30)
        ]
        return {
            "cells": cells,
            "ladders": {"external_M4": {"oracle_full_session_labels_m30_activity": oracle}},
        }

    real = z1.readings(payload_with(0.4449, 0.4449))
    assert "P1 justified" in real["pre_registered_reading"]
    collapsed = z1.readings(payload_with(0.25, 0.4449))
    assert "P4 becomes the main line" in collapsed["pre_registered_reading"]
    middle = z1.readings(payload_with(0.38, 0.4449))
    assert "intermediate" in middle["pre_registered_reading"]


class _FakeLedger:
    def __init__(self, bodies):
        self.bodies = bodies


# ---------------------------------------------------------------------------
# module hygiene
# ---------------------------------------------------------------------------


def test_z1_module_documents_the_two_anchors_and_boundaries(z1):
    text = Path(z1.__file__).read_text()
    for needle in (
        "ANCHOR_TOLERANCE", "normalized_side_sha256", "leakage",
        "b3s_calibration_activity_restricted_to_first_M_trials",
    ):
        assert needle in text


def test_z4_module_declares_primary_scalar_before_unblinding(z4):
    text = Path(z4.__file__).read_text()
    for needle in (
        "THRESHOLD_M10_DEG", "THRESHOLD_M4_DEG", "primary_scalar",
        "LARGEST", "BOOTSTRAP_SEED",
    ):
        assert needle in text
