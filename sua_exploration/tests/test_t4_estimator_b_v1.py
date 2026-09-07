"""No-NWB contracts for the source-only Experiment B estimator selection machinery."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.t4_estimator_b_v1 import (  # noqa: E402
    CANDIDATE_NAMES, N_CALIBRATION_TRIALS, SCORE_TRIAL_START, SCORE_TRIAL_STOP,
    SessionCounts, _ordinary_fit, descriptor_rate_hz, estimate_eb_prior, fit_eb_ridge,
    fit_poisson_irls, fit_second_harmonic, paired_cosine_reliability, prospective_deviance,
    select_unique_winner, summarize_candidate,
)
from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD  # noqa: E402


def _session(name: str, *, second_harmonic: bool = False) -> SessionCounts:
    directions = np.asarray([index % 8 for index in range(50)], dtype=np.int64)
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions])
    durations = np.ones(50, dtype=np.float64)
    rates = np.vstack([
        10.0 + 2.0 * np.cos(theta) - 1.0 * np.sin(theta) + (1.5 * np.cos(2.0 * theta) if second_harmonic else 0.0),
        8.0 - 1.2 * np.cos(theta) + 2.0 * np.sin(theta) + (0.7 * np.sin(2.0 * theta) if second_harmonic else 0.0),
        6.0 + 0.8 * np.cos(theta) + 0.5 * np.sin(theta),
    ])
    # Integer counts make the thinning contract meaningful while retaining nonzero directional signal.
    return SessionCounts(name=name, counts=np.maximum(np.rint(rates), 0).astype(np.int64), durations_s=durations, direction_indices=directions)


def test_candidate_contracts_export_four_finite_coordinates_and_score_only_exported_1h():
    session = _session("s0", second_harmonic=True)
    ordinary = _ordinary_fit(session)
    second = fit_second_harmonic(session)
    poisson = fit_poisson_irls(session)
    prior = estimate_eb_prior([session, _session("s1")])
    eb = fit_eb_ridge(session, prior, shrinkage_lambda=1.0)
    for fit in (ordinary, second, poisson, eb):
        assert fit.t4.shape == (3, 4)
        assert np.isfinite(fit.t4).all()
        assert prospective_deviance(session, fit) is not None
    assert second.metadata["scoring_uses"] == "derived_first_harmonic_[a,c,m,b]_only"
    # The public rate function only accepts the exported four-vector; no a2/c2 path exists.
    rates = descriptor_rate_hz(second.t4, session.direction_indices[SCORE_TRIAL_START:SCORE_TRIAL_STOP])
    assert rates.shape == (3, 20)
    assert np.isfinite(rates).all() and np.all(rates > 0.0)
    assert poisson.metadata["max_iterations"] == 32


def test_reliability_uses_exact_eight_disjoint_thinning_seeds_and_common_valid_units():
    session = _session("s0")
    reliability = paired_cosine_reliability(session, lambda counts: fit_second_harmonic(session, counts=counts))
    assert reliability["seeds"] == [1201, 1207, 1213, 1217, 1223, 1229, 1231, 1237]
    assert len(reliability["records"]) == 8
    assert reliability["defined_seed_count"] == 8
    assert reliability["median_delta"] is not None
    assert all(row["common_valid_unit_count"] == 3 for row in reliability["records"])


def test_rank_deficiency_and_short_trial_receipts_fail_closed():
    with pytest.raises(ValueError, match=r"\[units,50\]"):
        SessionCounts("bad", np.zeros((2, 49), dtype=np.int64), np.ones(50), np.zeros(50, dtype=np.int64))
    directions = np.zeros(50, dtype=np.int64)
    session = SessionCounts("rank_bad", np.ones((2, 50), dtype=np.int64), np.ones(50), directions)
    fit = fit_second_harmonic(session)
    assert fit.rank_deficient_units == 2
    assert np.isnan(fit.t4).all()


def _passing_row() -> dict:
    return {
        "prospective_deviance_ratio": 0.97,
        "reliability": {"median_delta": 0.03},
        "rank_increase": False, "nonconvergence_increase": False, "invalid_increase": False,
        "cost": {"calibration_ops_proxy": 5, "persistent_state_bytes": 96},
    }


def test_frozen_gate_and_unique_winner_tie_break_semantics():
    summary = summarize_candidate([_passing_row() for _ in range(27)])
    assert summary["decision"] == "pass"
    results = {
        "eb_ridge": {"summary": summary, "outer_folds": [_passing_row() for _ in range(27)]},
        "second_harmonic": {"summary": {**summary, "mean_prospective_deviance_ratio": 0.96}, "outer_folds": [_passing_row() for _ in range(27)]},
        "poisson_irls": {"summary": {**summary, "decision": "fail"}, "outer_folds": [_passing_row() for _ in range(27)]},
    }
    winner = select_unique_winner(results)
    assert winner["winner"] == "second_harmonic"
    assert winner["decision"] == "unique_cpu_winner_requires_new_gpu_prelaunch"
    failed = summarize_candidate([_passing_row() for _ in range(19)] + [{**_passing_row(), "reliability": {"median_delta": -0.01}} for _ in range(8)])
    assert failed["decision"] == "fail"


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_t4_estimator_b_v1.py"
    spec = importlib.util.spec_from_file_location("t4_estimator_b_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_receipt_seals_development_and_formal_without_opening_data(tmp_path):
    module = _runner_module()
    # Use the actual immutable source-audit binding but a synthetic manifest; no NWB is present.
    manifest = {"schema_version": 1, "session_splits": {
        "train": [f"train_{i}" for i in range(27)], "val": [f"val_{i}" for i in range(6)], "test": [f"test_{i}" for i in range(6)]}}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    receipt = module.build_dry_run_receipt(path)
    assert receipt["no_dataset_opened"] is True
    assert receipt["no_development_session_opened"] is True
    assert receipt["no_formal_session_opened"] is True
    assert receipt["chronology"]["support"] == [0, N_CALIBRATION_TRIALS]
    assert receipt["chronology"]["score"] == [30, 50]
    assert receipt["selection_gate"]["irreversible_fail_fast"] == "stop candidate when joint-failure folds > 7"
    assert set(receipt["candidates"]) == set(CANDIDATE_NAMES)


def test_runner_can_resolve_only_source_files_never_sealed_names(tmp_path):
    module = _runner_module()
    base = tmp_path / "sub-C"; base.mkdir()
    names = [f"source_{i}" for i in range(27)]
    for name in names:
        (base / f"{name}_behavior+ecephys.nwb").touch()
    paths = module.resolve_source_paths({"source_session_names": names, "sealed_development_names": ["dev"], "sealed_formal_names": ["formal"]}, tmp_path)
    assert len(paths) == 27
    assert not (base / "dev_behavior+ecephys.nwb").exists()
    assert not (base / "formal_behavior+ecephys.nwb").exists()
