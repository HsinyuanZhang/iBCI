"""Tests for RT classical comparators."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.comparators.core import rt_classical_comparators as core
from sua_exploration.mc_maze.subm_v9_f0_pv_ridge import fit_ridge, predict_ridge


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_ridge_matches_hand_closed_form() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(12, 8))
    targets = rng.normal(size=(12, 2))
    readout = fit_ridge(features, targets, normalized_lambda=1.0, device="cpu")
    prediction = predict_ridge(features, readout, device="cpu")
    mean, scale, coefficients = core.hand_solve_normalized_ridge(features, targets, normalized_lambda=1.0)
    manual = ((features - mean) / scale) @ coefficients + targets.mean(axis=0)
    assert np.allclose(prediction, manual, atol=1.0e-6)


def test_lambda_selection_never_reads_query_data() -> None:
    rng = np.random.default_rng(1)
    support_features = rng.normal(size=(30, 6))
    support_targets = rng.normal(size=(30, 2))
    selected, detail = core.select_lambda_on_support_only(support_features, support_targets)
    assert detail["query_rows_used"] == 0
    assert selected in core.LAMBDA_CV_GRID
    # Selection API accepts only support arrays; query tensors are structurally absent.
    assert "query" not in str(core.select_lambda_on_support_only.__code__.co_varnames)


def test_direction_degeneracy_detector() -> None:
  degenerate = np.asarray([0.7853981633974483, 0.7853981633974483, 0.7853981633974483])
  nondegenerate = np.asarray([0.0, 0.7853981633974483, 1.5707963267948966])
  assert int(np.unique(np.round(degenerate, decimals=12)).size) == 1
  assert int(np.unique(np.round(nondegenerate, decimals=12)).size) == 3
  verdict_deg = core.direction_degeneracy_verdict(
      [
          core.NativeDirectionAudit(
              session_name="s1",
              trial_count=3,
              finite_target_dir_count=3,
              unique_target_dir_count=1,
              unique_target_dir_values_rad=(0.7853981633974483,),
              m24_finite_target_dir_count=3,
              m24_unique_target_dir_count=1,
              pv_definable_on_native_field=False,
          )
      ]
  )
  assert verdict_deg["verdict"] == "PV_INAPPLICABLE_NATIVE_FIELD_DEGENERATE"
  verdict_non = core.direction_degeneracy_verdict(
      [
          core.NativeDirectionAudit(
              session_name="s2",
              trial_count=3,
              finite_target_dir_count=3,
              unique_target_dir_count=2,
              unique_target_dir_values_rad=(0.0, 0.7853981633974483),
              m24_finite_target_dir_count=3,
              m24_unique_target_dir_count=2,
              pv_definable_on_native_field=True,
          )
      ]
  )
  assert verdict_non["verdict"] == "PV_PARTIALLY_DEFINABLE_ON_NATIVE_FIELD"


def test_paired_contrast_sign_counts() -> None:
    contrast = core.paired_contrast([0.2, 0.5, 0.1], [0.1, 0.6, 0.2])
    assert contrast["positive"] == 1
    assert contrast["zero"] == 0
    assert contrast["negative"] == 2
    assert contrast["per_fold_delta"] == pytest.approx([0.1, -0.1, -0.1])


def test_rt_outer_layout_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    neural = rng.random((5000, 4), dtype=np.float32)
    cov = rng.random((5000, 2), dtype=np.float32)
    trial_change = np.zeros(5000, dtype=bool)
    trial_change[::100] = True
    eval_mask = np.ones(5000, dtype=bool)
    layout_a = core.rt_outer_window_layout("synthetic", neural, cov, trial_change, eval_mask)
    layout_b = core.rt_outer_window_layout("synthetic", neural, cov, trial_change, eval_mask)
    assert np.array_equal(layout_a.query_window_starts, layout_b.query_window_starts)
    assert layout_a.query_window_audit["ordered_query_identity_sha256"] == layout_b.query_window_audit[
        "ordered_query_identity_sha256"
    ]


@pytest.mark.slow
def test_sealed_query_identity_binds_on_fold0() -> None:
    data_dir = REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"
    stage2_cell_root = (
        REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical/matrix_v1/cells"
    )
    if not data_dir.is_dir() or not stage2_cell_root.is_dir():
        pytest.skip("RT data or sealed Stage-2 cells unavailable")
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions, load_rt_session

    sessions = {core.session_name_from_nwb_path(Path(path)): path for path in find_rt_sessions(data_dir)}
    sealed = core.load_sealed_stage2_cell(stage2_cell_root, 0)
    fold = core.evaluate_fold(
        fold=0,
        nwb_path=sessions[sealed["session_name"]],
        stage2_cell_root=stage2_cell_root,
        raw_loader=load_rt_session,
    )
    assert fold.query_identity_bound is True


def test_aggregate_sign_counts_on_fifteen_hand_vectors() -> None:
    folds = []
    for index in range(core.EXPECTED_FOLDS):
        session = f"s{index}"
        ridge_r2 = 0.10
        t4d_r2 = 0.20
        ridge = core.RidgeArmResult(
            arm="ridge_w50_lambda1",
            normalized_lambda=1.0,
            r2=ridge_r2,
            calibration_rows=100,
            query_rows=50,
            feature_dim=200,
            parameters_fitted=400,
            observations_per_parameter=0.25,
            supervision_coordinates_consumed=200,
        )
        folds.append(
            core.FoldEvaluation(
                fold=index,
                session_name=session,
                t4d_reference_r2=t4d_r2,
                sealed_query_identity={"ordered_query_identity_sha256": "0" * 64},
                query_identity_bound=True,
                ridge_fixed_lambda=ridge,
                ridge_cv_lambda=ridge,
                native_direction_audit=core.NativeDirectionAudit(
                    session_name=session,
                    trial_count=30,
                    finite_target_dir_count=30,
                    unique_target_dir_count=1,
                    unique_target_dir_values_rad=(0.0,),
                    m24_finite_target_dir_count=24,
                    m24_unique_target_dir_count=1,
                    pv_definable_on_native_field=False,
                ),
                endpoint_pv_diagnostic=None,
            )
        )
    aggregate = core.aggregate_fold_results(folds)
    assert aggregate["arms"]["ridge_w50_lambda1"]["paired_vs_t4d"]["negative"] == core.EXPECTED_FOLDS
    assert aggregate["native_population_vector"]["verdict"] == "PV_INAPPLICABLE_NATIVE_FIELD_DEGENERATE"
