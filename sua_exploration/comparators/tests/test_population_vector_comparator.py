"""Unit tests for the population-vector comparator skeleton."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.comparators.core import population_vector_comparator as core
from sua_exploration.mc_maze.subm_v9_f0_pv_ridge import (
    fit_population_vector_gain,
    population_vectors,
    predict_population_vector,
    preferred_directions_from_cosine,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _runner_module():
    path = REPO_ROOT / "sua_exploration/scripts/run_population_vector_comparator.py"
    spec = importlib.util.spec_from_file_location("run_population_vector_comparator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_degeneracy_detector_on_synthetic_counts() -> None:
    degenerate = core.applicability_verdict(
        core.M2_DATASET,
        [
            core.SessionDirectionAudit(
                session_name="deg",
                native_field_exists=True,
                native_field_path=core.M2_DIRECTION_FIELD_PATH,
                derived_field_path=core.M2_DERIVED_ANGLE_PATH,
                trial_count=3,
                finite_direction_count=3,
                unique_direction_count=1,
                unique_direction_values=(0.7853981633974483,),
                direction_value_distribution={"0.785398": 3},
                calibration_trial_count=3,
                calibration_finite_direction_count=3,
                calibration_unique_direction_count=1,
                calibration_unique_direction_values=(0.7853981633974483,),
                calibration_direction_distribution={"0.785398": 3},
                pv_definable_on_native_field=False,
            )
        ],
    )
    assert degenerate["verdict"] == "PV_INAPPLICABLE_NATIVE_FIELD_DEGENERATE"

    spread = core.applicability_verdict(
        core.M2_DATASET,
        [
            core.SessionDirectionAudit(
                session_name="ok",
                native_field_exists=True,
                native_field_path=core.M2_DIRECTION_FIELD_PATH,
                derived_field_path=core.M2_DERIVED_ANGLE_PATH,
                trial_count=3,
                finite_direction_count=3,
                unique_direction_count=3,
                unique_direction_values=(0.0, 0.7853981633974483, 1.5707963267948966),
                direction_value_distribution={"0.0": 1, "0.785398": 1, "1.570796": 1},
                calibration_trial_count=3,
                calibration_finite_direction_count=3,
                calibration_unique_direction_count=3,
                calibration_unique_direction_values=(0.0, 0.7853981633974483, 1.5707963267948966),
                calibration_direction_distribution={"0.0": 1, "0.785398": 1, "1.570796": 1},
                pv_definable_on_native_field=True,
            )
        ],
    )
    assert spread["verdict"] == "PV_DEFINABLE"


def test_cosine_tuning_matches_hand_solution() -> None:
    thetas = np.asarray([0.0, np.pi / 2.0, np.pi], dtype=np.float64)
    rates = np.asarray([3.0, 1.0, -1.0], dtype=np.float64)
    a, c, m, b = core.fit_cosine_tuning_hand(thetas, rates)
    design = np.stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)], axis=1)
    expected_b, expected_a, expected_c = np.linalg.lstsq(design, rates, rcond=None)[0]
    assert np.allclose([a, c, b], [expected_a, expected_c, expected_b], atol=1.0e-12)
    assert np.isclose(m, math.hypot(a, c))


def test_sealed_estimator_reuse_on_hand_checkable_case() -> None:
    thetas = np.asarray([0.0, np.pi / 2.0, np.pi], dtype=np.float64)
    channel_rates = np.asarray([3.0, 1.0, -1.0], dtype=np.float64)
    a, c, m, b = core.fit_cosine_tuning_hand(thetas, channel_rates)
    preferred, _zero = preferred_directions_from_cosine(np.asarray([a]), np.asarray([c]), np.asarray([m]))
    calib_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, -1.0]], dtype=np.float64)
    calib_targets = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    gain, intercept, rank = fit_population_vector_gain(calib_vectors, calib_targets)
    query_rates = np.asarray([[2.0], [0.0]], dtype=np.float64)
    query_vectors = population_vectors(query_rates, preferred, np.asarray([b]))
    prediction = predict_population_vector(query_vectors, gain, intercept)
    manual_vectors = (query_rates - b) @ preferred
    manual_prediction = manual_vectors @ gain + intercept
    assert rank == 3
    assert np.allclose(prediction, manual_prediction, atol=1.0e-6)


def test_h1_refuses_score_when_audit_inapplicable() -> None:
    applicability = core.applicability_verdict(
        core.H1_DATASET,
        [
            core.SessionDirectionAudit(
                session_name="h1",
                native_field_exists=False,
                native_field_path=None,
                derived_field_path=None,
                trial_count=0,
                finite_direction_count=0,
                unique_direction_count=0,
                unique_direction_values=(),
                direction_value_distribution={},
                calibration_trial_count=0,
                calibration_finite_direction_count=0,
                calibration_unique_direction_count=0,
                calibration_unique_direction_values=(),
                calibration_direction_distribution={},
                pv_definable_on_native_field=False,
                projection_assessment="synthetic inapplicable",
            )
        ],
    )
    with pytest.raises(core.PopulationVectorComparatorError, match="H1 population-vector scoring refused"):
        core.refuse_h1_scoring_if_inapplicable(applicability)


def test_runner_refuses_to_overwrite_existing_receipt(tmp_path: Path) -> None:
    runner = _runner_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    receipt_path = runner.receipt_path_for(core.H1_DATASET, output_dir, audit_only=True)
    receipt_path.write_text("{}", encoding="utf-8")

    fake_payload = {
        "dataset": core.H1_DATASET,
        "verdict": "PV_INAPPLICABLE_NO_NATIVE_DISCRETE_DIRECTION_FIELD",
        "sessions_with_nondegenerate_native_field": 0,
        "sessions_total": 1,
        "input_paths": {},
    }
    original_audit = runner.core.audit_dataset_applicability
    runner.core.audit_dataset_applicability = lambda dataset, path: dict(fake_payload)
    try:
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            runner.build_receipt(
                dataset=core.H1_DATASET,
                data_dir=data_dir,
                output_dir=output_dir,
                audit_only=True,
                reference_manifest_path=None,
            )
    finally:
        runner.core.audit_dataset_applicability = original_audit


def test_audit_receipt_is_deterministic_for_synthetic_payload(tmp_path: Path) -> None:
    runner = _runner_module()
    payload = {
        "dataset": core.H1_DATASET,
        "verdict": "PV_INAPPLICABLE_NO_NATIVE_DISCRETE_DIRECTION_FIELD",
        "sessions_with_nondegenerate_native_field": 0,
        "sessions_total": 1,
        "per_session": {},
        "input_paths": {},
    }

    original_audit = runner.core.audit_dataset_applicability
    runner.core.audit_dataset_applicability = lambda dataset, path: dict(payload)
    try:
        shared_data = tmp_path / "data_shared"
        shared_data.mkdir()
        first = runner.build_receipt(
            dataset=core.H1_DATASET,
            data_dir=shared_data,
            output_dir=tmp_path / "out_a",
            audit_only=True,
            reference_manifest_path=None,
        )
        second = runner.build_receipt(
            dataset=core.H1_DATASET,
            data_dir=shared_data,
            output_dir=tmp_path / "out_b",
            audit_only=True,
            reference_manifest_path=None,
        )
    finally:
        runner.core.audit_dataset_applicability = original_audit
    assert first["receipt_sha256"] == second["receipt_sha256"]
    first_receipt = json.loads(Path(first["receipt_path"]).read_text(encoding="utf-8"))
    second_receipt = json.loads(Path(second["receipt_path"]).read_text(encoding="utf-8"))
    first_receipt.pop("date", None)
    second_receipt.pop("date", None)
    assert first_receipt == second_receipt
