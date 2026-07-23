"""Tests for baseline prerequisite validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.metrics.baseline import BaselineValidationError, validate_baseline_prerequisites
from src.metrics.run_artifacts import METRICS_SUMMARY_FIELDS, write_metrics_table
from src.models.falcon_module import DATASET_NAMES


def _write_minimal_baseline(path: Path) -> None:
    rows = []
    for session in DATASET_NAMES["m2"]["heldin"]:
        rows.append(
            {
                "split": "test_heldin",
                "session": session,
                "M": 33,
                "R2_variance_weighted": "0.60",
            }
        )
    for session in DATASET_NAMES["m2"]["heldout"]:
        rows.append(
            {
                "split": "test_heldout",
                "session": session,
                "M": 33,
                "R2_variance_weighted": "0.20",
            }
        )
    write_metrics_table(path.parent, rows, path.name, METRICS_SUMMARY_FIELDS)


def test_validate_baseline_prerequisites_fails_when_missing(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "baseline_metrics_path": str(tmp_path / "missing.csv"),
            "model": {"teacher_ckpt_path": str(tmp_path / "teacher.ckpt")},
            "data": {"calibration_n_trials": 33},
        }
    )
    with pytest.raises(BaselineValidationError):
        validate_baseline_prerequisites(cfg)


def test_validate_baseline_prerequisites_accepts_complete_baseline(tmp_path: Path):
    baseline = tmp_path / "metrics_per_session.csv"
    _write_minimal_baseline(baseline)
    teacher = tmp_path / "teacher.ckpt"
    teacher.write_bytes(b"teacher")
    cfg = OmegaConf.create(
        {
            "baseline_metrics_path": str(baseline),
            "model": {"teacher_ckpt_path": str(teacher)},
            "data": {"calibration_n_trials": 33},
        }
    )
    result = validate_baseline_prerequisites(cfg)
    assert len(result["heldin_sessions"]) == 7
    assert len(result["heldout_sessions"]) == 6
