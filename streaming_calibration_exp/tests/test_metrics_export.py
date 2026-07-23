"""Tests for metrics export helpers."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.metrics.run_artifacts import (
    METRICS_SUMMARY_FIELDS,
    build_test_metric_rows,
    load_baseline_session_r2,
    parse_test_metrics,
    write_metrics_table,
)
from src.models.components.streaming_encoders import EncoderCostProfile


def _profile() -> EncoderCostProfile:
    return EncoderCostProfile(
        parameter_count=1000,
        weight_bytes=4000,
        trial_buffer_bytes=100,
        support_state_bytes=200,
        peak_live_state_bytes=300,
        mac_per_trial=10,
        mac_per_session=330,
        variant="B3",
    )


def test_parse_test_metrics_extracts_session_and_aggregate_values():
    metric_dict = {
        "test_heldin/r2_mean": torch.tensor(0.63),
        "test_heldout/r2_mean": torch.tensor(0.21),
        "test_heldin/identity_mse": torch.tensor(0.006),
        "test_heldout/identity_mse": torch.tensor(0.008),
        "test_heldin/prediction_distill_mse": torch.tensor(1e-7),
        "test_heldout/prediction_distill_mse": torch.tensor(2e-7),
        "test_heldin_ses-2020-10-19-Run1/r2": torch.tensor(0.65),
        "test_heldout_ses-2020-11-19-Run1/r2": torch.tensor(0.07),
        "test_heldin_ses-2020-10-19-Run1/identity_mse": torch.tensor(0.005),
        "test_heldout_ses-2020-11-19-Run1/identity_mse": torch.tensor(0.009),
    }
    parsed = parse_test_metrics(metric_dict)
    assert parsed.heldin_mean_r2 == pytest.approx(0.63)
    assert parsed.heldout_mean_r2 == pytest.approx(0.21)
    assert parsed.heldin_session_r2["ses-2020-10-19-Run1"] == pytest.approx(0.65)
    assert parsed.heldout_session_r2["ses-2020-11-19-Run1"] == pytest.approx(0.07)


def test_build_test_metric_rows_writes_delta_against_baseline(tmp_path: Path):
    baseline_csv = tmp_path / "baseline.csv"
    write_metrics_table(
        tmp_path,
        [
            {
                "split": "test_heldout",
                "session": "ses-2020-11-19-Run1",
                "R2_variance_weighted": "0.17303000",
            }
        ],
        "baseline.csv",
        METRICS_SUMMARY_FIELDS,
    )
    baseline = load_baseline_session_r2(baseline_csv)
    parsed = parse_test_metrics(
        {
            "test_heldout/r2_mean": 0.069,
            "test_heldout_ses-2020-11-19-Run1/r2": 0.069,
        }
    )
    summary_rows, per_session_rows = build_test_metric_rows(
        run_id="run",
        variant="B3",
        seed=42,
        calibration_trials=33,
        parsed=parsed,
        profile=_profile(),
        baseline=baseline,
        validation_protocol="loso",
        fold_id=0,
    )
    heldout_row = next(row for row in per_session_rows if row["session"] == "ses-2020-11-19-Run1")
    assert heldout_row["R2_delta_vs_matched_baseline"] == pytest.approx("-0.10403000")
    aggregate = next(row for row in summary_rows if row["split"] == "test_heldout")
    assert aggregate["R2_delta_vs_matched_baseline"] == pytest.approx("-0.10403000")
