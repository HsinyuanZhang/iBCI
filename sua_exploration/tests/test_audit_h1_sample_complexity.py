"""Focused tests for the H1 sample-complexity accounting audit."""
from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_h1_sample_complexity.py"
DATA_ROOT = ROOT / "SPINT-main/data/000954"


def _module():
    spec = importlib.util.spec_from_file_location("audit_h1_sample_complexity", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arithmetic_helpers_on_synthetic_inputs() -> None:
    mod = _module()
    support_events = 11
    eval_bins = 8800
    endpoint_coords = mod.acquisition_endpoint_coordinates(support_events)
    dense_coords = mod.dense_velocity_coordinates(eval_bins)
    assert endpoint_coords == 154
    assert mod.derived_displacement_coordinates(support_events) == 77
    assert mod.projected_estimator_inputs(support_events) == 44
    assert dense_coords == 61600
    assert mod.coordinate_ratio(dense_coords=dense_coords, endpoint_coords=endpoint_coords) == pytest.approx(
        61600 / 154
    )
    assert mod.conservative_ratio_100ms(eval_bins=eval_bins, endpoint_coords=endpoint_coords) == pytest.approx(
        (eval_bins / 5.0 * 7) / 154
    )
    assert mod.carrier_observations_per_parameter(support_events) == pytest.approx(11 / 5)
    assert mod.ridge_observations_per_parameter(eval_bins) == pytest.approx(8800 / (50 * 176))
    carrier_opp = mod.carrier_observations_per_parameter(support_events)
    ridge_opp = mod.ridge_observations_per_parameter(eval_bins)
    assert mod.posedness_ratio(carrier_opp=carrier_opp, ridge_opp=ridge_opp) == pytest.approx(
        carrier_opp / ridge_opp
    )


def test_pooled_from_sums_and_mean_of_session_ratios_differ() -> None:
    mod = _module()
    rows = [
        {
            "support_events": 10,
            "eval_bins": 1000,
            "acquisition_endpoint_coordinates": 140,
            "dense_velocity_coordinates": 7000,
            "coordinate_ratio": 7000 / 140,
            "conservative_ratio_100ms": (1000 / 5.0 * 7) / 140,
            "posedness_ratio": (10 / 5) / (1000 / (50 * 176)),
        },
        {
            "support_events": 20,
            "eval_bins": 2000,
            "acquisition_endpoint_coordinates": 280,
            "dense_velocity_coordinates": 5600,
            "coordinate_ratio": 5600 / 280,
            "conservative_ratio_100ms": (2000 / 5.0 * 7) / 280,
            "posedness_ratio": (20 / 5) / (2000 / (50 * 176)),
        },
    ]
    pooled = mod.pooled_from_sums(
        support_events=30,
        eval_bins=3000,
        endpoint_coords=420,
        dense_coords=12600,
    )
    mean_ratios = mod.mean_of_per_session_ratios(rows)
    assert pooled["coordinate_ratio_from_sums"] == pytest.approx(12600 / 420)
    assert mean_ratios["coordinate_ratio_mean_of_sessions"] == pytest.approx(
        ((7000 / 140) + (5600 / 280)) / 2
    )
    assert pooled["coordinate_ratio_from_sums"] != pytest.approx(mean_ratios["coordinate_ratio_mean_of_sessions"])


def test_conservative_ratio_is_one_fifth_of_coordinate_ratio() -> None:
    mod = _module()
    eval_bins = 12345
    endpoint_coords = 574
    raw = mod.coordinate_ratio(
        dense_coords=mod.dense_velocity_coordinates(eval_bins),
        endpoint_coords=endpoint_coords,
    )
    conservative = mod.conservative_ratio_100ms(eval_bins=eval_bins, endpoint_coords=endpoint_coords)
    assert conservative == pytest.approx(raw / 5.0)


def test_audit_refuses_to_overwrite_existing_receipt(tmp_path: Path) -> None:
    mod = _module()
    output = tmp_path / "audit.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        mod.write_immutable(output, {"schema": mod.SCHEMA})


def test_script_source_never_references_open_loop_velocity_series() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "OpenLoopKinematicsVelocity" not in source


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_end_to_end_audit_writes_immutable_receipt(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-root",
            str(DATA_ROOT),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "coord_ratio" in completed.stdout
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["schema"] == "h1_sample_complexity_audit_v1"
    assert body["scope"]["cuda_used"] is False
    assert body["scope"]["dense_velocity_series_opened"] is False
