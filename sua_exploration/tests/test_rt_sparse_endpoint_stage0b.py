"""Focused synthetic contracts for the RT sparse-endpoint Stage 0B audit."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_endpoint_stage0b.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rt_sparse_endpoint_stage0b", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load_module()


def _reach(start: float, end: float, trial: int = 0) -> dict:
    return {"trial_index": trial, "reach_index": 0, "start_s": start, "end_s": end,
            "left_bin": int(np.ceil(start / 0.02 - 1e-10)), "right_bin": int(np.floor(end / 0.02 + 1e-10))}


def test_exact_and_interpolated_endpoints(mod) -> None:
    times = np.array([0.0, 0.01, 0.02])
    pos = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    exact, exact_reason = mod.interpolate_endpoint(times, pos, 0.01)
    interpolated, interpolation_reason = mod.interpolate_endpoint(times, pos, 0.015)
    assert exact_reason is None and np.array_equal(exact, np.array([1.0, 2.0]))
    assert interpolation_reason is None and np.allclose(interpolated, np.array([1.5, 3.0]))


def test_endpoint_bracket_over_20ms_fails_closed(mod) -> None:
    point, reason = mod.interpolate_endpoint(np.array([0.0, 0.021]), np.array([[0.0, 0.0], [1.0, 1.0]]), 0.01)
    assert point is None
    assert reason == "endpoint_bracket_exceeds_20ms"


def test_nonused_position_nan_does_not_poison_endpoint_but_bracket_nan_fails(mod) -> None:
    times = np.array([0.0, 0.01, 0.02, 0.03])
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [3.0, 0.0]])
    exact, reason = mod.interpolate_endpoint(times, positions, 0.01)
    assert reason is None and np.array_equal(exact, np.array([1.0, 0.0]))
    interpolated, bracket_reason = mod.interpolate_endpoint(times, positions, 0.015)
    assert interpolated is None
    assert bracket_reason == "endpoint_bracket_position_nonfinite"


def test_scalar_nwb_conversion_and_unit_contract(mod) -> None:
    converted = mod.apply_nwb_conversion(np.array([[1.0, 2.0]]), 2.0, -1.0, label="synthetic")
    assert np.array_equal(converted, np.array([[1.0, 3.0]]))
    assert mod.require_unit(" cm/s ", allowed={"cm/s"}, label="synthetic") == "cm/s"
    with pytest.raises(ValueError, match="must be scalar"):
        mod.apply_nwb_conversion(np.array([1.0]), np.array([1.0, 2.0]), 0.0, label="synthetic")
    with pytest.raises(ValueError, match="unit"):
        mod.require_unit("m", allowed={"cm"}, label="synthetic")


def test_short_endpoint_displacement_has_no_label(mod) -> None:
    reaches = [_reach(0.0, 0.02)]
    mod.attach_endpoint_labels(reaches, np.array([0.0, 0.01, 0.02]), np.array([[0.0, 0.0], [0.1, 0.0], [0.49, 0.0]]))
    assert reaches[0]["endpoint_label"] is False
    assert reaches[0]["endpoint_reason"] == "short_endpoint_displacement"


def test_shared_endpoint_is_deduplicated_for_raw_scalar_accounting(mod) -> None:
    reaches = [_reach(0.0, 0.02), _reach(0.02, 0.04)]
    times = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    pos = np.array([[0.0, 0.0], [0.3, 0.0], [1.0, 0.0], [1.3, 0.0], [2.0, 0.0]])
    mod.attach_endpoint_labels(reaches, times, pos)
    accounting = mod.endpoint_scalar_report(reaches)
    assert accounting["derived_direction_count"] == 2
    assert accounting["unique_endpoint_timestamps_s"] == [0.0, 0.02, 0.04]
    assert accounting["raw_scalar_coordinates"] == 6


def test_final_reach_uses_trial_stop_and_malformed_cues_fail_closed(mod) -> None:
    reaches, records = mod.parse_reaches(np.array([0.0]), np.array([3.0]), np.array([[1.0, 2.0]]), np.array([2]))
    assert [(row["start_s"], row["end_s"]) for row in reaches] == [(1.0, 2.0), (2.0, 3.0)]
    assert records[0]["complete_cue"] is True
    bad_reaches, bad_records = mod.parse_reaches(np.array([0.0]), np.array([3.0]), np.array([[1.0, 2.0]]), np.array([1]))
    assert bad_reaches == []
    assert bad_records[0]["exclusion_reason"] == "finite_undeclared_go_cue"


def test_primary_is_one_row_per_reach_and_dense_mask_cannot_change_it(mod) -> None:
    reach = _reach(0.0, 0.30)
    mod.attach_endpoint_labels([reach], np.arange(31) * 0.01, np.column_stack([np.arange(31) * 0.1, np.zeros(31)]))
    neural = np.zeros((20, 2), dtype=float)
    neural[:10, 0] = 1.0
    mod.attach_primary_neural_rows([reach], neural)
    assert reach["eligible_blocks"] >= 2
    assert reach["primary_row"] is True
    expected = np.asarray(reach["reach_mean_rate"]).copy()
    velocity = np.ones((20, 2), dtype=float)
    invalid_dense_mask = np.zeros(20, dtype=bool)
    mod.dense_audit_after_freeze([reach], velocity, invalid_dense_mask)
    assert reach["primary_row"] is True
    assert np.array_equal(reach["reach_mean_rate"], expected)
    assert reach["dense_audit_reason"] == "missing_or_empty_dense_velocity_interval"


def test_dense_retained_rows_use_the_sealed_all_active_union_rule(mod) -> None:
    reach = _reach(0.0, 0.14)
    mod.attach_endpoint_labels([reach], np.arange(15) * 0.01, np.column_stack([np.arange(15), np.zeros(15)]))
    velocity = np.ones((8, 2), dtype=float)
    mod.dense_audit_after_freeze([reach], velocity, np.ones(8, dtype=bool))
    assert reach["dense_retained_rows"] == 1
    velocity[3] = 0.0  # In both sealed active spans' union [0,7).
    mod.dense_audit_after_freeze([reach], velocity, np.ones(8, dtype=bool))
    assert reach["dense_retained_rows"] == 0


def test_all15_gate_aggregation_requires_every_session(mod) -> None:
    good = {"gate_conditions": {"endpoint_labels": True, "direction_design": True,
                                "dense_audit_availability": True, "dense_audit_agreement": True}, "gate_pass": True}
    rows = {f"ses-{index:02d}": dict(good) for index in range(15)}
    allowed = set(rows)
    aggregate = mod.aggregate_all15(rows, allowed_sessions=allowed)
    assert aggregate["all_sessions_pass"] is True
    assert aggregate["status"] == "PASS_STAGE0B_ENDPOINT_CONSTRUCTIBLE_NO_GPU"
    rows["ses-07"] = {"gate_conditions": {**good["gate_conditions"], "dense_audit_agreement": False}, "gate_pass": False}
    failed = mod.aggregate_all15(rows, allowed_sessions=allowed)
    assert failed["all_sessions_pass"] is False
    assert failed["failing_sessions"] == {"ses-07": ["dense_audit_agreement"]}
    with pytest.raises(ValueError, match="expected 15"):
        mod.aggregate_all15(dict(list(rows.items())[:-1]), allowed_sessions=allowed)


def test_frozen_scope_output_and_niceness_fail_closed(mod, tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="data-root must equal"):
        mod.validate_run_scope(tmp_path, mod.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="output-dir must equal"):
        mod.validate_run_scope(mod.DATA_ROOT, tmp_path / "other-output")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setattr(mod.os, "nice", lambda _increment: 9)
    with pytest.raises(ValueError, match="niceness"):
        mod.validate_cpu_environment()
    monkeypatch.setattr(mod.os, "nice", lambda _increment: 10)
    _caps, niceness = mod.validate_cpu_environment()
    assert niceness == 10


def test_exact_stage0_allowlist_rejects_same_count_wrong_path(mod, tmp_path: Path) -> None:
    paths = []
    allowlist = {}
    for index in range(15):
        path = tmp_path / f"sub-C_ses-RT-2013{index:04d}_behavior+ecephys.nwb"
        path.write_bytes(bytes([index]))
        session = path.name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb")
        paths.append(path)
        allowlist[session] = {"nwb_path": str(path.resolve()), "nwb_size_bytes": path.stat().st_size}
    mod.validate_discovered_scope(paths, allowlist)
    allowlist[next(iter(allowlist))]["nwb_path"] = str((tmp_path / "wrong.nwb").resolve())
    with pytest.raises(ValueError, match="path differs"):
        mod.validate_discovered_scope(paths, allowlist)


def test_implementation_provenance_binds_code_tests_and_runtime(mod) -> None:
    provenance = mod.implementation_provenance()
    assert provenance["not_a_gate"] is True
    assert provenance["script"]["path"] == str(SCRIPT.resolve())
    assert provenance["script"]["sha256"] == mod.sha256_file(SCRIPT)
    assert provenance["focused_test"]["path"] == str(Path(__file__).resolve())
    assert provenance["focused_test"]["sha256"] == mod.sha256_file(Path(__file__))
    runtime = provenance["runtime"]
    assert isinstance(runtime["python_version"], str) and runtime["python_version"]
    assert isinstance(runtime["numpy_version"], str) and runtime["numpy_version"]
    assert runtime["pynwb_version"] is None or isinstance(runtime["pynwb_version"], str)


def test_atomic_receipt_is_new_directory_and_mode_0444(mod, tmp_path: Path) -> None:
    receipt = mod.write_atomic_receipt(tmp_path / "stage0b", {"schema": "synthetic"})
    assert receipt.is_file()
    assert (receipt.stat().st_mode & 0o777) == 0o444
    with pytest.raises(FileExistsError):
        mod.write_atomic_receipt(tmp_path / "stage0b", {"schema": "again"})


def test_script_has_no_torch_import_or_gpu_enablement() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "CUDA_VISIBLE_DEVICES must be empty" in source
    assert "dense_audit_after_freeze" in source
    assert os.path.basename(str(SCRIPT)) == "rt_sparse_endpoint_stage0b.py"
