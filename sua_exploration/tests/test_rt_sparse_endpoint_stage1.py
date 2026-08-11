"""Synthetic contracts for the sparse-endpoint Stage 1 CPU screen."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_endpoint_stage1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rt_sparse_endpoint_stage1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load_module()


def rows_for(coefficients: np.ndarray, angles: list[float], *, trial_offset: int = 0) -> list[dict]:
    rows = []
    for index, theta in enumerate(angles):
        x = np.array([1.0, np.cos(theta), np.sin(theta)])
        rows.append({"trial_index": trial_offset + index, "theta_rad": theta, "primary_row": True,
                     "endpoint_label": True, "start_s": float(index), "end_s": float(index + 1),
                     "reach_mean_rate": x @ coefficients})
    return rows


def test_synthetic_coefficient_recovery_and_split_cosine(mod) -> None:
    coefficient = np.array([[2.0, 5.0], [3.0, -2.0], [-4.0, 1.0]])
    angles = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    full, fit = mod.fit_ac4(rows_for(coefficient, angles))
    assert fit["status"] == "defined"
    assert np.allclose(full, coefficient)
    first, _ = mod.fit_ac4(rows_for(coefficient, angles))
    second, _ = mod.fit_ac4(rows_for(coefficient, angles))
    values, summary = mod.split_cosines(first, second, channels=2)
    assert np.allclose(values, np.ones(2))
    assert summary["median"] == pytest.approx(1.0) and summary["fraction_ge_040"] == 1.0


def test_split_cosine_requires_each_half_norm_above_epsilon(mod) -> None:
    # A large partner cannot rescue a first-half vector at/below the fixed epsilon.
    first = np.array([[0.0], [mod.NORM_EPSILON], [0.0]])
    second = np.array([[0.0], [10.0], [0.0]])
    values, summary = mod.split_cosines(first, second, channels=1)
    assert values[0] != values[0] and summary["defined_channels"] == 0
    # Both individual norms above epsilon are defined even when their product is below epsilon.
    first = np.array([[0.0], [2.0e-12], [0.0]])
    second = np.array([[0.0], [2.0e-12], [0.0]])
    values, summary = mod.split_cosines(first, second, channels=1)
    assert values[0] == pytest.approx(1.0) and summary["defined_channels"] == 1


def test_deterministic_shuffle_preserves_multiset_and_has_no_self_identity(mod) -> None:
    order, shift = mod.deterministic_rotation(7, session="ses-synthetic")
    assert 1 <= shift < 7
    assert sorted(order.tolist()) == list(range(7))
    assert not np.any(order == np.arange(7))
    labels = np.arange(7) * 0.1
    assert np.array_equal(np.sort(labels[order]), np.sort(labels))


def test_later_reach_r2_deltas_and_constant_channel_undefined(mod) -> None:
    coefficient = np.array([[1.0, 4.0], [2.0, 0.0], [-1.0, 0.0]])
    angles = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    support = rows_for(coefficient, angles)
    later = rows_for(coefficient, angles, trial_offset=24)
    audit = mod.forward_transfer("ses-synthetic", support, later, channels=2)
    assert audit["status"] == "defined"
    assert audit["defined_channels"] == 1
    assert audit["undefined_channels"] == 1
    assert audit["session_median_shuffle"] is not None and audit["session_median_shuffle"] > 0
    assert audit["per_channel_delta_shuffle"][1] is None


def test_mean_positive_median_negative_cannot_pass_forward_gate(mod) -> None:
    names = [f"ses-{index:02d}" for index in range(15)]
    rows = {}
    for index, name in enumerate(names):
        # Seven large positives make the mean positive; eight negatives make the median negative.
        shuffle = 1.0 if index < 7 else -0.10
        rows[name] = {"split": {"median": 0.8, "per_channel": [0.8, 0.7]},
                      "forward": {"session_median_shuffle": shuffle, "session_median_intercept": 0.2}}
    aggregate = mod.aggregate_stage1(rows, allowlist=set(names))
    report = aggregate["forward_correct_minus_shuffle"]
    assert report["mean"] > 0 and report["median"] < 0
    assert aggregate["gate_conditions"]["correct_pair_transfer"] is False
    assert aggregate["status"].startswith("STOP_")
    assert [row["session"] for row in report["ordered_values"]] == names


def test_all_undefined_aggregate_stops_without_key_error(mod) -> None:
    names = [f"ses-{index:02d}" for index in range(15)]
    rows = {name: {"split": {"median": None, "per_channel": [None]},
                   "forward": {"session_median_shuffle": None, "session_median_intercept": None}} for name in names}
    aggregate = mod.aggregate_stage1(rows, allowlist=set(names))
    assert aggregate["status"].startswith("STOP_")
    for key in ("forward_correct_minus_shuffle", "forward_correct_minus_intercept"):
        assert aggregate[key]["defined_sessions"] == 0
        assert aggregate[key]["removed_session"] is None
        assert len(aggregate[key]["ordered_values"]) == 15


def test_leave_largest_out_removes_largest_absolute_value(mod) -> None:
    summary = mod._summary([0.1, -0.7, 0.2])
    assert summary["removed_session_index"] == 1
    assert summary["leave_largest_out_mean"] == pytest.approx(0.15)


def test_stage0b_identity_drift_fails_before_fit(mod) -> None:
    coefficient = np.array([[1.0], [2.0], [3.0]])
    rows = rows_for(coefficient, [0.0, np.pi / 2, np.pi])
    for row in rows:
        row.update({"start_s": float(row["trial_index"]), "end_s": float(row["trial_index"] + 1)})
    design = mod.primary_design(rows); scalar = mod.endpoint_scalar_accounting(rows)
    expected = {"m24": {"endpoint_labelled_reaches": 3, "primary_reach_rows": 3,
                          "primary_reach_design": design,
                          "endpoint_scalar_accounting": {**scalar, "dense_rt_retained_rows": 0, "dense_rt_target_scalars": 0}}}
    mod.validate_stage0b_identity("ses-ok", rows, expected)
    expected["m24"]["primary_reach_rows"] = 4
    with pytest.raises(ValueError, match="primary-row identity drift"):
        mod.validate_stage0b_identity("ses-drift", rows, expected)


def test_stage1_source_has_no_continuous_stream_access() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "cursor_vel" not in source
    assert "Velocity" not in source
    assert "data_interfaces" not in source


def test_scope_niceness_atomic_and_implementation_provenance(mod, tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="data-root"):
        mod.validate_scope(tmp_path, mod.DEFAULT_OUTPUT, {"sessions": {}})
    with pytest.raises(ValueError, match="output-dir"):
        mod.validate_scope(mod.DATA_ROOT, tmp_path / "wrong", {"sessions": {}})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setattr(mod.os, "nice", lambda _value: 9)
    with pytest.raises(ValueError, match="niceness"):
        mod.validate_cpu()
    monkeypatch.setattr(mod.os, "nice", lambda _value: 10)
    _caps, niceness = mod.validate_cpu(); assert niceness == 10
    receipt = mod.write_atomic(tmp_path / "stage1", {"schema": "synthetic"})
    assert receipt.is_file() and (receipt.stat().st_mode & 0o777) == 0o444
    with pytest.raises(ValueError, match="output exists"):
        mod.write_atomic(tmp_path / "stage1", {})
    provenance = mod.implementation_provenance()
    assert provenance["not_a_gate"] is True
    assert provenance["script"]["sha256"] == mod.sha256_file(SCRIPT)
    assert provenance["focused_test"]["path"] == str(Path(__file__).resolve())
