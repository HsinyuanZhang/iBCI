from __future__ import annotations

import hashlib
import h5py
import json
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import h1_kpoint_event_carrier as screen
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "SPINT-main/data/000954"


def _load_real_sessions() -> dict[str, v1.EventSession]:
    paths = v1.index_heldin_calib(DATA_ROOT)
    return {name: v1.load_event_session(paths[name]) for name in v1.H1_HELDIN_SESSIONS}


def test_k2_families_match_event_displacement() -> None:
    sessions = _load_real_sessions()
    for name in v1.H1_HELDIN_SESSIONS:
        kpoint_session = screen.load_kpoint_session(sessions[name], 2)
        for event in kpoint_session.events:
            inc = screen.raw_increments(event.positions)
            curv = screen.raw_delta_curvature(event.positions)
            np.testing.assert_allclose(inc, event.base.displacement, rtol=0.0, atol=1.0e-12)
            np.testing.assert_allclose(curv, event.base.displacement, rtol=0.0, atol=1.0e-12)


def test_even_time_spacing_and_endpoints() -> None:
    sessions = _load_real_sessions()
    session = sessions[v1.H1_HELDIN_SESSIONS[0]]
    with h5py.File(session.path, "r") as handle:
        group = handle["acquisition/OpenLoopKinematics"]
        times, positions, _step, _conversion, _offset = v1._converted_series(group, expected_dim=7)
    event = session.events[0]
    for k in (2, 3, 5):
        sample_times = screen.event_sample_times(event.start_time, event.stop_time, k)
        assert sample_times[0] == event.start_time
        assert sample_times[-1] == event.stop_time
        if k > 2:
            expected = event.start_time + np.arange(k, dtype=np.float64) / float(k - 1) * (event.stop_time - event.start_time)
            np.testing.assert_allclose(sample_times, expected, rtol=0.0, atol=1.0e-12)
        sampled = screen.read_event_positions(times, positions, event, k)
        assert sampled is not None
        assert sampled.shape == (k, 7)


def test_straight_line_synthetic_features() -> None:
    p0 = np.zeros(7, dtype=np.float64)
    direction = np.arange(7, dtype=np.float64) + 1.0
    for k in (3, 5):
        positions = np.stack([p0 + direction * (j / float(k - 1)) for j in range(k)], axis=0)
        increments = screen.raw_increments(positions)
        expected_inc = direction / float(k - 1)
        for block in increments.reshape(k - 1, 7):
            np.testing.assert_allclose(block, expected_inc, rtol=0.0, atol=1.0e-12)
        curv = screen.raw_delta_curvature(positions)
        delta = curv[:7]
        np.testing.assert_allclose(delta, direction, rtol=0.0, atol=1.0e-12)
        if k > 2:
            np.testing.assert_allclose(curv[7:], 0.0, rtol=0.0, atol=1.0e-12)


def test_curved_trajectory_has_nonzero_curvature_block() -> None:
    k = 5
    positions = np.zeros((k, 7), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 1.0, k)
    positions[2, 1] = 0.5
    curv = screen.raw_delta_curvature(positions)
    assert np.linalg.norm(curv[7:]) > 1.0e-6


def test_flat_arms_zero_curvature_block() -> None:
    k = 5
    positions = np.zeros((k, 7), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 1.0, k)
    positions[2, 1] = 0.5
    flat_curv = screen.raw_delta_curvature(positions, flat_interior=True)
    np.testing.assert_allclose(flat_curv[7:], 0.0, rtol=0.0, atol=1.0e-12)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_scope_rejects_velocity_series(monkeypatch) -> None:
    paths = v1.index_heldin_calib(DATA_ROOT)
    path = paths[v1.H1_HELDIN_SESSIONS[0]]
    original_file = h5py.File

    class VelocityRejectingFile:
        def __init__(self, *args, **kwargs):
            self._file = original_file(*args, **kwargs)

        def __enter__(self):
            self._file.__enter__()
            return self

        def __exit__(self, *args):
            return self._file.__exit__(*args)

        def __getitem__(self, key):
            assert "OpenLoopKinematicsVelocity" not in str(key)
            return self._file[key]

        def __getattr__(self, name):
            return getattr(self._file, name)

    monkeypatch.setattr(screen.h5py, "File", VelocityRejectingFile)
    sessions = _load_real_sessions()
    body = screen.run_screen(sessions)
    assert body["scope"]["dense_velocity_series_opened"] is False
    assert body["scope"]["cuda_used"] is False


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_deterministic_canonical_json() -> None:
    sessions = _load_real_sessions()
    first = screen.run_screen(sessions)
    second = screen.run_screen(sessions)
    first_bytes = v1.canonical_json_bytes(first)
    second_bytes = v1.canonical_json_bytes(second)
    assert first_bytes == second_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(second_bytes).hexdigest()


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_sealed_hse5_reproduction_within_tolerance() -> None:
    sessions = _load_real_sessions()
    body = screen.run_screen(sessions)
    integrity = body["integrity_checks"]["sealed_hse5_reproduction"]
    assert integrity["maximum_absolute_difference"] is not None
    assert integrity["maximum_absolute_difference"] <= screen.SEALED_TOLERANCE
