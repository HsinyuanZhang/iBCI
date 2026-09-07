"""Focused tests for the H1 within-event position trajectory geometry diagnostic."""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from sua_exploration.mc_maze import h1_curvature_geometry as geom
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "SPINT-main/data/000954"
TOL = 1.0e-12


def _line_positions(delta: np.ndarray) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, geom.SAMPLE_COUNT, dtype=np.float64)
    return fractions[:, None] * np.asarray(delta, dtype=np.float64)[None, :]


def test_straight_constant_speed_trajectory_metrics() -> None:
    delta = np.array([8.0, 0.0, 0.0, 1.0, 0.5, 0.25, 0.125], dtype=np.float64)
    metrics = geom.measure_trajectory(_line_positions(delta))
    assert metrics["max_deviation_ratio"] == pytest.approx(0.0, abs=TOL)
    assert metrics["arc_chord_ratio"] == pytest.approx(1.0, abs=TOL)
    assert metrics["speed_cv"] == pytest.approx(0.0, abs=TOL)


def test_straight_nonuniform_speed_is_independent_of_arc_chord() -> None:
    """Collinear uneven parameterization keeps arc_chord_ratio at 1 while speed_cv rises."""

    delta = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    alphas = np.asarray([0.0, 0.01, 0.02, 0.30, 0.50, 0.51, 0.52, 0.99, 1.0], dtype=np.float64)
    positions = alphas[:, None] * delta[None, :]
    metrics = geom.measure_trajectory(positions)
    assert metrics["arc_chord_ratio"] == pytest.approx(1.0, abs=TOL)
    assert metrics["speed_cv"] > 0.1
    assert metrics["max_deviation_ratio"] is not None and metrics["max_deviation_ratio"] > 0.1


def test_semicircular_path_metrics() -> None:
    radius = 1.0
    angles = np.linspace(0.0, math.pi, geom.SAMPLE_COUNT, dtype=np.float64)
    xy = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    positions = np.zeros((geom.SAMPLE_COUNT, v1.POSITION_DIM), dtype=np.float64)
    positions[:, 0:2] = xy
    metrics = geom.measure_trajectory(positions)
    assert metrics["arc_chord_ratio"] == pytest.approx(math.pi / 2.0, rel=0.02)
    assert metrics["max_deviation_ratio"] is not None and metrics["max_deviation_ratio"] > 0.25


def test_degenerate_chord_is_counted_but_excluded_from_ratios() -> None:
    positions = np.zeros((geom.SAMPLE_COUNT, v1.POSITION_DIM), dtype=np.float64)
    metrics = geom.measure_trajectory(positions)
    assert metrics["degenerate_chord"] is True
    assert metrics["max_deviation_ratio"] is None
    assert metrics["mean_deviation_ratio"] is None
    assert metrics["arc_chord_ratio"] is None
    assert metrics["chord_norm"] <= geom.CHORD_FLOOR


def test_even_time_sampling_matches_event_endpoints() -> None:
    start = 1.0
    stop = 1.64
    sample_times = geom.sample_event_times(start, stop)
    assert sample_times.shape == (geom.SAMPLE_COUNT,)
    assert sample_times[0] == pytest.approx(start, abs=TOL)
    assert sample_times[-1] == pytest.approx(stop, abs=TOL)
    assert np.allclose(np.diff(sample_times), np.full(geom.SAMPLE_COUNT - 1, (stop - start) / 8.0), atol=TOL)

    times = np.arange(0.0, 2.0, 0.02, dtype=np.float64)
    positions = np.column_stack([times, times ** 2, np.sin(times), np.zeros_like(times),
                                 np.ones_like(times), times / 10.0, times / 100.0])
    sampled = geom.sample_event_positions(times, positions, start, stop)
    assert sampled is not None
    np.testing.assert_allclose(sampled[0], v1.interpolate_position(times, positions, start), atol=TOL)
    np.testing.assert_allclose(sampled[-1], v1.interpolate_position(times, positions, stop), atol=TOL)


def test_aggregate_excludes_degenerate_chord_from_ratio_statistics() -> None:
    good = geom.EventGeometry(
        session_name=v1.H1_HELDIN_SESSIONS[0],
        tag="Reach",
        trial_index=0,
        chord_norm=1.0,
        max_deviation=0.1,
        mean_deviation=0.05,
        max_deviation_ratio=0.1,
        mean_deviation_ratio=0.05,
        arc_length=1.2,
        arc_chord_ratio=1.2,
        speed_cv=0.2,
        degenerate_chord=False,
        dof_curvature_ratio=np.ones(7, dtype=np.float64),
        dof_degenerate=np.zeros(7, dtype=bool),
    )
    degenerate = geom.EventGeometry(
        session_name=v1.H1_HELDIN_SESSIONS[0],
        tag="Reach",
        trial_index=0,
        chord_norm=0.0,
        max_deviation=0.0,
        mean_deviation=0.0,
        max_deviation_ratio=None,
        mean_deviation_ratio=None,
        arc_length=0.0,
        arc_chord_ratio=None,
        speed_cv=0.0,
        degenerate_chord=True,
        dof_curvature_ratio=np.zeros(7, dtype=np.float64),
        dof_degenerate=np.ones(7, dtype=bool),
    )
    block = geom._aggregate_block((good, degenerate))
    assert block["event_count"] == 2
    assert block["degenerate_chord_count"] == 1
    assert block["max_deviation_ratio"]["count"] == 1
    assert block["max_deviation_ratio"]["median"] == pytest.approx(0.1)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_diagnostic_never_opens_dense_velocity(monkeypatch) -> None:
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

    monkeypatch.setattr(geom.h5py, "File", VelocityRejectingFile)
    monkeypatch.setattr(v1.h5py, "File", VelocityRejectingFile)
    session = v1.load_event_session(path)
    measured, rejections = geom.measure_session(session)
    assert len(measured) > 0
    assert rejections["sample_read_failed"] == 0
    assert measured[0].max_deviation_ratio is not None
