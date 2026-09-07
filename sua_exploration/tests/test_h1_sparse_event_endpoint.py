from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import h5py

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "SPINT-main/data/000954"


def event(index: int, trial: int) -> v1.MovementEvent:
    displacement = np.zeros(7, dtype=np.float64)
    displacement[index % 7] = 1.0 + index / 10.0
    return v1.MovementEvent(
        row_id=index,
        tag=v1.MOVEMENT_TAGS[index % len(v1.MOVEMENT_TAGS)],
        trial_value=float(trial + 1),
        trial_index=trial,
        start_time=float(index),
        stop_time=float(index) + 0.5,
        duration_seconds=0.5,
        eval_bins=25,
        displacement=displacement,
        log_rates=np.full(v1.EXPECTED_NEURONS, index / 100.0, dtype=np.float64),
    )


def test_endpoint_interpolation_is_linear_and_seven_dimensional() -> None:
    times = np.asarray([0.0, 0.02, 0.04])
    positions = np.stack([np.arange(7) + offset for offset in (0.0, 2.0, 4.0)])
    value = v1.interpolate_position(times, positions, 0.01)
    np.testing.assert_allclose(value, np.arange(7) + 1.0)
    with pytest.raises(v1.SparseEventEndpointError):
        v1.interpolate_position(times, positions, 0.07)


def test_native_position_conversion_offset_and_time_axis(tmp_path: Path) -> None:
    path = tmp_path / "position.h5"
    raw = np.arange(21, dtype=np.float64).reshape(3, 7)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("position")
        data = group.create_dataset("data", data=raw)
        data.attrs["conversion"] = 2.0
        data.attrs["offset"] = -3.0
        starting = group.create_dataset("starting_time", data=1.25)
        starting.attrs["rate"] = v1.BIN_SECONDS
    with h5py.File(path, "r") as handle:
        times, values, step, conversion, offset = v1._converted_series(
            handle["position"], expected_dim=v1.POSITION_DIM,
        )
    np.testing.assert_allclose(times, [1.25, 1.27, 1.29], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(values, raw * 2.0 - 3.0, rtol=0.0, atol=0.0)
    assert step == v1.BIN_SECONDS
    assert conversion == 2.0
    assert offset == -3.0


def test_closed_form_carrier_column_order_is_weights_then_intercept() -> None:
    rng = np.random.default_rng(42)
    z = rng.normal(size=(24, 3))
    weights = rng.normal(size=(3, v1.EXPECTED_NEURONS))
    intercept = rng.normal(size=(v1.EXPECTED_NEURONS,))
    response = z @ weights + intercept
    carrier = v1.fit_carrier_from_arrays(z, response, ridge_lambda=0.0)
    assert carrier.shape == (v1.EXPECTED_NEURONS, 4)
    np.testing.assert_allclose(carrier[:, :3], weights.T, atol=1e-10)
    np.testing.assert_allclose(carrier[:, 3], intercept, atol=1e-10)


def test_v2_normalized_ridge_matches_frozen_closed_form() -> None:
    rng = np.random.default_rng(420)
    z = rng.normal(size=(31, v2.LATENT_DIM))
    response = rng.normal(size=(31, v1.EXPECTED_NEURONS))
    carrier = v2.fit_carrier_arrays(z, response)
    design = np.column_stack((np.ones(z.shape[0]), z))
    penalty = np.diag([0.0] + [1.0] * v2.LATENT_DIM) * (z.shape[0] * v2.RIDGE_LAMBDA)
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    expected = np.column_stack((coefficient[1:].T, coefficient[0]))
    assert carrier.shape == (v1.EXPECTED_NEURONS, v2.CARRIER_DIM)
    np.testing.assert_allclose(carrier, expected, rtol=0.0, atol=1.0e-12)


def test_event_label_and_channel_row_shuffles_have_no_fixed_points() -> None:
    events = tuple(event(index, 0 if index < 4 else 1) for index in range(8))
    order, manifest = v1.within_trial_label_shuffle(events, session=v1.H1_HELDIN_SESSIONS[0], budget=3)
    assert manifest["fixed_points"] == 0
    assert sorted(order.tolist()) == list(range(8))
    assert not np.any(order == np.arange(8))
    carrier = np.arange(v1.EXPECTED_NEURONS * 5, dtype=np.float64).reshape(v1.EXPECTED_NEURONS, 5)
    shuffled, row_manifest = v2.row_shuffle(carrier, session=v1.H1_HELDIN_SESSIONS[0], budget=3)
    assert row_manifest["fixed_points"] == 0
    assert not np.array_equal(shuffled, carrier)
    np.testing.assert_array_equal(np.sort(shuffled[:, 0]), np.sort(carrier[:, 0]))


def test_path_scope_rejects_non_heldin_and_formal_paths(tmp_path: Path) -> None:
    with pytest.raises(v1.SparseEventEndpointError):
        v1.reject_path_scope(tmp_path / "sub-HumanPitt-held-out-calib/file.nwb")
    with pytest.raises(v1.SparseEventEndpointError):
        v1.reject_path_scope(tmp_path / "random/file.nwb")


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_real_h1_parser_does_not_require_all_eight_phases_in_m3() -> None:
    paths = v1.index_heldin_calib(DATA_ROOT)
    session = v1.load_event_session(paths[v1.H1_HELDIN_SESSIONS[0]])
    support = session.events_before(3)
    assert 13 <= len(support) <= 24
    assert len({event.tag for event in support}) < len(v1.MOVEMENT_TAGS)
    assert all(event.trial_index < 3 for event in support)
    assert session.exclusion_counts["cross_trial_or_unassigned"] > 0
    assert session.exclusion_counts["fewer_than_five_eval_bins"] > 0
    support_m4 = session.events_before(4)
    later_m3 = session.events_after(3)
    later_m4 = session.events_after(4)
    support_ids = {item.row_id for item in support}
    support_m4_ids = {item.row_id for item in support_m4}
    later_m3_ids = {item.row_id for item in later_m3}
    later_m4_ids = {item.row_id for item in later_m4}
    all_ids = {item.row_id for item in session.events}
    assert support_ids.isdisjoint(later_m3_ids)
    assert support_m4_ids.isdisjoint(later_m4_ids)
    assert support_ids | later_m3_ids == all_ids
    assert support_m4_ids | later_m4_ids == all_ids


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_v2_basis_excludes_outer_date_and_has_canonical_component_signs() -> None:
    paths = v1.index_heldin_calib(DATA_ROOT)
    sessions = {name: v1.load_event_session(path) for name, path in paths.items()}
    outer_date = v1.H1_DATES[0]
    basis = v2.fit_source_all_event_basis(sessions, outer_date=outer_date)
    assert all(v1.session_date(name) != outer_date for name in basis.source_sessions)
    assert set(basis.source_sessions) == {
        name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date
    }
    for component in basis.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] > 0
    assert basis.components.shape == (v2.LATENT_DIM, v1.POSITION_DIM)
    assert basis.retained_variance >= v2.RETAINED_MINIMUM


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_real_h1_parser_never_opens_dense_velocity_and_uses_native_endpoints(monkeypatch) -> None:
    """Bind the sparse label path to native position endpoints on a real NWB.

    The proxy makes an accidental read of the dense velocity TimeSeries a hard
    failure.  The independent endpoint reconstruction then verifies that the
    stored event label is exactly position(stop)-position(start).
    """

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

    monkeypatch.setattr(v1.h5py, "File", VelocityRejectingFile)
    session = v1.load_event_session(path)
    first = session.events[0]

    # Use the unpatched constructor to reconstruct the endpoint displacement
    # independently from the native position TimeSeries.
    with original_file(path, "r") as handle:
        group = handle["acquisition/OpenLoopKinematics"]
        times, positions, _step, _conversion, _offset = v1._converted_series(group, expected_dim=7)
    expected = (
        v1.interpolate_position(times, positions, first.stop_time)
        - v1.interpolate_position(times, positions, first.start_time)
    )
    np.testing.assert_allclose(first.displacement, expected, rtol=0.0, atol=1.0e-12)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="public H1 held-in data unavailable")
def test_frozen_v1_v2_and_accounting_receipts_verify() -> None:
    commands = (
        (
            ROOT / "sua_exploration/scripts/verify_h1_sparse_event_endpoint_receipt.py",
            ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v1/source_audit.json",
        ),
        (
            ROOT / "sua_exploration/scripts/verify_h1_sparse_event_endpoint_v2_receipt.py",
            ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json",
        ),
        (
            ROOT / "sua_exploration/scripts/verify_h1_sparse_event_endpoint_v2r2_accounting.py",
            ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit_v2r2.json",
        ),
    )
    for script, receipt in commands:
        result = subprocess.run([sys.executable, str(script), str(receipt)], check=True, capture_output=True, text=True)
        assert json.loads(result.stdout)["status"] == "PASS"
