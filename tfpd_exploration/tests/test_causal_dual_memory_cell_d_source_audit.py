"""No-data/CPU gates for the CDM-D source-adapter and safety-audit scaffold."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tfpd_exploration") not in sys.path:
    sys.path.insert(0, str(ROOT / "tfpd_exploration"))

from src.causal_dual_memory_cell_d_v1 import core  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import lifecycle  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import physical  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_adapter as adapter  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_audit as audit  # noqa: E402


SESSION = "sub-C_ses-CO-synthetic"


def _channels(units: int = 8) -> np.ndarray:
    return np.arange(100, 100 + units, dtype=np.int64)


def _accepted_b3s(full_counts: np.ndarray) -> np.ndarray:
    # Synthetic stand-in for a frozen datamodule cubic/pad helper: exact [100,N]
    # content is independently supplied and must be reproduced byte-for-byte.
    base = np.asarray(full_counts, dtype=np.float64).mean(axis=0)
    ramp = np.linspace(-0.25, 0.25, 100, dtype=np.float64)[:, None]
    return np.ascontiguousarray(base[None, :] + ramp)


def _record(
    *,
    trial_id: str = "trial-0",
    units: int = 8,
    full_counts: np.ndarray | None = None,
    rebuilder=None,
    endpoints: np.ndarray | None = None,
    availability: np.ndarray | None = None,
    channel_ids: np.ndarray | None = None,
) -> adapter.HeldSourceTrialRecord:
    channels = _channels(units) if channel_ids is None else np.asarray(channel_ids, dtype=np.int64)
    assert channels.shape == (units,)
    raw = (
        np.tile(np.arange(1, units + 1, dtype=np.int64), (128, 1))
        if full_counts is None else np.asarray(full_counts)
    )
    accepted = _accepted_b3s(raw)
    build = (lambda record: record.accepted_calibration_b3s_activity.copy()) if rebuilder is None else rebuilder
    endpoint_rows = np.arange(59, 65, dtype=np.int64) if endpoints is None else np.asarray(endpoints, dtype=np.int64)
    available = np.ones(endpoint_rows.size, dtype=np.bool_) if availability is None else np.asarray(availability, dtype=np.bool_)
    source_record_sha = hashlib.sha256(f"{SESSION}/{trial_id}".encode()).hexdigest()
    channel_sha = core.channel_order_digest(channels)
    full_sha = core.array_digest(raw)
    accepted_sha = core.array_digest(accepted)
    descriptor = adapter.SourceTrialDescriptor(
        session_id=SESSION,
        trial_id=trial_id,
        source_record_sha256=source_record_sha,
        channel_order_sha256=channel_sha,
        full_native_counts_sha256=full_sha,
        accepted_b3s_activity_sha256=accepted_sha,
        raw_trial_binding_sha256=adapter.raw_trial_binding_sha256(
            session_id=SESSION,
            trial_id=trial_id,
            source_record_sha256=source_record_sha,
            channel_order_sha256=channel_sha,
            full_native_counts_sha256=full_sha,
            rewarded_interval_start_bin=10,
            rewarded_interval_stop_bin=100,
            accepted_b3s_activity_sha256=accepted_sha,
            window_endpoint_bins_sha256=core.array_digest(endpoint_rows),
            neural_endpoint_available_sha256=core.array_digest(available),
            b3s_rebuilder_semantics=adapter.B3S_REBUILDER_SEMANTICS,
        ),
    )
    return adapter.HeldSourceTrialRecord(
        descriptor=descriptor,
        channel_ids=channels,
        full_native_binned_counts=raw,
        rewarded_interval_start_bin=10,
        rewarded_interval_stop_bin=100,
        accepted_calibration_b3s_activity=accepted,
        rebuild_b3s_activity=build,
        window_endpoint_bins=endpoint_rows,
        neural_endpoint_available=available,
    )


def _views(trial_id: str = "trial-0") -> adapter.SourceTrialViews:
    return adapter.materialize_source_trial_views(_record(trial_id=trial_id), roster=_roster())


def _roster() -> adapter.StrictSourceRoster:
    return adapter.StrictSourceRoster((SESSION,) + tuple(f"sub-C_ses-CO-{index:02d}" for index in range(26)))


def _behavior_normalizer() -> physical.SealedSourceBehaviorNormalizer:
    return physical.SealedSourceBehaviorNormalizer(
        mean=np.asarray([0.1, -0.2], dtype=np.float64),
        std=np.asarray([2.0, 3.0], dtype=np.float64),
        authority_sha256="c" * 64,
    )


def _state_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rng_snapshot() -> physical.RNGStateSnapshot:
    """Synthetic exact four-domain snapshot without importing Torch/CUDA."""
    py_state = repr(random.getstate()).encode("utf-8")
    np_state = np.random.get_state()
    np_bytes = b"|".join((
        np_state[0].encode("ascii"),
        np.ascontiguousarray(np_state[1]).tobytes(),
        str(np_state[2]).encode("ascii"),
        str(np_state[3]).encode("ascii"),
        repr(np_state[4]).encode("ascii"),
    ))
    return physical.RNGStateSnapshot(
        python_sha256=_state_sha(py_state),
        numpy_sha256=_state_sha(np_bytes),
        torch_cpu_sha256=_state_sha(b"synthetic-torch-cpu-state"),
        torch_cuda_sha256=_state_sha(b"synthetic-torch-cuda-state"),
    )


def _true(row: int, *, budget: int = 30, accepted: bool = True, pseudo: float | None = None) -> audit.PseudoAuditRow:
    theta = core.CANONICAL_DIRECTIONS_RAD[row % len(core.CANONICAL_DIRECTIONS_RAD)]
    return audit.PseudoAuditRow(
        budget=budget,
        session_id=SESSION,
        trial_id=f"audit-{budget}-{row}",
        accepted=accepted,
        rejection_reason=None if accepted else core.UpdateRejectionReason.LOW_MEAN_SPEED,
        pseudo_direction_radians=theta if pseudo is None and accepted else pseudo,
        true_direction=adapter.AuditOnlyTrueDirection(SESSION, f"audit-{budget}-{row}", theta),
        complementary_disagreement_radians=0.01 if accepted else None,
        canonical_distance_radians=0.02 if accepted else None,
        displacement=1.0 if accepted else None,
        mean_speed=1.0 if accepted else None,
        design_rank=3 if accepted else None,
        design_condition=10.0 if accepted else None,
        departure=0.1 if accepted else None,
        prefix_digest=hashlib.sha256(f"prefix-{budget}-{row}".encode()).hexdigest(),
    )


def _carrier_fixture(
    *,
    budget: int,
    pseudo_shift_indices: int | tuple[int, int, int, int] = 0,
    constant_native_rates: bool = False,
    channel_permutation: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> tuple[
    tuple[audit.GroupedPseudoAuditRow, ...],
    tuple[adapter.SourceTrialViews, ...],
    adapter.SourceAuditBudgetAuthority,
    core.ComplementaryGroups,
]:
    """Eight canonical directions with typed native count rows for B8 parity."""
    first30 = tuple(f"carrier-{budget}-first-{index:02d}" for index in range(30))
    next30 = tuple(f"carrier-{budget}-next-{index:02d}" for index in range(30))
    authority = adapter.build_source_audit_authorities(
        first30 + next30,
        m4_d_optimal_indices=(0, 8, 16, 24),
    )[budget]
    offsets = (
        (int(pseudo_shift_indices),) * core.GROUP_COUNT
        if isinstance(pseudo_shift_indices, int)
        else tuple(int(item) for item in pseudo_shift_indices)
    )
    assert len(offsets) == core.GROUP_COUNT
    permutation = np.asarray(channel_permutation, dtype=np.int64)
    assert np.array_equal(np.sort(permutation), np.arange(core.GROUP_COUNT, dtype=np.int64))
    channels = _channels(4)[permutation]
    groups = core.ComplementaryGroups(
        channel_ids=channels,
        assignment=np.arange(core.GROUP_COUNT, dtype=np.int64)[permutation],
        valid_mask=np.ones(core.GROUP_COUNT, dtype=np.bool_),
    )
    rows: list[audit.GroupedPseudoAuditRow] = []
    views: list[adapter.SourceTrialViews] = []
    for index, trial_id in enumerate(authority.audit_trial_ids):
        theta = core.CANONICAL_DIRECTIONS_RAD[index % len(core.CANONICAL_DIRECTIONS_RAD)]
        if constant_native_rates:
            per_bin = np.asarray([7, 7, 7, 7], dtype=np.int64)
        else:
            # Integer per-bin counts yield native rates via the frozen /0.020
            # rule while retaining a nondegenerate [a,c] tuning geometry.
            per_bin = np.asarray([
                max(1, int(np.rint(9.0 + 3.0 * np.cos(theta) + 2.0 * np.sin(theta)))),
                max(1, int(np.rint(12.0 - 2.0 * np.cos(theta) + 3.0 * np.sin(theta)))),
                max(1, int(np.rint(15.0 + 4.0 * np.cos(theta) - 2.0 * np.sin(theta)))),
                max(1, int(np.rint(18.0 - 3.0 * np.cos(theta) - 3.0 * np.sin(theta)))),
            ], dtype=np.int64)
        raw = np.tile(per_bin[permutation][None, :], (128, 1))
        views.append(adapter.materialize_source_trial_views(_record(
            trial_id=trial_id, units=4, full_counts=raw, channel_ids=channels,
        ), roster=_roster()))
        pseudo_indices = tuple(
            (index + offset) % len(core.CANONICAL_DIRECTIONS_RAD)
            for offset in offsets
        )
        rows.append(audit.GroupedPseudoAuditRow(
            budget=budget,
            session_id=SESSION,
            trial_id=trial_id,
            true_direction=adapter.AuditOnlyTrueDirection(SESSION, trial_id, theta),
            pseudo_direction_indices=pseudo_indices,
            rejection_reasons=(None,) * core.GROUP_COUNT,
            prefix_digest=hashlib.sha256(f"carrier-prefix-{budget}-{index}".encode()).hexdigest(),
        ))
    return tuple(rows), tuple(views), authority, groups


def _carrier_fixture_with_undefined_theta_rows(
    *,
    channel_permutation: tuple[int, int, int, int, int, int] = (0, 1, 2, 3, 4, 5),
) -> tuple[
    tuple[audit.GroupedPseudoAuditRow, ...],
    tuple[adapter.SourceTrialViews, ...],
    adapter.SourceAuditBudgetAuthority,
    core.ComplementaryGroups,
]:
    """Carrier B8 fixture with two physical-but-undefined theta rows.

    The rows stay in the source channel order and native-rate arrays, while
    only the four authority-valid rows have complementary group assignments.
    This mirrors the strict-27 topology without reading a live session.
    """
    budget = 30
    first30 = tuple(f"undefined-first-{index:02d}" for index in range(30))
    next30 = tuple(f"undefined-next-{index:02d}" for index in range(30))
    authority = adapter.build_source_audit_authorities(
        first30 + next30,
        m4_d_optimal_indices=(0, 8, 16, 24),
    )[budget]
    permutation = np.asarray(channel_permutation, dtype=np.int64)
    assert np.array_equal(np.sort(permutation), np.arange(6, dtype=np.int64))
    base_channels = _channels(6)
    base_valid = np.asarray([True, False, True, True, False, True], dtype=np.bool_)
    base_assignment = np.asarray([0, -1, 1, 2, -1, 3], dtype=np.int64)
    channels = base_channels[permutation]
    groups = core.ComplementaryGroups(
        channel_ids=channels,
        assignment=base_assignment[permutation],
        valid_mask=base_valid[permutation],
    )
    rows: list[audit.GroupedPseudoAuditRow] = []
    views: list[adapter.SourceTrialViews] = []
    for index, trial_id in enumerate(authority.audit_trial_ids):
        theta = core.CANONICAL_DIRECTIONS_RAD[index % len(core.CANONICAL_DIRECTIONS_RAD)]
        # The two undefined-theta rows have valid physical native counts.  They
        # must nevertheless stay out of the directional cosine estimand.
        per_bin = np.asarray([
            max(1, int(np.rint(9.0 + 3.0 * np.cos(theta) + 2.0 * np.sin(theta)))),
            max(1, int(np.rint(21.0 - 2.0 * np.cos(theta) + np.sin(theta)))),
            max(1, int(np.rint(12.0 - 2.0 * np.cos(theta) + 3.0 * np.sin(theta)))),
            max(1, int(np.rint(15.0 + 4.0 * np.cos(theta) - 2.0 * np.sin(theta)))),
            max(1, int(np.rint(25.0 + np.cos(theta) - 2.0 * np.sin(theta)))),
            max(1, int(np.rint(18.0 - 3.0 * np.cos(theta) - 3.0 * np.sin(theta)))),
        ], dtype=np.int64)
        raw = np.tile(per_bin[permutation][None, :], (128, 1))
        views.append(adapter.materialize_source_trial_views(_record(
            trial_id=trial_id, units=6, full_counts=raw, channel_ids=channels,
        ), roster=_roster()))
        pseudo_index = index % len(core.CANONICAL_DIRECTIONS_RAD)
        rows.append(audit.GroupedPseudoAuditRow(
            budget=budget,
            session_id=SESSION,
            trial_id=trial_id,
            true_direction=adapter.AuditOnlyTrueDirection(SESSION, trial_id, theta),
            pseudo_direction_indices=(pseudo_index,) * core.GROUP_COUNT,
            rejection_reasons=(None,) * core.GROUP_COUNT,
            prefix_digest=hashlib.sha256(f"undefined-prefix-{index}".encode()).hexdigest(),
        ))
    return tuple(rows), tuple(views), authority, groups


def _load_frozen_pseudo_label_gate():
    """Test-only reference import; production source-audit code stays local/no-data."""
    frozen_parent = str(ROOT / "sua_exploration")
    sys.path.insert(0, frozen_parent)
    try:
        return importlib.import_module("mc_maze.pseudo_label_carrier_gate")
    finally:
        sys.path.remove(frozen_parent)


def test_exact_stage0_and_source_audit_closure_binding() -> None:
    closure = lifecycle.closure_payload(ROOT)
    assert closure["stage0_workorder_sha256"] == lifecycle.STAGE0_WORKORDER_SHA256
    assert closure["stage0_closure_sha256"] == lifecycle.STAGE0_CLOSURE_SHA256
    assert [row["path"] for row in closure["paths"]] == list(lifecycle.SOURCE_AUDIT_CLOSURE_PATHS)
    assert "sua_exploration/mc_maze/pseudo_label_carrier_gate.py" in lifecycle.SOURCE_AUDIT_CLOSURE_PATHS
    assert len(str(closure["closure_sha256"])) == 64
    with pytest.raises(lifecycle.SourceAuditLifecycleError, match="Stage-0 closure"):
        # A complete but altered Stage-0 row order must not be treated as the
        # accepted authority.
        lifecycle._require(hashlib.sha256(b"altered").hexdigest() == lifecycle.STAGE0_CLOSURE_SHA256, "accepted CDM-D Stage-0 closure SHA drift")


def test_carrier_gate_reference_is_an_explicit_fail_closed_closure_leaf(monkeypatch) -> None:
    reference = "sua_exploration/mc_maze/pseudo_label_carrier_gate.py"
    altered = tuple(
        "sua_exploration/mc_maze/missing_pseudo_label_carrier_gate.py" if item == reference else item
        for item in lifecycle.SOURCE_AUDIT_CLOSURE_PATHS
    )
    monkeypatch.setattr(lifecycle, "SOURCE_AUDIT_CLOSURE_PATHS", altered)
    with pytest.raises(lifecycle.SourceAuditLifecycleError, match="not a regular file"):
        lifecycle.closure_payload(ROOT)


def test_strict_source_roster_and_non_source_surface_rejection() -> None:
    roster = _roster()
    assert roster.payload()["count"] == 27
    assert len(roster.digest) == 64
    for surface in (adapter.Surface.WITHIN, adapter.Surface.EXTERNAL, adapter.Surface.FORMAL, adapter.Surface.TARGET):
        with pytest.raises(adapter.SourceAdapterError, match="non-source"):
            adapter.require_source_only_surface(surface)
    with pytest.raises(adapter.SourceAdapterError, match="27"):
        adapter.StrictSourceRoster(tuple(f"sub-C-{index}" for index in range(26)))
    with pytest.raises(adapter.SourceAdapterError, match="sub-C"):
        adapter.StrictSourceRoster(tuple(f"sub-M-{index}" for index in range(27)))
    absent_roster = adapter.StrictSourceRoster(tuple(f"sub-C_ses-CO-other-{index:02d}" for index in range(27)))
    with pytest.raises(adapter.SourceAdapterError, match="strict-27"):
        adapter.materialize_source_trial_views(_record(), roster=absent_roster)


def test_exact_m4_m10_m30_deployment_support_and_distinct_audit_identities() -> None:
    trial_ids = tuple(f"trial-{index:02d}" for index in range(30))
    authorities = adapter.build_budget_authorities(trial_ids, m4_d_optimal_indices=(1, 5, 17, 28))
    assert authorities[4].support_indices == (1, 5, 17, 28)
    assert authorities[4].pseudo_indices == tuple(index for index in range(30) if index not in {1, 5, 17, 28})
    assert authorities[10].support_indices == tuple(range(10))
    assert authorities[10].pseudo_indices == tuple(range(10, 30))
    assert authorities[30].support_indices == tuple(range(30))
    assert authorities[30].pseudo_indices == ()
    next30 = tuple(f"post-support-{index:02d}" for index in range(30))
    audit_authorities = adapter.build_source_audit_authorities(
        trial_ids + next30,
        m4_d_optimal_indices=(1, 5, 17, 28),
    )
    assert audit_authorities[4].audit_trial_ids == authorities[4].pseudo_trial_ids
    assert audit_authorities[10].audit_trial_ids == trial_ids[10:30]
    assert audit_authorities[30].audit_trial_ids == next30
    assert audit_authorities[30].audit_enters_deployment_memory is False
    assert adapter.require_m30_post_support_audit_only(audit_authorities[30]) == next30
    assert not (set(next30) & set(audit_authorities[30].deployment_memory_trial_ids))
    with pytest.raises(adapter.SourceAdapterError, match="first-60"):
        adapter.build_source_audit_authorities(
            trial_ids + trial_ids,
            m4_d_optimal_indices=(1, 5, 17, 28),
        )


def test_support_authority_must_be_sealed_before_first_pseudo_trial() -> None:
    trial_ids = tuple(f"trial-{index:02d}" for index in range(30))
    authority = adapter.build_budget_authorities(trial_ids, m4_d_optimal_indices=(1, 5, 17, 28))[4]
    assert adapter.require_sealed_before_pseudo(authority) == authority.pseudo_trial_ids
    forged = object.__new__(adapter.SealedBudgetAuthority)
    object.__setattr__(forged, "budget", 4)
    object.__setattr__(forged, "first30_trial_ids", trial_ids)
    object.__setattr__(forged, "support_indices", (1, 5, 17, 28))
    object.__setattr__(forged, "pseudo_indices", authority.pseudo_indices)
    object.__setattr__(forged, "support_sealed", False)
    with pytest.raises(adapter.SourceAdapterError, match="sealed"):
        adapter.require_sealed_before_pseudo(forged)


def test_m4_m10_m30_first30_b3s_multiset_and_mean_parity() -> None:
    trial_ids = tuple(f"trial-{index:02d}" for index in range(30))
    views = tuple(_views(trial_id=trial_id) for trial_id in trial_ids)
    authorities = adapter.build_budget_authorities(trial_ids, m4_d_optimal_indices=(1, 5, 17, 28))
    for budget in (4, 10, 30):
        evidence = adapter.first30_b3s_parity(authorities[budget], views)
        assert evidence["tolerance_mean_parity"] is True
        assert len(evidence["support_then_pseudo_trial_ids"]) == 30
        assert evidence["multiset_sha256"]


def test_m30_current_pseudo_audit_rows_are_post_support_and_cannot_enter_memory() -> None:
    first30 = tuple(f"first-{index:02d}" for index in range(30))
    next30 = tuple(f"next-{index:02d}" for index in range(30))
    authority = adapter.build_source_audit_authorities(
        first30 + next30,
        m4_d_optimal_indices=(0, 7, 15, 22),
    )[30]
    payload = authority.payload()
    assert payload["audit_population"] == "post_first30_next30_rewarded_source_trials"
    assert payload["audit_enters_deployment_memory"] is False
    assert payload["m30_current_pseudo_trials_disjoint_from_labelled_support"] is True
    assert tuple(payload["deployment_memory_trial_ids"]) == first30
    assert tuple(payload["audit_trial_ids"]) == next30
    assert tuple(payload["audit_source_chronology_positions"]) == tuple(range(30, 60))


def test_independent_reconstruction_of_b3s_native_and_velocity_validity_views() -> None:
    record = _record(availability=np.asarray([True, True, False, True, True, True], dtype=np.bool_))
    views = adapter.materialize_source_trial_views(record, roster=_roster())
    assert views.b3s_activity.activity.shape == (100, 8)
    assert views.carrier_counts.counts.shape == (90, 8)
    assert views.velocity_validity.valid_mask.tolist() == [True, True, False, True, True, True]
    assert views.b3s_activity.trial_id == views.carrier_counts.trial_id == views.velocity_validity.trial_id
    assert views.carrier_counts.channel_order_sha256 == views.b3s_activity.channel_order_sha256
    assert "true_direction" not in views.payload()


def test_native_counts_cannot_be_rewrapped_interpolated_or_fractional() -> None:
    wrong_rebuild = _record(rebuilder=lambda record: record.accepted_calibration_b3s_activity + 0.5)
    with pytest.raises(adapter.SourceAdapterError, match="differs"):
        adapter.materialize_source_trial_views(wrong_rebuild, roster=_roster())
    fractional = np.tile(np.linspace(1.0, 8.0, 8), (128, 1))
    fractional[0, 0] = 1.5
    with pytest.raises(adapter.SourceAdapterError, match="integer-valued"):
        _record(full_counts=fractional)
    assert list(inspect.signature(adapter.materialize_source_trial_views).parameters) == ["record", "roster", "surface"]


def test_held_record_raw_identity_binds_channel_trial_native_b3s_and_endpoint_evidence() -> None:
    record = _record()
    with pytest.raises(adapter.SourceAdapterError, match="raw session/trial/channel"):
        replace(record, neural_endpoint_available=np.zeros(record.neural_endpoint_available.shape, dtype=np.bool_))
    with pytest.raises(adapter.SourceAdapterError, match="channel-order"):
        replace(record, channel_ids=np.arange(200, 208, dtype=np.int64))


def test_native_rate_matches_full_raw_rewarded_slice_over_20ms() -> None:
    record = _record()
    views = adapter.materialize_source_trial_views(record, roster=_roster())
    expected = record.native_rewarded_slice().mean(axis=0, dtype=np.float64) / 0.020
    observed = core.scalar_rates_from_native_rewarded_counts(views.carrier_counts)
    assert np.array_equal(observed, expected)
    assert views.carrier_counts.rewarded_interval_stop_bin - views.carrier_counts.rewarded_interval_start_bin == 90


def test_standardized_velocity_is_restored_to_physical_before_direction_integration() -> None:
    standardized = np.repeat(np.asarray([[1.0, 0.0]], dtype=np.float64), 6, axis=0)
    physical_velocity = physical.restore_physical_velocity(
        standardized,
        behavior_mean=np.asarray([0.0, 4.0], dtype=np.float64),
        behavior_std=np.asarray([2.0, 3.0], dtype=np.float64),
    )
    assert not np.array_equal(physical_velocity, standardized)
    result = core.pseudo_direction_from_velocity(
        physical_velocity,
        np.ones(6, dtype=np.bool_),
        config=core.CDMDConfig(support_budget_m=4, dt=1.0, minimum_movement_bins=2,
                               minimum_displacement=0.01, minimum_mean_speed=0.01,
                               active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL),
    )
    assert result.accepted
    assert result.theta_raw_rad != pytest.approx(0.0)


def test_velocity_validity_uses_only_w50_endpoints_bounds_and_neural_availability() -> None:
    endpoints = np.arange(55, 61, dtype=np.int64)
    availability = np.asarray([True, True, False, True, True, False], dtype=np.bool_)
    record = _record(endpoints=endpoints, availability=availability)
    views = adapter.materialize_source_trial_views(record, roster=_roster())
    # Endpoints 55--58 have 50-bin starts below rewarded start 10; endpoint
    # 59 is valid, while endpoint 60 is invalid solely by neural availability.
    assert views.velocity_validity.valid_mask.tolist() == [False, False, False, False, True, False]
    assert views.velocity_validity.source == "neural_window_availability_and_trial_bounds"
    assert views.velocity_validity.target_behavior_used is False
    assert "true" not in inspect.signature(adapter.materialize_source_trial_views).parameters


def test_group_slicing_physically_removes_same_units_from_neural_b3s_and_t4() -> None:
    neural = np.arange(6 * 50 * 8, dtype=np.float64).reshape(6, 50, 8)
    b3s = np.arange(30 * 100 * 8, dtype=np.float64).reshape(30, 100, 8)
    t4 = np.arange(8 * 4, dtype=np.float64).reshape(8, 4)
    held = np.asarray([False, True, False, True, False, False, True, False], dtype=np.bool_)
    sliced = physical.physically_slice_held_units(neural, b3s, t4, held)
    active = np.flatnonzero(~held)
    assert np.array_equal(sliced.retained_channel_indices, active)
    assert np.array_equal(sliced.neural_windows, neural[:, :, active])
    assert np.array_equal(sliced.b3s_activity_stack, b3s[:, :, active])
    assert np.array_equal(sliced.normalized_t4, t4[active])
    assert sliced.neural_windows.shape[2] == 5


def test_cpu_synthetic_set_reader_variable_n_repeat_state_rng_and_no_dropout() -> None:
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    model = physical.CPUSyntheticCellDSetReader()
    captured: list[physical.PhysicalForwardEvidence] = []
    validity = core.VelocityValidityEvidence(
        valid_mask=np.ones(6, dtype=np.bool_), session_id=SESSION, trial_id="physical",
        prediction_interval_start_bin=59, prediction_interval_stop_bin=65,
    )
    for units in (5, 8):
        neural = np.arange(6 * 50 * units, dtype=np.float64).reshape(6, 50, units)
        b3s = np.arange(30 * 100 * units, dtype=np.float64).reshape(30, 100, units)
        t4 = np.linspace(-1.0, 1.0, units * 4, dtype=np.float64).reshape(units, 4)
        held = np.zeros(units, dtype=np.bool_)
        held[0] = True
        first = physical.forward_held_completed_trial(
            model, neural_windows=neural, b3s_activity_stack=b3s, normalized_t4=t4,
            held_unit_mask=held, behavior_normalizer=_behavior_normalizer(),
            velocity_validity=validity, no_grad_context=model.no_grad,
            rng_state_snapshot=_rng_snapshot,
            dropout_counter=lambda: model.dynamic_dropout_calls,
            evidence_sink=captured.append,
        )
        assert first.velocity.shape == (6, 2) and np.isfinite(first.velocity).all()
    assert model.dynamic_dropout_calls == 0
    assert model.no_grad_calls == 2
    assert model.eval_calls == 2
    assert len(captured) == 2
    assert all(item.payload()["rng_unchanged"] is True for item in captured)
    assert all(item.payload()["dynamic_dropout_unchanged"] is True for item in captured)
    assert random.getstate() == random_before
    after = np.random.get_state()
    assert after[0] == numpy_before[0] and after[2:] == numpy_before[2:] and np.array_equal(after[1], numpy_before[1])


def test_physical_forward_rejects_hidden_rng_consumption_even_when_outputs_are_deterministic() -> None:
    class DeterministicRNGConsumer(physical.CPUSyntheticCellDSetReader):
        def __call__(self, *args, **kwargs):
            random.random()
            np.random.random()
            return super().__call__(*args, **kwargs)

    model = DeterministicRNGConsumer()
    validity = core.VelocityValidityEvidence(
        valid_mask=np.ones(6, dtype=np.bool_), session_id=SESSION, trial_id="rng-consumer",
        prediction_interval_start_bin=59, prediction_interval_stop_bin=65,
    )
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    try:
        with pytest.raises(physical.PhysicalSeamError, match="RNG state"):
            physical.forward_held_completed_trial(
                model,
                neural_windows=np.zeros((6, 50, 5), dtype=np.float64),
                b3s_activity_stack=np.zeros((30, 100, 5), dtype=np.float64),
                normalized_t4=np.zeros((5, 4), dtype=np.float64),
                held_unit_mask=np.asarray([True, False, False, False, False], dtype=np.bool_),
                behavior_normalizer=_behavior_normalizer(),
                velocity_validity=validity,
                no_grad_context=model.no_grad,
                rng_state_snapshot=_rng_snapshot,
                dropout_counter=lambda: model.dynamic_dropout_calls,
            )
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)


def test_physical_forward_requires_explicit_dropout_and_complete_rng_proofs() -> None:
    model = physical.CPUSyntheticCellDSetReader()
    validity = core.VelocityValidityEvidence(
        valid_mask=np.ones(6, dtype=np.bool_), session_id=SESSION, trial_id="missing-proof",
        prediction_interval_start_bin=59, prediction_interval_stop_bin=65,
    )
    kwargs = dict(
        neural_windows=np.zeros((6, 50, 5), dtype=np.float64),
        b3s_activity_stack=np.zeros((30, 100, 5), dtype=np.float64),
        normalized_t4=np.zeros((5, 4), dtype=np.float64),
        held_unit_mask=np.asarray([True, False, False, False, False], dtype=np.bool_),
        behavior_normalizer=_behavior_normalizer(),
        velocity_validity=validity,
        no_grad_context=model.no_grad,
    )
    with pytest.raises(physical.PhysicalSeamError, match="dropout counter"):
        physical.forward_held_completed_trial(model, rng_state_snapshot=_rng_snapshot, dropout_counter=None, **kwargs)
    with pytest.raises(physical.PhysicalSeamError, match="RNG proof"):
        physical.forward_held_completed_trial(
            model,
            rng_state_snapshot=lambda: object(),
            dropout_counter=lambda: model.dynamic_dropout_calls,
            **kwargs,
        )


def test_true_source_direction_cannot_enter_stage0_update_constructor() -> None:
    parameters = inspect.signature(core.CausalDualMemory.observe_completed_trial).parameters
    assert "true_direction" not in parameters and "source_label" not in parameters
    assert "true_direction" not in inspect.signature(adapter.materialize_source_trial_views).parameters
    truth = adapter.AuditOnlyTrueDirection(SESSION, "trial-0", 0.0)
    assert truth.trial_id == _views().b3s_activity.trial_id


def test_trial_direction_metrics_are_descriptive_only() -> None:
    rows = tuple(_true(index) for index in range(8))
    first = audit.descriptive_trial_direction_aggregate(rows)
    second = audit.descriptive_trial_direction_aggregate(rows)
    assert first["shuffled_trial_identities"] == second["shuffled_trial_identities"]
    assert first["trial_cosine_median"] == pytest.approx(1.0)
    assert first["non_governing"] is True
    assert "pass" not in first and "thresholds" not in first
    diagnostics = audit.summarize_trial_diagnostics(rows)
    assert diagnostics["accepted"] == 8 and diagnostics["rejected"] == 0


def test_carrier_level_b8_matches_frozen_gate_and_raw_trial_cosine_cannot_license_it() -> None:
    rows, views, authority, groups = _carrier_fixture(budget=30)
    inputs = audit.carrier_b8_inputs_from_finalized_rows(
        rows, views, audit_authority=authority, groups=groups,
    )
    observed = audit.carrier_b8_aggregate(inputs)
    frozen = _load_frozen_pseudo_label_gate()
    expected = frozen.evaluate_pseudo_label_gate(
        inputs.native_trial_rates(),
        inputs.directions.true_direction_indices,
        inputs.directions.pseudo_direction_indices[:, 0],
        budget_m=len(rows),
        session_name=SESSION,
        shuffle_seed=42,
    )
    expected_payload = expected.as_dict()
    # Identical four pseudo columns must reduce *exactly* to the frozen
    # single-label B8 arithmetic, not merely within a tolerance.
    assert observed["median_cosine_correct"] == expected_payload["median_cosine_correct"]
    assert observed["median_cosine_shuffled"] == expected_payload["median_cosine_shuffled"]
    assert observed["correct_minus_shuffle"] == expected_payload["correct_minus_shuffle"]
    assert observed["fraction_ge_040_correct"] == expected_payload["fraction_ge_040_correct"]
    assert observed["gates"] == expected_payload["gates"]
    assert observed["shuffle_shift"] == expected_payload["shuffle_shift"]
    assert observed["frozen_reference"] == audit.FROZEN_PSEUDO_LABEL_GATE_SEMANTICS

    # Identical raw trial directions do not imply an identifiable carrier.  A
    # constant native-rate table has raw cosine one but undefined [a,c] norms,
    # so it must fail the actual governing B8 screen.
    constant_rows, constant_views, constant_authority, constant_groups = _carrier_fixture(
        budget=30, constant_native_rates=True,
    )
    descriptive = audit.descriptive_group_trial_direction_aggregate(constant_rows)
    constant_evidence = audit.carrier_b8_aggregate(
        audit.carrier_b8_inputs_from_finalized_rows(
            constant_rows, constant_views, audit_authority=constant_authority, groups=constant_groups,
        )
    )
    assert descriptive["groups"][0]["trial_cosine_median"] == pytest.approx(1.0)
    assert constant_evidence["median_cosine_correct"] is None
    assert constant_evidence["pass"] is False


def test_grouped_b8_uses_one_common_shuffle_and_is_joint_unit_group_permutation_invariant() -> None:
    rows, views, authority, groups = _carrier_fixture(budget=30, pseudo_shift_indices=(0, 1, 3, 5))
    observed = audit.carrier_b8_aggregate(
        audit.carrier_b8_inputs_from_finalized_rows(rows, views, audit_authority=authority, groups=groups)
    )
    assert observed["status"] == "COMPLETE_FIXED_POOL"
    assert len({item["common_shuffle_order_sha256"] for item in observed["group_evidence"]}) == 1
    assert observed["common_shuffle_order_sha256"] == observed["group_evidence"][0]["common_shuffle_order_sha256"]
    assert len({item["pseudo_direction_column_sha256"] for item in observed["group_evidence"]}) == 4

    permuted_rows, permuted_views, permuted_authority, permuted_groups = _carrier_fixture(
        budget=30,
        pseudo_shift_indices=(0, 1, 3, 5),
        channel_permutation=(2, 0, 3, 1),
    )
    permuted = audit.carrier_b8_aggregate(
        audit.carrier_b8_inputs_from_finalized_rows(
            permuted_rows, permuted_views,
            audit_authority=permuted_authority, groups=permuted_groups,
        )
    )
    assert groups.channel_to_group() == permuted_groups.channel_to_group()
    for key in (
        "median_cosine_correct", "median_cosine_shuffled", "correct_minus_shuffle",
        "fraction_ge_040_correct", "defined_units_correct", "defined_units_shuffled", "gates", "pass",
    ):
        assert permuted[key] == observed[key]


def test_grouped_b8_preserves_undefined_theta_rows_but_excludes_them_from_cosines() -> None:
    rows, views, authority, groups = _carrier_fixture_with_undefined_theta_rows()
    inputs = audit.carrier_b8_inputs_from_finalized_rows(
        rows, views, audit_authority=authority, groups=groups,
    )
    topology = inputs.unit_topology_payload()
    assert topology == {
        "total_unit_count": 6,
        "valid_unit_count": 4,
        "invalid_unit_count": 2,
        "valid_mask_sha256": core.array_digest(groups.valid_mask),
        "complementary_assignment_sha256": core.array_digest(groups.assignment),
        "invalid_assignment_is_minus_one": True,
    }
    assert np.array_equal(groups.assignment[~groups.valid_mask], np.asarray([-1, -1], dtype=np.int64))
    assert set(groups.channel_to_group()) == set(groups.channel_ids[groups.valid_mask].tolist())

    observed = audit.carrier_b8_aggregate(inputs)
    frozen = _load_frozen_pseudo_label_gate()
    rates = inputs.native_trial_rates()
    valid = groups.valid_mask
    expected = frozen.evaluate_pseudo_label_gate(
        rates[:, valid],
        inputs.directions.true_direction_indices,
        inputs.directions.pseudo_direction_indices[:, 0],
        budget_m=len(rows), session_name=SESSION, shuffle_seed=42,
    ).as_dict()
    for key in (
        "median_cosine_correct", "median_cosine_shuffled", "correct_minus_shuffle",
        "fraction_ge_040_correct", "gates",
    ):
        assert observed[key] == expected[key]
    assert observed["pass"] is observed["gates"]["all_predeclared_gates"]
    assert observed["unit_topology"] == topology
    assert observed["valid_rows_assigned_exactly_once"] is True
    assert observed["invalid_rows_unassigned"] is True
    assert observed["invalid_correct_cosines_all_nan"] is True
    assert observed["invalid_shuffled_cosines_all_nan"] is True
    # Defined counts are finite *valid* [a,c] cosines, never a count of the
    # six physical rows.  Reconstruct the full channel-order vectors to prove
    # the invalid rows are exact NaNs in the bound digest.
    true_carriers = audit._frozen_gate_fit_carriers(rates, inputs.directions.true_direction_indices)
    pseudo_carriers = audit._frozen_gate_fit_carriers(rates, inputs.directions.pseudo_direction_indices[:, 0])
    order, _ = audit._frozen_gate_shuffle_order(rates.shape[0], session_id=SESSION, seed=42)
    shuffled_carriers = audit._frozen_gate_fit_carriers(
        rates, inputs.directions.pseudo_direction_indices[order, 0],
    )
    correct = audit._frozen_gate_ac_cosines(true_carriers[:, :2], pseudo_carriers[:, :2])
    shuffled = audit._frozen_gate_ac_cosines(true_carriers[:, :2], shuffled_carriers[:, :2])
    expected_correct = np.full(groups.units, np.nan, dtype=np.float64)
    expected_shuffled = np.full(groups.units, np.nan, dtype=np.float64)
    expected_correct[valid] = correct[valid]
    expected_shuffled[valid] = shuffled[valid]
    assert observed["correct_cosines_channel_order_sha256"] == core.array_digest(expected_correct)
    assert observed["shuffled_cosines_channel_order_sha256"] == core.array_digest(expected_shuffled)
    assert observed["defined_units_correct"] == int(np.isfinite(expected_correct[valid]).sum())
    assert observed["defined_units_shuffled"] == int(np.isfinite(expected_shuffled[valid]).sum())

    # Permuting every physical channel row together with its group and validity
    # authority leaves the valid-unit B8 estimand unchanged.
    perm_rows, perm_views, perm_authority, perm_groups = _carrier_fixture_with_undefined_theta_rows(
        channel_permutation=(5, 2, 4, 0, 3, 1),
    )
    permuted = audit.carrier_b8_aggregate(audit.carrier_b8_inputs_from_finalized_rows(
        perm_rows, perm_views, audit_authority=perm_authority, groups=perm_groups,
    ))
    assert groups.channel_to_group() == perm_groups.channel_to_group()
    for key in (
        "median_cosine_correct", "median_cosine_shuffled", "correct_minus_shuffle",
        "fraction_ge_040_correct", "defined_units_correct", "defined_units_shuffled", "gates", "pass",
    ):
        assert permuted[key] == observed[key]

    # An undefined physical row may not be smuggled into a complementary group.
    with pytest.raises(core.CDMDStage0Error, match="invalid units must carry group -1"):
        core.ComplementaryGroups(
            channel_ids=groups.channel_ids,
            assignment=np.asarray([0, 0, 1, 2, -1, 3], dtype=np.int64),
            valid_mask=groups.valid_mask,
        )


def test_grouped_b8_rejects_broadcast_substitution_and_preserves_fixed_pool_rejections() -> None:
    rows, views, authority, groups = _carrier_fixture(budget=30, pseudo_shift_indices=(0, 4, 0, 4))
    grouped_inputs = audit.carrier_b8_inputs_from_finalized_rows(
        rows, views, audit_authority=authority, groups=groups,
    )
    grouped = audit.carrier_b8_aggregate(grouped_inputs)
    frozen = _load_frozen_pseudo_label_gate().evaluate_pseudo_label_gate(
        grouped_inputs.native_trial_rates(),
        grouped_inputs.directions.true_direction_indices,
        grouped_inputs.directions.pseudo_direction_indices[:, 0],
        budget_m=len(rows), session_name=SESSION, shuffle_seed=42,
    ).as_dict()
    # A one-column broadcast would falsely look perfect because column zero is
    # correct.  The real CDM-D consumer sees two opposite group columns.
    assert frozen["gates"]["all_predeclared_gates"] is True
    assert grouped["median_cosine_correct"] != frozen["median_cosine_correct"]
    assert grouped["pass"] is False
    with pytest.raises(audit.SourceAuditError, match="invalid integer shape"):
        audit.FinalizedGroupedCarrierDirectionIndices(
            30, SESSION, authority.audit_trial_ids,
            grouped_inputs.directions.true_direction_indices,
            grouped_inputs.directions.pseudo_direction_indices[:, 0],
            tuple((None,) * core.GROUP_COUNT for _ in authority.audit_trial_ids),
        )

    rejected_first = replace(
        rows[0],
        pseudo_direction_indices=(rows[0].pseudo_direction_indices[0], None,
                                  rows[0].pseudo_direction_indices[2], rows[0].pseudo_direction_indices[3]),
        rejection_reasons=(None, core.UpdateRejectionReason.LOW_MEAN_SPEED, None, None),
    )
    rejected_rows = (rejected_first,) + rows[1:]
    descriptive = audit.descriptive_group_trial_direction_aggregate(rejected_rows)
    assert descriptive["non_governing"] is True
    assert "pass" not in descriptive
    rejected_inputs = audit.carrier_b8_inputs_from_finalized_rows(
        rejected_rows, views, audit_authority=authority, groups=groups,
    )
    rejected = audit.carrier_b8_aggregate(rejected_inputs)
    assert rejected["status"] == "STOP_FIXED_POOL_REJECTION"
    assert rejected["rejected_group_count"] == 1
    assert rejected["rejected_trial_count"] == 1
    assert rejected["rejection_counts"] == {core.UpdateRejectionReason.LOW_MEAN_SPEED.value: 1}
    assert rejected["pass"] is False
    decision = audit.fail_fast_b8_decision({30: rejected_inputs, 10: grouped_inputs, 4: grouped_inputs})
    assert decision["verdict"] == audit.SourceAuditVerdict.STOP.value


def test_fail_fast_m30_to_m10_to_m4_uses_carrier_b8_not_raw_rows() -> None:
    bad_rows, bad_views, bad_authority, bad_groups = _carrier_fixture(budget=30, pseudo_shift_indices=4)
    good10_rows, good10_views, good10_authority, good10_groups = _carrier_fixture(budget=10)
    good4_rows, good4_views, good4_authority, good4_groups = _carrier_fixture(budget=4)
    inputs = {
        30: audit.carrier_b8_inputs_from_finalized_rows(
            bad_rows, bad_views, audit_authority=bad_authority, groups=bad_groups,
        ),
        10: audit.carrier_b8_inputs_from_finalized_rows(
            good10_rows, good10_views, audit_authority=good10_authority, groups=good10_groups,
        ),
        4: audit.carrier_b8_inputs_from_finalized_rows(
            good4_rows, good4_views, audit_authority=good4_authority, groups=good4_groups,
        ),
    }
    decision = audit.fail_fast_b8_decision(inputs)
    assert decision["verdict"] == audit.SourceAuditVerdict.STOP.value
    assert decision["by_budget"]["30"]["verdict"] == audit.SourceAuditVerdict.STOP.value
    assert decision["by_budget"]["10"]["verdict"] == audit.SourceAuditVerdict.NOT_REACHED.value
    assert decision["by_budget"]["4"]["verdict"] == audit.SourceAuditVerdict.NOT_REACHED.value
    with pytest.raises(audit.SourceAuditError, match="carrier B8 inputs"):
        audit.fail_fast_b8_decision({30: bad_rows, 10: good10_rows, 4: good4_rows})


class _FakeWriter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def attempt(self, payload):
        self.events.append(("attempt", dict(payload)))
        return "a" * 64

    def failure(self, payload):
        self.events.append(("failure", dict(payload)))
        return "b" * 64


def test_future_lifecycle_attempt_precedes_data_and_failure_is_at_most_typed_failure() -> None:
    writer = _FakeWriter()
    observed: list[str] = []

    def source_action():
        observed.append(writer.events[-1][0])
        raise RuntimeError("synthetic source open failure")

    result = lifecycle.run_injected_future_lifecycle(writer=writer, source_action=source_action, execution_capability=object())
    assert observed == ["attempt"]
    assert [event[0] for event in writer.events] == ["attempt", "failure"]
    assert writer.events[0][1]["source_opened"] is False
    assert writer.events[1][1]["source_opened"] is True
    assert result.failure_sha256 == "b" * 64 and result.terminal_published is False
    with pytest.raises(lifecycle.SourceAuditLifecycleError, match="capability"):
        lifecycle.run_injected_future_lifecycle(writer=_FakeWriter(), source_action=lambda: None, execution_capability=None)


def test_static_cli_imports_no_torch_writes_nothing_and_executes_nothing(tmp_path: Path) -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_audit.py"
    code = (
        "import importlib.util,sys; "
        f"spec=importlib.util.spec_from_file_location('cdm_source_cli',{str(script)!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "module.main(['--dry-run']); print('TORCH_IMPORTED='+str('torch' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        text=True, capture_output=True, check=True,
    )
    assert "TORCH_IMPORTED=False" in completed.stdout
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(SystemExit):
        # argparse's fail-closed public execution route never reaches data.
        import importlib.util
        spec = importlib.util.spec_from_file_location("cdm_source_cli_local", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.main(["--execute"])
