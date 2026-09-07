from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
H1_SRC = ROOT / "tfpd_exploration" / "h1_series_20260830" / "src"
SPINT_MAIN = ROOT / "SPINT-main"
for value in (str(H1_SRC), str(SPINT_MAIN)):
    if value not in sys.path:
        sys.path.insert(0, value)

from h1_causal_activity_completion_v1.core import (
    CandidateSchedule,
    build_chunk_schedule,
    pool_cardinality,
    resample_activity_member,
)
from h1_causal_activity_completion_v1.plan import ARM_ORDER, DATE_ORDER, decide
from h1_causal_activity_completion_v1.stage1 import (
    ARTIFACT_RELATIVE,
    C1_AUTHORITIES,
    C1M3StrictTargetDataset,
    _load_model,
    _load_plan,
    _selected_fixed_chunk_readout,
)


def _neural(length: int, *, units: int = 176) -> np.ndarray:
    time = np.linspace(0.0, 1.0, length, dtype=np.float32)[:, None]
    channel = np.linspace(0.1, 1.0, units, dtype=np.float32)[None, :]
    return np.ascontiguousarray(time * channel, dtype=np.float32)


def test_resample_is_finite_deterministic_and_exact_shape() -> None:
    source = _neural(768)
    first = resample_activity_member(source)
    second = resample_activity_member(source.copy())
    assert first.shape == (1024, 176)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


def test_fixed_schedule_decodes_before_commit_and_freezes_at_m7() -> None:
    schedule = build_chunk_schedule(
        _neural(4 * 32), origin=0, chunk_length=32, initial_members=4, energy_gated=False,
    )
    assert len(schedule.members) == 3
    assert schedule.completion_end_exclusive == (32, 64, 96)
    assert pool_cardinality(schedule, initial_members=4, endpoint_inclusive=31) == 4
    assert pool_cardinality(schedule, initial_members=4, endpoint_inclusive=32) == 5
    assert pool_cardinality(schedule, initial_members=4, endpoint_inclusive=64) == 6
    assert pool_cardinality(schedule, initial_members=4, endpoint_inclusive=96) == 7
    assert len(schedule.candidate_end_exclusive) == 3


def test_m3_schedule_accepts_exactly_four_members_then_freezes() -> None:
    schedule = build_chunk_schedule(
        _neural(6 * 16), origin=0, chunk_length=16, initial_members=3, energy_gated=False,
    )
    assert len(schedule.members) == 4
    assert schedule.completion_end_exclusive == (16, 32, 48, 64)
    assert pool_cardinality(schedule, initial_members=3, endpoint_inclusive=64) == 7


def test_energy_gate_uses_previous_candidate_median_and_rejections_enter_history() -> None:
    chunks = [
        np.full((8, 176), 2.0, np.float32),
        np.full((8, 176), 1.0, np.float32),
        np.full((8, 176), 3.0, np.float32),
        np.full((8, 176), 1.5, np.float32),
        np.full((8, 176), 4.0, np.float32),
    ]
    schedule = build_chunk_schedule(
        np.concatenate(chunks), origin=0, chunk_length=8, initial_members=4, energy_gated=True,
    )
    assert schedule.accepted == (True, False, True, False, True)
    assert schedule.candidate_energies == pytest.approx((2.0, 1.0, 3.0, 1.5, 4.0))
    assert schedule.completion_end_exclusive == (8, 24, 40)


def test_candidate_schedule_state_is_explicit_and_slot_local() -> None:
    left = build_chunk_schedule(_neural(64), origin=0, chunk_length=16, initial_members=4, energy_gated=False)
    right = build_chunk_schedule(_neural(64) * 2.0, origin=0, chunk_length=16, initial_members=4, energy_gated=True)
    assert left is not right
    assert left.members[0] is not right.members[0]
    assert left.candidate_energies != right.candidate_energies


def test_decision_keeps_content_and_detector_gates_separate() -> None:
    rows = []
    for index, date in enumerate(DATE_ORDER):
        base = 0.2 + index * 0.01
        values = (base, base + 0.02, base + 0.012, base + 0.011)
        rows.append({"outer_date": date, "results": [
            {"arm": arm, "equal_recording_mean_r2": value}
            for arm, value in zip(ARM_ORDER, values, strict=True)
        ]})
    result = decide(rows)
    assert result["gate_1_content"]["pass"] is True
    assert result["gate_2_deployable_recovery"]["pass"] is True
    assert result["verdict"] == "PASS_H1_CAC_STAGE0_FOR_C1_SPECIFIC_M3_TO_M7"


def test_decision_stops_content_before_detector_interpretation() -> None:
    rows = [{"outer_date": date, "results": [
        {"arm": arm, "equal_recording_mean_r2": 0.3}
        for arm in ARM_ORDER
    ]} for date in DATE_ORDER]
    result = decide(rows)
    assert result["gate_1_content"]["pass"] is False
    assert result["gate_2_deployable_recovery"]["pass"] is False
    assert result["verdict"] == "STOP_H1_ACTIVITY_COMPLETION_CONTENT_DID_NOT_TRANSFER"


def test_bad_activity_geometry_fails_closed() -> None:
    with pytest.raises(Exception):
        resample_activity_member(np.zeros((3, 176), np.float32))
    with pytest.raises(Exception):
        build_chunk_schedule(np.zeros((100, 175), np.float32), origin=0, chunk_length=32,
                             initial_members=4, energy_gated=False)


def test_exact_uploaded_c1_lodo_artifacts_strict_load_without_cuda() -> None:
    import torch

    assert tuple(C1_AUTHORITIES) == DATE_ORDER
    artifact = ROOT / ARTIFACT_RELATIVE
    for date in DATE_ORDER:
        directory = artifact / date
        plan, s_src, plan_receipt = _load_plan(directory, C1_AUTHORITIES[date], date)
        model, state, model_receipt = _load_model(directory, C1_AUTHORITIES[date], date, "cpu")
        assert plan.q in (8, 12, 16)
        assert plan.ridge_lambda == 10.0
        assert s_src > 0.0
        assert plan_receipt["plan_npz_sha256"] == C1_AUTHORITIES[date].plan_npz_sha256
        assert state == C1_AUTHORITIES[date].terminal_state_sha256
        assert model_receipt["strict_load_missing_keys"] == []
        assert model_receipt["strict_load_unexpected_keys"] == []
        assert sum(parameter.numel() for parameter in model.parameters()) == 10_947_836
    assert torch.cuda.is_initialized() is False


def test_c1_m3_surface_uses_deployment_carrier_without_padding(monkeypatch) -> None:
    import src.data.h1_m4_eb_pilot as pilot

    calls: list[tuple[float, ...]] = []

    def deployment_carrier(record, plan, values):
        del record, plan
        calls.append(tuple(values))
        return {"carrier": np.zeros((176, 4), np.float64)}

    monkeypatch.setattr(pilot, "fit_deployment_carrier", deployment_carrier)
    monkeypatch.setattr(
        pilot,
        "interpolate_trial_identity",
        lambda record, value: np.full((1024, 176), value, np.float32),
    )
    record = SimpleNamespace(
        session_name="synthetic",
        trial_values=(1.0, 2.0, 3.0, 4.0),
        eval_mask=np.ones(705, dtype=bool),
        trial_num=np.full(705, 4.0, dtype=np.float64),
        neural=np.zeros((705, 176), np.float32),
    )
    dataset = C1M3StrictTargetDataset(
        {"synthetic": record}, object(), 1.0, outer_date="19250108",
    )
    assert calls == [(1.0, 2.0, 3.0)]
    assert dataset.manifest()["support_members"] == 3


def test_fixed_chunk_stage1_readout_is_selection_informed_and_cannot_replace_primary() -> None:
    rows = []
    for index, date in enumerate(DATE_ORDER):
        base = 0.2 + index * 0.01
        values = (base, base + 0.02, base + 0.012, base + 0.001)
        rows.append({"outer_date": date, "results": [
            {"arm": arm, "equal_recording_mean_r2": value}
            for arm, value in zip(ARM_ORDER, values, strict=True)
        ]})
    primary = decide(rows)
    selected = _selected_fixed_chunk_readout(primary)
    assert primary["gate_2_deployable_recovery"]["pass"] is False
    assert selected["pass"] is True
    assert selected["selection_history_disclosed"] is True
    assert selected["may_not_replace_primary_D_in_this_result"] is True
