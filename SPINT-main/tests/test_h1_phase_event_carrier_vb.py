"""Synthetic/no-NWB tests for the isolated H1 phase/event Version-B gate."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data import h1_phase_event_carrier_vb as vb


def _session(
    name: str,
    rng: np.random.Generator,
    *,
    time_locked: bool = False,
    zero_snapto_eval: bool = True,
) -> vb.PhaseEventSession:
    channels, phases = vb.EXPECTED_NEURONS, len(vb.EVENT_LABELS)
    phase_axis = np.linspace(-1.0, 1.0, phases)
    loadings = np.stack((
        phase_axis,
        np.sin(np.linspace(0.0, 2.0 * np.pi, phases, endpoint=False)),
        np.cos(np.linspace(0.0, 2.0 * np.pi, phases, endpoint=False)),
        np.where(np.arange(phases) % 2 == 0, 1.0, -1.0),
    ))
    latent_rng = np.random.default_rng(73)
    latent = latent_rng.normal(size=(channels, 4))
    log_rate = 3.4 + 0.28 * latent @ loadings
    rate = np.exp(log_rate)
    exposures = np.full((4, phases), 8.0, dtype=np.float64)
    counts = np.stack([
        rng.poisson(rate * exposures[trial][None, :]).astype(np.float64)
        for trial in range(4)
    ])
    eval_bins = np.full((4, phases), 100, dtype=np.int64)
    if zero_snapto_eval:
        eval_bins[:, vb.EVENT_LABELS.index("SnapTo")] = 0
    intervals = np.empty((4, phases, 2), dtype=np.float64)
    for trial in range(4):
        if time_locked:
            order = np.arange(phases)
        else:
            order = np.random.default_rng(9000 + trial).permutation(phases)
        for position, label in enumerate(order):
            start = 0.02 + 0.115 * position
            intervals[trial, label] = (start, start + 0.08)
    return vb.PhaseEventSession(
        session_name=name,
        date=vb.session_date(name),
        input_path=Path(f"/{name}.nwb"),
        input_sha256=(hex(abs(hash(name)))[2:] * 64)[:64].ljust(64, "0"),
        trial_values=(1.0, 2.0, 3.0, 4.0),
        counts=counts,
        exposures=exposures,
        eval_valid_bins=eval_bins,
        normalized_event_midpoints=intervals.mean(axis=-1),
        raw_epoch_hits=np.ones((4, phases), dtype=np.int64),
        clock_step_seconds=0.02,
        clock_semantics="synthetic",
    )


def _sessions(*, time_locked: bool = False) -> dict[str, vb.PhaseEventSession]:
    rng = np.random.default_rng(20260809)
    return {
        name: _session(name, rng, time_locked=time_locked)
        for name in vb.H1_HELDIN_SESSIONS
    }


def test_frozen_proposal_is_immutable_and_matches_implementation() -> None:
    proposal = (
        Path(__file__).parents[1]
        / "pilot_artifacts/h1_phase_event_carrier_vb/H1_PHASE_EVENT_CARRIER_VB_PROPOSAL_v1.json"
    )
    body = vb.load_frozen_proposal(proposal)
    assert body["schema"] == vb.PROPOSAL_SCHEMA
    assert body["scope"]["gpu_runs_authorized"] == 0
    assert body["exposure_and_label_semantics"]["velocity_used"] is False


def test_source_lodo_p4_recovers_stable_per_channel_shape_and_beats_controls() -> None:
    sessions = _sessions()
    outer = "19250108"
    plan = vb.fit_source_plan(sessions, outer)
    target = next(value for value in sessions.values() if value.date == outer)
    first, second = vb.event_view(target, (0, 1)), vb.event_view(target, (2, 3))
    p_first, p_second = vb.project_shape(first, plan), vb.project_shape(second, plan)
    correct = float(np.nanmedian(vb.row_cosines(p_first, p_second)))
    row = float(np.nanmedian(vb.row_cosines(p_first, p_second[::-1])))
    event_permutation = np.roll(np.arange(len(vb.EVENT_LABELS)), 1)
    shuffled = vb.EventView(
        log_rates=second.log_rates[:, event_permutation],
        shape=second.shape[:, event_permutation],
        log_baseline_rate=second.log_baseline_rate,
        exposures=second.exposures[event_permutation],
    )
    event = float(np.nanmedian(vb.row_cosines(p_first, vb.project_shape(shuffled, plan))))
    assert plan.rank >= 4
    assert correct > 0.8
    assert correct > row + 0.5
    assert correct > event + 0.3
    assert float(np.nanmedian(vb.forward_gain(first, second))) > 0.5


def test_outer_date_never_refits_source_plan() -> None:
    sessions = _sessions()
    outer = "19250108"
    before = vb.fit_source_plan(sessions, outer)
    changed = dict(sessions)
    target_name = next(name for name, value in sessions.items() if value.date == outer)
    changed[target_name] = replace(
        sessions[target_name],
        counts=sessions[target_name].counts * 1000.0 + 123.0,
    )
    after = vb.fit_source_plan(changed, outer)
    assert before.plan_sha256 == after.plan_sha256
    np.testing.assert_array_equal(before.components, after.components)
    np.testing.assert_array_equal(before.event_mean, after.event_mean)


def test_snapto_raw_exposure_is_constructible_with_zero_eval_valid_bins() -> None:
    session = next(iter(_sessions().values()))
    snapto = vb.EVENT_LABELS.index("SnapTo")
    assert session.eval_valid_bins[:, snapto].sum() == 0
    assert session.exposures[:, snapto].sum() > 0
    view = vb.event_view(session, (0, 1))
    assert np.isfinite(view.log_rates[:, snapto]).all()


def test_zero_raw_event_exposure_fails_closed() -> None:
    session = next(iter(_sessions().values()))
    exposure = session.exposures.copy()
    exposure[:2, vb.EVENT_LABELS.index("SnapTo")] = 0.0
    broken = replace(session, exposures=exposure)
    with pytest.raises(vb.PhaseEventCarrierError, match="zero exposure"):
        vb.event_view(broken, (0, 1))


def test_nonoverlapping_native_fragments_accumulate_but_overlap_fails() -> None:
    phases = len(vb.EVENT_LABELS)
    counts = np.zeros((4, vb.EXPECTED_NEURONS, phases), dtype=np.float64)
    exposures = np.zeros((4, phases), dtype=np.float64)
    eval_bins = np.zeros((4, phases), dtype=np.int64)
    midpoint_sum = np.zeros((4, phases), dtype=np.float64)
    hits = np.zeros((4, phases), dtype=np.int64)
    segments = [[[] for _ in vb.EVENT_LABELS] for _ in range(4)]
    spike_rows = [np.asarray([0.02, 0.82], dtype=np.float64)] + [
        np.empty(0, dtype=np.float64) for _ in range(vb.EXPECTED_NEURONS - 1)
    ]
    timestamps = np.arange(50, dtype=np.float64) * 0.02
    trial_num = np.ones(50, dtype=np.float64)
    common = dict(
        counts=counts, exposures=exposures, eval_bins=eval_bins,
        midpoint_weighted_sum=midpoint_sum, hits=hits, segments=segments,
        trial_index=0, event_index=vb.EVENT_LABELS.index("Release"),
        trial_start=0.0, trial_stop=1.0, spike_rows=spike_rows,
        timestamps=timestamps, eval_mask=np.ones(50, dtype=bool),
        trial_num=trial_num, trial_value=1.0,
    )
    vb._accumulate_event_overlap(left=0.0, right=0.1, **common)
    vb._accumulate_event_overlap(left=0.8, right=0.9, **common)
    event = vb.EVENT_LABELS.index("Release")
    assert exposures[0, event] == pytest.approx(0.2)
    assert counts[0, 0, event] == 2
    assert hits[0, event] == 2
    with pytest.raises(vb.PhaseEventCarrierError, match="double count"):
        vb._accumulate_event_overlap(left=0.85, right=0.95, **common)


def test_time_only_redundancy_probe_separates_locked_from_unlocked_profiles() -> None:
    locked = next(iter(_sessions(time_locked=True).values()))
    unlocked = next(iter(_sessions(time_locked=False).values()))
    assert vb.time_only_phase_accuracy(locked) == pytest.approx(1.0)
    assert vb.time_only_phase_accuracy(unlocked) < 0.5


def test_time_only_diagnostic_uses_available_half_level_event_intervals() -> None:
    session = next(iter(_sessions(time_locked=True).values()))
    midpoints = session.normalized_event_midpoints.copy()
    midpoints[0, vb.EVENT_LABELS.index("Carry")] = np.nan
    midpoints[2, vb.EVENT_LABELS.index("Release")] = np.nan
    sparse = replace(session, normalized_event_midpoints=midpoints)
    accuracy, coverage = vb.time_only_phase_diagnostic(sparse)
    assert accuracy == pytest.approx(1.0)
    assert coverage == pytest.approx(15 / 16)


def test_audit_kills_time_locked_candidate_even_when_shape_is_reliable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vb, "NULL_REPLICATES", 16)
    result = vb.audit_sessions(_sessions(time_locked=True))
    assert result["confirmatory_gate_counts"]["reliability_dates"] >= 4
    assert result["confirmatory_gate_counts"]["time_locked_dates"] == 5
    assert result["status"] == "FAIL_TIME_LOCKED_REDUNDANCY_NO_GPU"
    assert result["scope"]["r2_values_read_or_computed"] == 0
    assert result["scope"]["gpu_used"] is False


def test_frozen_exposure_gate_returns_constructibility_failure_not_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vb, "NULL_REPLICATES", 8)
    sessions = _sessions(time_locked=False)
    name = vb.H1_HELDIN_SESSIONS[0]
    exposure = sessions[name].exposures.copy()
    exposure[2:, vb.EVENT_LABELS.index("Carry")] = 0.03
    sessions[name] = replace(sessions[name], exposures=exposure)
    result = vb.audit_sessions(sessions)
    assert result["constructibility"]["all_event_half_exposures_at_least_minimum"] is False
    assert result["status"] == "FAIL_CONSTRUCTIBILITY_NO_GPU"


def test_audit_source_contains_no_training_or_r2_import() -> None:
    source = (Path(__file__).parents[1] / "scripts/h1_phase_event_carrier_vb_source_audit.py").read_text(
        encoding="utf-8"
    ).lower()
    assert "import torch" not in source
    assert "subprocess" not in source
    assert "nvidia-smi" not in source
    assert "train.py" not in source
    assert "evalai_submit" not in source
