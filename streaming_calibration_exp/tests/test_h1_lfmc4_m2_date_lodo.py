"""Synthetic contracts for the CPU-only H1 LFMC4 M=2 gate.

These tests use in-memory native 20-ms arrays only.  They do not load NWBs,
query/minival data, a decoder, or a GPU.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_h1_lfmc4_m2_date_lodo.py"
SPEC = importlib.util.spec_from_file_location("h1_lfmc4_m2_date_lodo_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
LFMC = AUDIT.lfmc4
DATES = AUDIT.H1_DATES


def _basis() -> LFMC.FrozenBasis:
    """A fixed synthetic source basis whose whitened latent is ~10*y[:4]."""
    w1 = np.zeros((16, 7), dtype=np.float64)
    w1[:7] = np.eye(7)
    w2 = np.zeros((4, 16), dtype=np.float64)
    w2[:, :4] = np.eye(4)
    return LFMC.FrozenBasis(
        velocity_mean=np.zeros(7), velocity_scale=np.ones(7), w1=w1, b1=np.zeros(16),
        # Uniform(-.08,.08) source coordinates have s.d. about .046; this
        # fixed scale is the synthetic analogue of source latent whitening.
        w2=w2, b2=np.zeros(4), latent_mean=np.zeros(4), whitener=np.eye(4) * 22.0,
        source_state_sha256="synthetic-basis", training_audit={},
    )


def _record(date: str, index: int, *, independent: bool = False, trials: int = 2, channels: int = 9):
    rng = np.random.default_rng(100 + index)
    bins_per_trial = 80
    velocity = rng.uniform(-0.08, 0.08, size=(trials * bins_per_trial, 7))
    weights = np.linspace(1.0, 2.0, channels)
    if independent:
        rate = 20.0 + rng.normal(scale=2.0, size=(trials * bins_per_trial, weights.size))
    else:
        rate = 20.0 + velocity[:, 0:1] * weights[None, :] * 12.0
    # Raw counts may be real-valued in a synthetic contract; all formulae use
    # the exposure explicitly and do not assume a particular count dtype.
    neural = rate * LFMC.BIN_SECONDS
    trial_num = np.repeat(np.arange(1.0, trials + 1.0), bins_per_trial)
    return AUDIT.record_from_arrays(
        session_name=f"ses-{date}T12{index:04d}", neural=neural, velocity=velocity,
        eval_mask=np.ones(trials * bins_per_trial, dtype=bool), trial_num=trial_num,
        input_sha256=f"synthetic-{index}",
    )


def test_native_twenty_ms_trial_runs_and_rotation_never_cross_invalid_gap():
    neural = np.ones((10, 3), dtype=np.float64)
    velocity = np.arange(70, dtype=np.float64).reshape(10, 7)
    trial_num = np.r_[np.ones(5), np.full(5, 2.0)]
    mask = np.ones(10, dtype=bool)
    mask[2] = False
    record = AUDIT.record_from_arrays(
        session_name="ses-19250101T120000", neural=neural, velocity=velocity,
        eval_mask=mask, trial_num=trial_num,
    )
    assert record.trials[0].counts.shape[0] == 4
    assert record.trials[0].run_lengths == (2, 2)
    rotated, offsets = LFMC.rotated_velocity(record.trials[0], offset_seed=7)
    assert all(offset > 0 for offset in offsets)
    # Each contiguous run preserves its own values and cannot receive values
    # from the other run or from a different TrialNum.
    np.testing.assert_array_equal(np.sort(rotated[:2], axis=0), np.sort(record.trials[0].velocity[:2], axis=0))
    np.testing.assert_array_equal(np.sort(rotated[2:], axis=0), np.sort(record.trials[0].velocity[2:], axis=0))


def test_exposure_weighted_cross_moment_matches_explicit_sufficient_statistic():
    basis = _basis()
    trial = LFMC.RawTrial(
        trial_number=1.0,
        counts=np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64),
        velocity=np.asarray([[.01, 0, 0, 0, 0, 0, 0], [-.01, 0, 0, 0, 0, 0, 0]], dtype=np.float64),
        exposure=np.asarray([.02, .04]), run_lengths=(2,), audit={},
    )
    got = LFMC.trial_moment(trial, basis)["moment"]
    psi = basis.latent(trial.velocity)
    total = trial.exposure.sum()
    expected = trial.counts.T @ psi / total - (trial.counts.sum(axis=0) / total)[:, None] * ((trial.exposure[:, None] * psi).sum(axis=0) / total)[None, :]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)


def test_correct_pairing_has_positive_no_normal_equation_transfer_and_beats_rotation_null():
    record = _record(DATES[0], 0)
    basis = _basis()
    gain = LFMC.symmetric_transfer_gain(record, basis)
    assert float(np.nanmedian(gain)) > 0.5
    rotated_one, _ = LFMC.rotated_velocity(record.trials[0], offset_seed=101)
    rotated_two, _ = LFMC.rotated_velocity(record.trials[1], offset_seed=102)
    null = LFMC.symmetric_transfer_gain(record, basis, fit_velocity_one=rotated_one, fit_velocity_two=rotated_two)
    assert float(np.nanmedian(gain)) > float(np.nanmedian(null))


def test_cpu_gate_is_four_clause_conjunction():
    rows = [{
        "status": "defined", "date_coverage_fraction": .95,
        "date_split_trial_moment_cosine": .70,
        "date_correct_symmetric_cross_trial_gain": .10,
        "date_correct_minus_null_q95": .02,
    } for _ in DATES]
    assert AUDIT.cpu_gate(rows)["pass"] is True
    for row in rows[:3]:
        row["date_correct_symmetric_cross_trial_gain"] = -.01
    gate = AUDIT.cpu_gate(rows)
    assert gate["pass"] is False
    assert gate["subgates"]["correct_symmetric_held_trial_rate_prediction_gain"]["dates_passing"] == 3


def test_record_coverage_uses_same_channel_intersection_and_does_not_require_100_percent(monkeypatch):
    record = _record(DATES[0], 0, channels=20)
    invalid_one = np.r_[np.ones(19), np.nan]
    monkeypatch.setattr(AUDIT.lfmc4, "cosine_by_channel", lambda _one, _two: invalid_one.copy())
    monkeypatch.setattr(AUDIT.lfmc4, "symmetric_transfer_gain", lambda *_a, **_k: invalid_one.copy())
    row = AUDIT._record_metrics(record, {"basis": _basis(), "plan_sha256": "synthetic"})
    assert row["status"] == "defined"
    assert row["coverage_fraction"] == .95
    assert row["coverage_rule"].startswith("mean(isfinite")
    passing = [{"status": "defined", "date_coverage_fraction": .95, "date_split_trial_moment_cosine": .7, "date_correct_symmetric_cross_trial_gain": .1, "date_correct_minus_null_q95": .01} for _ in DATES]
    assert AUDIT.cpu_gate(passing)["subgates"]["coverage_at_least_090"]["pass"] is True
    for value in passing[:3]:
        value["date_coverage_fraction"] = .89
    assert AUDIT.cpu_gate(passing)["subgates"]["coverage_at_least_090"]["pass"] is False


def test_correct_and_null_medians_share_one_predeclared_channel_mask(monkeypatch):
    record = _record(DATES[0], 0, channels=20)
    cosine = np.arange(20, dtype=np.float64)
    gain = 100.0 + np.arange(20, dtype=np.float64)
    cosine[0] = np.nan
    gain[1] = np.nan
    monkeypatch.setattr(AUDIT.lfmc4, "cosine_by_channel", lambda _one, _two: cosine.copy())
    monkeypatch.setattr(AUDIT.lfmc4, "symmetric_transfer_gain", lambda *_a, **_k: gain.copy())
    row = AUDIT._record_metrics(record, {"basis": _basis(), "plan_sha256": "synthetic"})
    assert row["status"] == "defined"
    assert row["valid_channel_mask"]["valid_channels"] == 18
    assert row["split_trial_moment_cosine"]["median"] == float(np.median(cosine[2:]))
    assert row["correct_symmetric_held_trial_gain"]["median"] == float(np.median(gain[2:]))
    assert all(item["defined_channels"] == 18 for item in row["label_rotation_null"]["replicate_rows"])


def test_outer_mutation_cannot_change_source_plan_when_basis_observes_only_source(monkeypatch):
    records = {record.session_name: record for index, date in enumerate(DATES) for record in (_record(date, index),)}
    # Avoid an expensive optimizer in this isolation test.  The fake basis
    # deliberately hashes the exact source velocity values it receives, so a
    # target leak would be detected by the plan hash.
    def fake_fit(source):
        values = np.concatenate([trial.velocity for record in source for trial in record.source_trials])
        result = _basis()
        return replace(result, source_state_sha256=LFMC.array_hash(values))
    monkeypatch.setattr(AUDIT.lfmc4, "fit_source_basis", fake_fit)
    before = AUDIT.source_plan(records, DATES[2])
    target_name = next(name for name, record in records.items() if record.date == DATES[2])
    target = records[target_name]
    changed = tuple(replace(trial, counts=trial.counts + 500.0, velocity=trial.velocity * -17.0) for trial in target.trials)
    mutated = dict(records)
    mutated[target_name] = replace(target, trials=changed)
    after = AUDIT.source_plan(mutated, DATES[2])
    assert before["plan_sha256"] == after["plan_sha256"]
    assert before["basis"].source_state_sha256 == after["basis"].source_state_sha256


def test_target_later_trial_never_enters_source_plan_and_source_later_pairs_enter_normalizer(monkeypatch):
    records = {record.session_name: record for index, date in enumerate(DATES) for record in (_record(date, index, trials=3),)}
    def fake_fit(source):
        return _basis()
    monkeypatch.setattr(AUDIT.lfmc4, "fit_source_basis", fake_fit)
    outer = DATES[1]
    before = AUDIT.source_plan(records, outer)
    target_name = next(name for name, value in records.items() if value.date == outer)
    target = records[target_name]
    changed_source_trials = target.source_trials[:2] + tuple(replace(trial, velocity=trial.velocity + 99.0) for trial in target.source_trials[2:])
    mutated = dict(records)
    mutated[target_name] = replace(target, source_trials=changed_source_trials)
    after = AUDIT.source_plan(mutated, outer)
    assert before["plan_sha256"] == after["plan_sha256"]
    # Deployment/gate features likewise remain first-two-only; a target later
    # trial may be retained for source use in other folds but is never read in
    # this target evaluation.
    original_metrics = AUDIT._record_metrics(target, before)
    mutated_metrics = AUDIT._record_metrics(mutated[target_name], after)
    assert original_metrics["combined_m2_descriptor_sha256"] == mutated_metrics["combined_m2_descriptor_sha256"]
    assert original_metrics["split_trial_moment_cosine"] == mutated_metrics["split_trial_moment_cosine"]
    source = [records[name] for name in before["source_sessions"]]
    normalizer = AUDIT._source_normalizer(source, _basis())
    assert all(value == 2 for value in normalizer["source_contiguous_m2_pair_counts"].values())


def test_equal_date_equal_recording_source_statistics_ignore_duplicated_bins(monkeypatch):
    records = [_record(date, i, trials=3) for i, date in enumerate(DATES[:5])]
    values, weights = LFMC._source_velocity_with_equal_record_weights(records)
    mean = np.sum(values * weights[:, None], axis=0)
    copied = records[0]
    copied_trials = tuple(replace(trial, velocity=np.repeat(trial.velocity, 2, axis=0), counts=np.repeat(trial.counts, 2, axis=0), exposure=np.repeat(trial.exposure, 2), run_lengths=tuple(length * 2 for length in trial.run_lengths)) for trial in copied.source_trials)
    copied_records = [replace(copied, source_trials=copied_trials), *records[1:]]
    repeat_values, repeat_weights = LFMC._source_velocity_with_equal_record_weights(copied_records)
    repeat_mean = np.sum(repeat_values * repeat_weights[:, None], axis=0)
    np.testing.assert_allclose(mean, repeat_mean, rtol=0, atol=1e-13)
    normalizer = AUDIT._source_normalizer(records, _basis())
    repeat_normalizer = AUDIT._source_normalizer(copied_records, _basis())
    np.testing.assert_allclose(normalizer["mean"], repeat_normalizer["mean"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(normalizer["raw_std"], repeat_normalizer["raw_std"], rtol=0, atol=1e-13)


def test_remainder_rotation_is_exactly_equal_date_over_complete_cycle():
    records = [_record(date, i) for i, date in enumerate(DATES[:5])]
    tagged = []
    for index, record in enumerate(records):
        source_trials = tuple(replace(trial, velocity=np.full_like(trial.velocity, float(index))) for trial in record.source_trials)
        tagged.append(replace(record, source_trials=source_trials))
    dates, by_date = LFMC._weighted_source_records(tagged)
    rng = np.random.default_rng(9)
    counts = np.zeros(5, dtype=int)
    for step in range(5):
        batch = LFMC._equal_date_equal_recording_batch(dates=dates, records_by_date=by_date, rng=rng, step=step)
        for index in range(5):
            counts[index] += int(np.sum(batch[:, 0] == float(index)))
    np.testing.assert_array_equal(counts, np.full(5, LFMC.BASIS_BATCH_SIZE))


def test_forbidden_paths_are_rejected_before_any_loader_can_open_them(tmp_path):
    with np.testing.assert_raises(AUDIT.AuditError):
        AUDIT._require_heldin_calib(tmp_path / "sub-HumanPitt-held-out-calib" / "x.nwb")
    with np.testing.assert_raises(AUDIT.AuditError):
        AUDIT._require_data_root(tmp_path / "formal" / "SPINT-main" / "data" / "000954")


def test_partitions_reject_any_six_unexpected_calendar_dates():
    records = {record.session_name: replace(record, date=f"200001{index + 1:02d}") for index, record in enumerate(_record(date, index) for index, date in enumerate(DATES))}
    with np.testing.assert_raises(AUDIT.AuditError):
        AUDIT.partitions(records)
