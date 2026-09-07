from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "streaming_calibration_exp/scripts/audit_h1_m4_gated_reduced_rank_carrier_date_lodo.py"
SPEC = importlib.util.spec_from_file_location("h1_m4_gated_reduced_rank", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _record(date: str, serial: int) -> audit.FullRecord:
    rng = np.random.default_rng(3100 + serial)
    mixing = rng.normal(scale=0.20, size=(7, 176))
    trials = []
    for number in range(1, 8):
        phase = np.linspace(0.0, 2.0 * np.pi, 30, endpoint=False) + number * 0.13
        velocity = np.column_stack([np.sin((axis + 1) * phase) for axis in range(7)])
        rates = 30.0 + velocity @ mixing
        # All support trials retain exactly 20/30 correct-label blocks under G1.
        active = np.ones(30, dtype=bool)
        active[::3] = False
        trials.append(audit.TrialBlocks(float(number), rates, velocity, active, {"synthetic": True}))
    return audit.FullRecord(f"ses-{date}T{serial:06d}", date, None, f"synthetic-{date}", tuple(trials))


def _records() -> dict[str, audit.FullRecord]:
    result = {}
    for serial, date in enumerate(audit.h1.H1_DATES):
        record = _record(date, serial)
        result[record.session_name] = record
    return result


def test_g1_is_raw_five_bin_all_samples_active() -> None:
    neural = np.ones((10, 176), dtype=np.float64)
    velocity = np.full((10, 7), 0.1, dtype=np.float64)
    velocity[2] = 0.0  # one still 20-ms sample invalidates its entire first 5-bin block
    trial = audit._trial_to_blocks(
        trial_number=1.0,
        neural=neural,
        velocity=velocity,
        eval_mask=np.ones(10, dtype=bool),
        trial_num=np.ones(10, dtype=np.float64),
    )
    assert trial.rates.shape == (2, 176)
    assert trial.active_5bin.tolist() == [False, True]


def test_source_plan_is_outer_date_only_and_rank_is_source_fixed() -> None:
    records = _records()
    outer = audit.h1.H1_DATES[0]
    source = audit._build_source_plan(records, outer, "G1_rt_active_5bin")
    plan = audit._cell_plan(source, 3)
    assert all(records[name].date != outer for name in source.source_sessions)
    assert source.pcs.shape == (audit.Q_NEURAL, 176)
    assert plan.U.shape == (7, 3)
    assert plan.prior_mu.shape == (3,)
    assert 0.0 < plan.source_energy_fraction <= 1.0


def test_rotation_keeps_correct_gate_selected_neural_rows_and_counts_fixed() -> None:
    records = _records()
    record = next(iter(records.values()))
    rotated, counts = audit._rotated_retained_labels(record, "G1_rt_active_5bin", replicate=0)
    support, _query = audit._support_query(record)
    expected = tuple(int(trial.active_5bin.sum()) for trial in support)
    assert counts == expected == (20, 20, 20, 20)
    for index, trial in enumerate(support):
        correct = trial.velocity[trial.active_5bin]
        assert rotated[index].shape == correct.shape
        assert not np.array_equal(rotated[index], correct)


def test_reconstructed_rank_carrier_and_ungated_query_are_recorded_for_all_six_cells() -> None:
    records = _records()
    output = audit.run_audit(records, require_authority_records=False)
    assert len(output["cells"]) == 6
    assert output["decision"]["gpu_authorized_by_this_receipt"] is False
    primary = next(cell for cell in output["cells"] if cell["is_primary"])
    assert primary["gate"] == "G1_rt_active_5bin"
    assert primary["q_kin"] == 3
    date_row = primary["date_lodo"][0]
    assert date_row["status"] == "defined"
    result = date_row["records"][0]
    assert result["carrier_shape"] == [176, 3]
    assert result["carrier_reconstruction"] == "E@U.T"
    assert result["support_candidate_100ms_blocks"] == 120
    assert result["support_retained_100ms_blocks"] == 80
    assert result["query_100ms_blocks_ungated"] == 90
    assert result["rotation_null_reuses_correct_frozen_retained_blocks"] is True
    assert len(result["rotation_null_reconstructed_query_r2"]) == audit.NULL_REPLICATES
    assert all(counts == [20, 20, 20, 20] for counts in result["rotation_null_retained_by_trial"])
    assert result["row_shuffle_permutation_sha256"]


def test_metric_is_exactly_the_carrier_implied_e_times_u_transpose_reconstruction() -> None:
    records = _records()
    outer = audit.h1.H1_DATES[0]
    record = records[f"ses-{outer}T000000"]
    plan = audit._cell_plan(audit._build_source_plan(records, outer, "G1_rt_active_5bin"), 2)
    beta, covariance, sigma2, _indices = audit._fit_target(record, plan)
    carrier, _weight, _variance = audit._shrink(beta, covariance, sigma2, plan)
    observed = audit._metric(record, carrier, beta[0], plan)
    assert observed is not None
    rates, labels, _query = audit._query_arrays(record)
    prediction = (rates - plan.source.mean[None, :]) @ (carrier @ plan.U.T) + beta[0][None, :]
    support_mean = audit._support_mean(record, plan.source.gate)
    expected = 1.0 - np.square(labels - prediction).sum() / np.square(labels - support_mean[None, :]).sum()
    assert observed["r2"] == expected


def test_real_audit_requires_exact_authority_session_set() -> None:
    with pytest.raises(audit.AuditError, match="exactly the 13"):
        audit.run_audit(_records())
