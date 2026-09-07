"""Synthetic contract tests for the standalone H1 M=2 date-LODO audit.

The audit deliberately lives under ``scripts/`` so it cannot inherit the
stopped H1 q3 pilot's data path or model conventions.  These tests load that
file directly and exercise only in-memory arrays: no NWB file, GPU, decoder,
or formal held-out scope is involved.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_h1_afc4_m2_date_lodo.py"
SPEC = importlib.util.spec_from_file_location("h1_afc4_m2_date_lodo_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
# ``dataclass`` resolves the module through sys.modules while the script is
# executed, so register the dynamically loaded module before exec_module.
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _synthetic_record(session_name: str, *, seed: int, blocks_per_trial: int = 18):
    """Make two stable trial blocks with a nondegenerate seven-D kinematic law."""

    rng = np.random.default_rng(seed)
    neurons = 11
    total_blocks = 2 * blocks_per_trial
    # Full-rank velocity coverage is deliberately shared across dates while a
    # small trial-specific perturbation makes the split-half check meaningful.
    block_velocity = rng.normal(size=(total_blocks, AUDIT.VELOCITY_DIM))
    block_velocity[:, 0] += np.linspace(-1.5, 1.5, total_blocks)
    block_velocity[:, 1] += np.sin(np.linspace(0.0, 3.0 * np.pi, total_blocks))
    weights = rng.normal(scale=0.35, size=(AUDIT.VELOCITY_DIM, neurons))
    intercept = rng.uniform(6.0, 10.0, size=(neurons,))
    block_rate = intercept[None, :] + block_velocity @ weights
    block_rate += rng.normal(scale=0.035, size=block_rate.shape)

    # ``record_from_arrays`` consumes 20-ms bins.  Repeating the block rate
    # five times makes the expected 100-ms value exact up to its controlled
    # within-block noise.  All bins are eval-valid and finite: no active gate
    # can accidentally participate in the test.
    neural = np.repeat(block_rate / AUDIT.BLOCK_BINS, AUDIT.BLOCK_BINS, axis=0)
    neural += rng.normal(scale=0.002, size=neural.shape)
    velocity = np.repeat(block_velocity, AUDIT.BLOCK_BINS, axis=0)
    trial_num = np.repeat(
        np.asarray((1.0, 2.0), dtype=np.float64),
        blocks_per_trial * AUDIT.BLOCK_BINS,
    )
    eval_mask = np.ones(neural.shape[0], dtype=bool)
    return AUDIT.record_from_arrays(
        session_name=session_name,
        neural=neural,
        velocity=velocity,
        eval_mask=eval_mask,
        trial_num=trial_num,
        input_sha256=f"synthetic-{seed}",
    )


def _six_date_records():
    records = {}
    for index, date in enumerate(("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")):
        session = f"ses-{date}T11{index:04d}"
        records[session] = _synthetic_record(session, seed=100 + index)
    return records


def test_m2_blocks_never_cross_trial_boundary_or_eval_gap():
    """A partial first trial and a gap cannot be joined into a 100-ms block."""

    bins = 3 + 12
    neural = np.ones((bins, 3), dtype=np.float64)
    velocity = np.ones((bins, AUDIT.VELOCITY_DIM), dtype=np.float64)
    trial_num = np.r_[np.ones(3), np.full(12, 2.0)]
    eval_mask = np.ones(bins, dtype=bool)
    eval_mask[7] = False
    record = AUDIT.record_from_arrays(
        session_name="ses-19250101T111740",
        neural=neural,
        velocity=velocity,
        eval_mask=eval_mask,
        trial_num=trial_num,
    )

    first, second = record.trials
    assert first.rates.shape == (0, 3)
    # Second trial contains runs of length 4 and 7, so it yields exactly one
    # legal five-bin block; neither trial boundary nor invalid bin is crossed.
    assert second.rates.shape == (1, 3)
    assert first.audit["block_crosses_trial_boundary"] is False
    assert second.audit["block_crosses_trial_boundary"] is False
    assert first.audit["activity_threshold_used"] is None


def test_date_lodo_has_six_outer_dates_and_never_leaks_target_date():
    records = _six_date_records()
    partitions = AUDIT.date_lodo_partitions(records)

    assert len(partitions) == 6
    for partition in partitions:
        outer = partition["outer_date"]
        assert len(partition["source_dates"]) == 5
        assert outer not in partition["source_dates"]
        assert all(records[name].date != outer for name in partition["source_sessions"])
        assert all(records[name].date == outer for name in partition["target_sessions"])


def test_target_mutation_cannot_change_source_basis_or_selection():
    records = _six_date_records()
    outer_date = "19250113"
    before = AUDIT.fit_source_plan(records, outer_date)
    target_name = next(name for name, record in records.items() if record.date == outer_date)
    target = records[target_name]
    # Alter every target neural response and velocity value.  The outer date
    # must not affect PCA, q choice, lag/ridge score, tie-break, or plan hash.
    mutated_trials = tuple(
        replace(
            trial,
            rates=trial.rates + 500.0,
            velocity=trial.velocity * -37.0 + 19.0,
        )
        for trial in target.trials
    )
    mutated = dict(records)
    mutated[target_name] = replace(target, trials=mutated_trials)
    after = AUDIT.fit_source_plan(mutated, outer_date)

    assert before.plan_sha256 == after.plan_sha256
    assert before.selected_lag_blocks == after.selected_lag_blocks
    assert before.selected_ridge == after.selected_ridge
    assert before.basis.q == after.basis.q
    np.testing.assert_allclose(before.basis.mean, after.basis.mean)
    np.testing.assert_allclose(before.basis.components, after.basis.components)
    np.testing.assert_allclose(before.basis.eigenvalues, after.basis.eigenvalues)
    assert before.candidate_selection == after.candidate_selection


def test_primary_s0_gate_ignores_rotationally_invariant_diagnostics():
    """Only six signed-W date values can pass S0; diagnostics cannot rescue it."""

    passing_rows = [
        {
            "status": "defined",
            "primary_signed_w_direction_cosine_date_aggregate": 0.51 if index < 4 else 0.49,
            "diagnostic_not_authorized": {
                "date_mean_w_norm_pearson": 0.99,
                "date_mean_b_pearson": 0.99,
            },
        }
        for index in range(6)
    ]
    gate = AUDIT.s0_gate(passing_rows)
    assert gate["pass"] is True
    assert gate["dates_at_or_above_threshold"] == 4
    assert gate["diagnostic_not_authorized"]["w_norm_b_and_label_rotation_influence_s0"] is False

    failing_rows = [dict(row) for row in passing_rows]
    failing_rows[0] = {
        **failing_rows[0],
        "status": "undefined",
        "primary_signed_w_direction_cosine_date_aggregate": None,
    }
    failed = AUDIT.s0_gate(failing_rows)
    assert failed["pass"] is False
    assert failed["dates_defined"] == 5
    assert failed["status"] == "STOP_S0_SIGNED_W_FEASIBILITY_FAILED"


def test_target_evaluation_reports_but_cannot_promote_diagnostic_candidate():
    records = _six_date_records()
    outer_date = "19250120"
    plan = AUDIT.fit_source_plan(records, outer_date)
    target_name = next(name for name, record in records.items() if record.date == outer_date)
    result = AUDIT.evaluate_target_record(records[target_name], plan)

    diagnostic = result["diagnostic_not_authorized"]
    assert result["support"]["trial_count"] == 2
    assert result["support"]["block_bins"] == 5
    assert result["support"]["activity_threshold_used"] is None
    assert diagnostic["not_part_of_signed_w_s0"] is True
    assert diagnostic["cannot_authorize_gpu"] is True
    assert "trial1_vs_trial2_w_norm_pearson" in diagnostic
    assert "trial1_vs_trial2_b_pearson" in diagnostic
    assert diagnostic["correct_pairing_vs_within_trial_block_rotation_ls"]["status"] == "defined"
