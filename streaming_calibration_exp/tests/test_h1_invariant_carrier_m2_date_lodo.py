"""Synthetic protocol tests for the H1 M=2 invariant-carrier CPU gate.

All records here are in-memory arrays.  These tests never open NWB/query/
formal data and never construct a model, optimizer, or GPU process.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_h1_invariant_carrier_m2_date_lodo.py"
SPEC = importlib.util.spec_from_file_location("h1_invariant_carrier_m2_date_lodo_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
SIGNED = AUDIT.signed_audit


DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")
NEURONS = 13
BLOCKS_PER_TRIAL = 34


def _records(*, law: str):
    """Six date records with either a true or deliberately label-independent law."""

    if law not in {"correct", "independent"}:
        raise ValueError(law)
    global_rng = np.random.default_rng(77)
    shared_weights = global_rng.normal(scale=0.60, size=(3, NEURONS))
    shared_intercept = global_rng.uniform(12.0, 17.0, size=NEURONS)
    records = {}
    for date_index, date in enumerate(DATES):
        rng = np.random.default_rng(1000 + date_index)
        total_blocks = 2 * BLOCKS_PER_TRIAL
        velocity_blocks = np.zeros((total_blocks, SIGNED.VELOCITY_DIM), dtype=np.float64)
        # Three independent axes make source q=3 both defined and meaningful.
        velocity_blocks[:, :3] = rng.normal(size=(total_blocks, 3))
        if law == "correct":
            rate_blocks = shared_intercept[None, :] + velocity_blocks[:, :3] @ shared_weights
            rate_blocks += rng.normal(scale=0.015, size=rate_blocks.shape)
        else:
            # Same marginal scale, but independent of every velocity block and
            # independent between the two trials.
            rate_blocks = shared_intercept[None, :] + rng.normal(
                scale=1.2, size=(total_blocks, NEURONS)
            )
        assert np.all(rate_blocks > 0.0)
        # raw 20-ms spike-count bins -> sum / 0.1 seconds recovers rate_blocks
        neural = np.repeat(rate_blocks * SIGNED.BIN_SECONDS, SIGNED.BLOCK_BINS, axis=0)
        velocity = np.repeat(velocity_blocks, SIGNED.BLOCK_BINS, axis=0)
        trial_num = np.repeat(
            np.asarray((1.0, 2.0)), BLOCKS_PER_TRIAL * SIGNED.BLOCK_BINS
        )
        session = f"ses-{date}T12{date_index:04d}"
        records[session] = SIGNED.record_from_arrays(
            session_name=session,
            neural=neural,
            velocity=velocity,
            eval_mask=np.ones(neural.shape[0], dtype=bool),
            trial_num=trial_num,
            input_sha256=f"synthetic-{law}-{date_index}",
        )
    return records


def _date_rows(*, norm: float = 0.7, gain: float = 0.1, margin: float = 0.05):
    return [
        {
            "status": "defined",
            "date_invariant_w_norm_spearman": norm,
            "date_correct_symmetric_cross_trial_gain": gain,
            "date_correct_minus_null_q95": margin,
        }
        for _ in DATES
    ]


def test_fit_side_null_rotation_never_exchanges_rows_between_trials():
    trial_one_scores = np.arange(20, dtype=np.float64).reshape(10, 2)
    trial_two_scores = 1000.0 + np.arange(20, dtype=np.float64).reshape(10, 2)
    rotated_one, offset_one = AUDIT.fit_side_rotated_scores(
        trial_one_scores,
        session_name="ses-19250101T120000",
        trial_number=1.0,
        replicate=0,
    )
    rotated_two, offset_two = AUDIT.fit_side_rotated_scores(
        trial_two_scores,
        session_name="ses-19250101T120000",
        trial_number=2.0,
        replicate=0,
    )

    assert offset_one != 0 and offset_two != 0
    np.testing.assert_array_equal(np.sort(rotated_one, axis=0), np.sort(trial_one_scores, axis=0))
    np.testing.assert_array_equal(np.sort(rotated_two, axis=0), np.sort(trial_two_scores, axis=0))
    assert float(rotated_one.max()) < 1000.0
    assert float(rotated_two.min()) >= 1000.0


def test_correct_cross_trial_synthetic_law_passes_all_i0_clauses():
    receipt = AUDIT.run_audit(_records(law="correct"))

    assert receipt["i0_gate"]["pass"] is True
    assert receipt["status"] == "PASS_CPU_I0_FEASIBILITY__NO_GPU_AUTHORIZATION"
    assert receipt["i0_gate"]["all_six_dates_defined"] is True
    assert all(clause["pass"] is True for clause in receipt["i0_gate"]["subgates"].values())
    for date_row in receipt["date_lodo"]:
        assert date_row["date_correct_symmetric_cross_trial_gain"] > 0.0
        assert date_row["date_correct_minus_null_q95"] > 0.0
        assert len(date_row["date_null_gain_replicates"]) == AUDIT.NULL_REPLICATES
        assert date_row["source_only_normalizer"]["used_by_i0_gate"] is False


def test_label_independent_synthetic_data_fails_i0():
    receipt = AUDIT.run_audit(_records(law="independent"))

    assert receipt["i0_gate"]["pass"] is False
    assert receipt["status"] == "STOP_CPU_I0_FEASIBILITY_FAILED__NO_GPU_AUTHORIZATION"
    # The test is intentionally not tied to one particular failure clause:
    # label-independent data can fail stability, transfer, or null separation.
    assert any(
        clause["pass"] is False for clause in receipt["i0_gate"]["subgates"].values()
    )


def test_outer_target_mutation_cannot_change_source_plan_or_normalizer():
    records = _records(law="correct")
    outer_date = "19250115"
    plan_before = SIGNED.fit_source_plan(records, outer_date)
    partition = next(
        item for item in SIGNED.date_lodo_partitions(records) if item["outer_date"] == outer_date
    )
    normalizer_before = AUDIT._source_only_normalizer(
        [records[name] for name in partition["source_sessions"]], plan_before
    )
    target_name = partition["target_sessions"][0]
    target = records[target_name]
    mutated_trials = tuple(
        replace(trial, rates=trial.rates + 100.0, velocity=trial.velocity * -9.0 + 3.0)
        for trial in target.trials
    )
    mutated = dict(records)
    mutated[target_name] = replace(target, trials=mutated_trials)
    plan_after = SIGNED.fit_source_plan(mutated, outer_date)
    normalizer_after = AUDIT._source_only_normalizer(
        [mutated[name] for name in partition["source_sessions"]], plan_after
    )

    assert plan_before.plan_sha256 == plan_after.plan_sha256
    np.testing.assert_allclose(plan_before.basis.mean, plan_after.basis.mean)
    np.testing.assert_allclose(plan_before.basis.components, plan_after.basis.components)
    assert normalizer_before["normalizer_sha256"] == normalizer_after["normalizer_sha256"]
    assert normalizer_before["source_descriptor_hashes"] == normalizer_after["source_descriptor_hashes"]


def test_i0_is_a_conjunction_not_an_average_or_single_metric_gate():
    passing = AUDIT.i0_gate(_date_rows())
    assert passing["pass"] is True

    # Five dates have strong norm stability and correct gain, but only three
    # exceed the null q95.  The conjunction must stop the route.
    failed_rows = _date_rows()
    for index in (0, 1, 2):
        failed_rows[index]["date_correct_minus_null_q95"] = -0.001
    failed = AUDIT.i0_gate(failed_rows)
    assert failed["pass"] is False
    assert failed["subgates"]["invariant_w_norm_spearman"]["pass"] is True
    assert failed["subgates"]["correct_symmetric_cross_trial_gain"]["pass"] is True
    assert failed["subgates"]["correct_gain_above_fit_side_rotation_null_q95"]["pass"] is False
